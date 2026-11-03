"""Attach the hypergraph bridge to a pinned Hugging Face decoder (charter-hnn-llm-bridge-v1, P5).

The decoder is loaded frozen at a fixed revision and never fine-tuned. Gated cross-attention
adapters are hooked onto selected decoder layers, so at gate 0 the model reproduces its own logits
exactly and only the adapters plus the memory projection carry gradient.

Runs are offline by default: `HF_HUB_OFFLINE=1` is set unless the caller opts out, so a missing
local snapshot fails loudly instead of silently downloading during a judged run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .gated_cross_attention import BridgeContext, GatedCrossAttention, attach_cross_attention_hooks
from .memory import HypergraphMemory

# Frozen in docs/charter-hnn-llm-bridge-v1.md ("모델 선택"). Do not change without a new charter.
BRIDGE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BRIDGE_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TEXT_TOWER_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_TOWER_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"

# 25% / 50% / 75% of the 28 decoder layers, per the frozen charter.
DEFAULT_INJECTION_LAYERS: tuple[int, ...] = (7, 14, 21)


def _set_offline(offline: bool) -> None:
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def decoder_layers(model: nn.Module) -> list[nn.Module]:
    """The list of transformer blocks, across the layouts transformers uses for causal LMs."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "model.decoder.layers"):
        node: Any = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            return list(node)
    raise AttributeError(f"could not locate decoder layers on {type(model).__name__}")


@dataclass
class BridgedDecoder:
    """A frozen decoder with hypergraph adapters attached."""

    model: nn.Module
    tokenizer: Any
    memory: HypergraphMemory
    adapters: list[GatedCrossAttention]
    context: BridgeContext
    layer_ids: tuple[int, ...]
    model_id: str = BRIDGE_MODEL_ID
    revision: str = BRIDGE_MODEL_REVISION
    soft: Any = None  # SoftPromptEncoder for the hnn_soft arm; unused by the others
    lora: list[Any] = field(default_factory=list)  # LoRALinear wrappers (v6 lora arm only)
    # Ceiling diagnostic only: the per-layer norms, the hooked decoder blocks the memory is read
    # from, and the frozen encodings keyed by (mode, text). The blocks are a plain list so the
    # frozen decoder never joins this bridge's parameters or saved state.
    text_memory: Any = None
    text_layers: list[Any] = field(default_factory=list)
    text_cache: dict[Any, Any] = field(default_factory=dict)
    spokes: Any = None  # SpokeMemory for the v7 connection-slot arm; unused by the others
    _meta: dict[str, Any] = field(default_factory=dict)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameters(self):
        yield from self.memory.parameters()
        for adapter in self.adapters:
            yield from adapter.parameters()
        if self.soft is not None:
            yield from self.soft.parameters()
        for module in self.lora:
            yield module.lora_a
            yield module.lora_b
        if self.text_memory is not None:
            yield from self.text_memory.parameters()
        if self.spokes is not None:
            yield from self.spokes.parameters()

    def set_spoke_memory(self, node_emb: Tensor, edge_emb: Tensor, role_ids: Tensor) -> None:
        """Install one slot per (fact, participant, role) membership (charter v7 §3-a)."""
        if self.spokes is None:
            raise RuntimeError("this bridge was not built for the connection-slot arm")
        self.context.set(*self.spokes.build(node_emb, edge_emb, role_ids))

    def set_text_memory(self, text: str, mode: str) -> None:
        """Install the decoder-native memory of `text` (ceiling diagnostic), one memory per layer.

        The encoding is a function of the frozen decoder and the text alone, so it is computed once
        per text with the memory switched off and kept on the CPU; only the norms are applied fresh.
        """
        from .text_memory import context_ids, encode_context

        if self.text_memory is None or not self.text_layers:
            raise RuntimeError("this bridge was not built for a ceiling arm")
        key = (mode, text)
        rows = self.text_cache.get(key)
        if rows is None:
            self.context.set(None)
            device = next(self.text_memory.parameters()).device
            ids, prefix_len = context_ids(self.tokenizer, text, device)
            rows = [r.to("cpu") for r in encode_context(self.model, self.text_layers, ids, prefix_len, mode)]
            self.text_cache[key] = rows
        self.context.set(*self.text_memory.build(rows))

    def set_memory(
        self,
        node_emb: Tensor | None,
        edge_emb: Tensor | None = None,
        block_sizes: Sequence[int] | None = None,
    ) -> None:
        """Install the hypergraph memory for the next forward, or clear it with None.

        `block_sizes` selects the per-fact block layout (charter v6 §3-a); without it the memory
        keeps the flat `[edges ; distinct nodes]` order every earlier arm was judged on.
        """
        if node_emb is None:
            self.context.set(None)
            return
        if edge_emb is None:
            raise ValueError("edge_emb is required when node_emb is given")
        self.context.set(*self.memory.build(node_emb, edge_emb, block_sizes))

    def open_gates(self, value: float) -> None:
        with torch.no_grad():
            for adapter in self.adapters:
                adapter.alpha.fill_(float(value))

    def gates(self) -> list[float]:
        return [float(adapter.gate.detach()) for adapter in self.adapters]

    def detach(self) -> None:
        self.context.remove()


def load_bridged_decoder(
    *,
    d_hnn: int,
    model_id: str = BRIDGE_MODEL_ID,
    revision: str = BRIDGE_MODEL_REVISION,
    layer_ids: Sequence[int] = DEFAULT_INJECTION_LAYERS,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    adapter_dtype: torch.dtype = torch.float32,
    offline: bool = True,
    d_memory: int | None = None,
    gated: bool = True,
    zero_init_output: bool = False,
) -> BridgedDecoder:
    _set_offline(offline)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    layers = decoder_layers(model)
    for layer_id in layer_ids:
        if not 0 <= layer_id < len(layers):
            raise IndexError(f"layer {layer_id} out of range for a {len(layers)}-layer decoder")

    d_model = int(model.config.hidden_size)
    memory_width = int(d_memory or d_model // 2)
    # The decoder is frozen, so it can stay in fp16 and save the VRAM. The adapters are the only
    # thing an optimiser touches, and stepping them in fp16 is what AdamW cannot survive: its
    # eps of 1e-8 is below fp16's smallest normal value, so the update divides by a rounded-down
    # zero and the whole adapter turns to NaN within a few steps. They are trained in fp32 and the
    # cross-attention casts at the seam.
    memory = HypergraphMemory(d_hnn=d_hnn, d_llm=memory_width).to(device=device, dtype=adapter_dtype)
    n_heads = int(getattr(model.config, "num_attention_heads", 8))
    adapters = [
        GatedCrossAttention(d_model, memory_width, n_heads,
                            gated=gated, zero_init_output=zero_init_output).to(device=device, dtype=adapter_dtype)
        for _ in layer_ids
    ]
    context = attach_cross_attention_hooks([layers[i] for i in layer_ids], adapters)
    return BridgedDecoder(
        model=model,
        tokenizer=tokenizer,
        memory=memory,
        adapters=adapters,
        context=context,
        layer_ids=tuple(int(i) for i in layer_ids),
        model_id=model_id,
        revision=revision,
        _meta={
            "num_layers": len(layers),
            "d_model": d_model,
            "d_memory": memory_width,
            "dtype": str(dtype),
            "adapter_dtype": str(adapter_dtype),
        },
    )


def load_text_tower(
    *,
    model_id: str = TEXT_TOWER_MODEL_ID,
    revision: str = TEXT_TOWER_REVISION,
    device: str | None = None,
    offline: bool = True,
):
    """Frozen multilingual sentence encoder used as the v2 text tower (charter v2 §V2-C)."""
    _set_offline(offline)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, revision=revision, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model
