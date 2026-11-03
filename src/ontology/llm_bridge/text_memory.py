"""Decoder-native memory for the channel ceiling diagnostic (charter-hnn-llm-bridge-ceiling-diagnostic).

Six bridge cycles injected HNN embeddings and never beat the `names_only` text. That left two
explanations standing: the cross-attention channel cannot carry composable structure at this budget,
or the HNN embeddings do not hold the content in a form the decoder can use. Every memory arm so far
changed the second half, so nothing could separate them.

These arms keep the v5_named channel and budget and replace only what goes into it. The memory is
built from exactly the text `names_only` reads, in two forms the decoder already speaks:

    hidden     the frozen decoder's own output at each injection layer, reading the text with the
               memory switched off - order and per-fact grouping already folded in by its attention
    token_emb  the input embeddings of the same tokens - the decoder's vocabulary space, but an
               orderless set, because cross-attention memory carries no positions
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

CONTEXT_PREFIX = "Facts:\n"
MODES: tuple[str, ...] = ("hidden", "token_emb")


class TextMemory(nn.Module):
    """One LayerNorm per injection layer and nothing else.

    The adapters' key/value projections already map the memory, so a projection here would only add
    a second linear map in front of the first. Keeping the module this small also means a ceiling
    arm cannot win on capacity: it trains fewer memory parameters than HypergraphMemory does.
    """

    def __init__(self, d_llm: int, n_layers: int) -> None:
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(d_llm) for _ in range(int(n_layers)))

    def build(self, per_layer: Sequence[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """[T, d] per layer -> ([1, T, d] memories, [1, T] masks), normalised in the module's dtype."""
        if len(per_layer) != len(self.norms):
            raise ValueError(f"got {len(per_layer)} layer memories for {len(self.norms)} norms")
        device, dtype = self.norms[0].weight.device, self.norms[0].weight.dtype
        memories, masks = [], []
        for norm, rows in zip(self.norms, per_layer, strict=True):
            memory = norm(rows.to(device=device, dtype=dtype)).unsqueeze(0)
            memories.append(memory)
            masks.append(torch.ones((1, memory.shape[1]), dtype=torch.bool, device=device))
        return memories, masks


def context_ids(tokenizer, context: str, device) -> tuple[Tensor, int]:
    """(ids [1, P+T], P): the prefix and the context tokenised separately, then joined.

    Separate tokenisation makes the boundary exact, so the prefix positions - including the first
    token, whose hidden state is the decoder's attention sink - can be dropped from the memory.
    """
    prefix = tokenizer(CONTEXT_PREFIX, return_tensors="pt").input_ids
    body = tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids
    if body.shape[1] == 0:
        raise ValueError("context tokenised to nothing; a ceiling memory needs at least one slot")
    return torch.cat([prefix, body], dim=1).to(device), int(prefix.shape[1])


@torch.no_grad()
def encode_context(
    model: nn.Module, layers: Sequence[nn.Module], ids: Tensor, prefix_len: int, mode: str
) -> list[Tensor]:
    """Per-layer [T, d] rows for the context positions of `ids`.

    The caller must have switched the bridge memory off: the capture hooks run after the adapter
    hooks, so with a memory installed they would record the injected state instead of the decoder's.
    """
    if mode == "token_emb":
        rows = model.get_input_embeddings()(ids)[0, prefix_len:]
        return [rows for _ in layers]
    if mode != "hidden":
        raise KeyError(f"unknown text memory mode {mode!r}; expected one of {MODES}")
    captured: list[Tensor | None] = [None] * len(layers)

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[0, prefix_len:].detach()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        model(ids)
    finally:
        for handle in handles:
            handle.remove()
    if any(rows is None for rows in captured):
        raise RuntimeError("a capture hook did not fire; the layer list does not belong to this model")
    return [rows for rows in captured if rows is not None]
