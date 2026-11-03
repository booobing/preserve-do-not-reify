"""Gated cross-attention adapter and the two injection paths (contract C10).

    x_out = x + tanh(alpha) * CrossAttn(Q = LN(x) W_q, K = M W_k, V = M W_v)      alpha_0 = 0

At alpha = 0 the adapter is an exact identity, so injecting it into a frozen decoder cannot change the
decoder's outputs until training moves alpha. Two ways to attach it:

1. `inject_into_decoder(toy_decoder, layer_ids, ...)` fills the cross slot of this repo's ToyDecoder.
2. `attach_cross_attention_hooks(layers, ...)` registers forward hooks on arbitrary decoder layers
   (the path for Hugging Face models in P5): the hook adds the adapter output to the layer's hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GatedCrossAttention(nn.Module):
    """Cross-attention adapter with two ways of starting as an identity.

    `gated=True` is the v1 configuration: a scalar `tanh(alpha)` initialised at 0. It is an exact
    identity, but it also scales every attend parameter's gradient by that same zero, so on the
    first step only the scalar can learn - and the scalar only grows if the randomly initialised
    attend output already helps, which it does not. The v1 run sat in that chicken-and-egg trap for
    a thousand steps (`docs/hnn-llm-bridge-readpath-rca-v1-results.md`).

    `gated=False, zero_init_output=True` is the v2 repair: no scalar, and the output projection
    starts at zero instead. The adapter is still an exact identity at init, but now the first
    backward puts gradient on a whole matrix, and q/k/v unlock as soon as it leaves zero.
    """

    def __init__(
        self,
        d_model: int,
        d_memory: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        gated: bool = True,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_memory, d_model)
        self.v_proj = nn.Linear(d_memory, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)
        self.gated = bool(gated)
        self.alpha = nn.Parameter(torch.zeros(()))
        if zero_init_output:
            with torch.no_grad():
                self.o_proj.weight.zero_()
                if self.o_proj.bias is not None:
                    self.o_proj.bias.zero_()

    @property
    def gate(self) -> Tensor:
        return torch.tanh(self.alpha)

    def attend(
        self, x: Tensor, memory: Tensor, mask: Tensor | None, *, return_probs: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Attend over the memory. With `return_probs` the softmax is done explicitly.

        The alignment loss has to see where attention lands, and the fused kernel does not hand
        that back. The explicit path is the same function - a test pins the two to the same
        numbers - so shaping the explicit probabilities shapes what evaluation actually uses.
        """
        batch, seq, _ = x.shape
        length = memory.shape[1]
        # The adapter is trained while the decoder around it is frozen at a lower precision, so the
        # seam casts in both directions: everything is attended in the adapter's own dtype and the
        # result goes back out in the hidden state's dtype.
        dtype = self.k_proj.weight.dtype
        memory = memory.to(dtype=dtype)
        x = x.to(dtype=dtype)
        q = self.q_proj(self.norm(x)).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].to(torch.bool)  # [B,1,1,L]
        probs: Tensor | None = None
        if return_probs:
            scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            probs = torch.nan_to_num(scores.softmax(dim=-1))
            out = probs @ v
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
            )
        out = out.transpose(1, 2).reshape(batch, seq, self.d_model)
        out = self.o_proj(out)
        if mask is not None:
            # rows with no valid memory slot attend to nothing -> contribute 0 (avoids NaN * 0)
            valid_rows = mask.any(dim=-1).view(batch, 1, 1).to(out.dtype)
            out = torch.nan_to_num(out) * valid_rows
        return (out, probs) if return_probs else out

    def forward(self, x: Tensor, memory: Tensor, mask: Tensor | None = None) -> Tensor:
        if memory.shape[0] == 1 and x.shape[0] > 1:
            memory = memory.expand(x.shape[0], -1, -1)
            if mask is not None:
                mask = mask.expand(x.shape[0], -1)
        if not self.gated:
            # No scalar to throttle the gradient; the identity at init comes from o_proj being zero.
            return x + self.attend(x, memory, mask).to(dtype=x.dtype)
        gate = self.gate
        if float(gate.detach()) == 0.0 and not torch.is_grad_enabled():
            return x  # exact identity in eval mode at alpha = 0
        return x + (gate * self.attend(x, memory, mask)).to(dtype=x.dtype)


# ---- toy decoder -------------------------------------------------------------------------------


class ToyDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.cross: GatedCrossAttention | None = None  # injection slot

    def forward(self, x: Tensor, memory: Tensor | None = None, memory_mask: Tensor | None = None) -> Tensor:
        seq = x.shape[1]
        causal = torch.triu(torch.ones((seq, seq), dtype=torch.bool, device=x.device), diagonal=1)
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=causal, need_weights=False)[0]
        if self.cross is not None and memory is not None:
            x = self.cross(x, memory, memory_mask)
        return x + self.ffn(self.norm2(x))


class ToyDecoder(nn.Module):
    """Small causal decoder used to verify the bridge contract offline."""

    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, max_len: int = 64) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(ToyDecoderBlock(d_model, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def get_input_embeddings(self) -> nn.Module:
        """Same accessor the Hugging Face models expose, so the soft-prompt arm has one code path."""
        return self.tok

    def forward(
        self,
        input_ids: Tensor | None = None,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        *,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.tok(input_ids)
        seq = inputs_embeds.shape[1]
        x = inputs_embeds + self.pos(torch.arange(seq, device=inputs_embeds.device))
        for layer in self.layers:
            x = layer(x, memory, memory_mask)
        return self.head(self.norm(x))


def inject_into_decoder(
    decoder: ToyDecoder, layer_ids: Iterable[int], *, d_memory: int, n_heads: int | None = None
) -> list[GatedCrossAttention]:
    """Fill the cross slots of the given layers; freeze everything else. Returns the adapters."""
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    adapters: list[GatedCrossAttention] = []
    for layer_id in layer_ids:
        block = decoder.layers[int(layer_id)]
        heads = n_heads or block.attn.num_heads
        block.cross = GatedCrossAttention(decoder.d_model, d_memory, heads)
        adapters.append(block.cross)
    return adapters


# ---- hook path (generic decoder layers, e.g. Hugging Face) -------------------------------------


@dataclass
class BridgeContext:
    """Holds the memory for the current forward; hooks read it (set to None to disable).

    A single tensor reaches every hooked layer. A list holds one memory per hooked layer, in hook
    order - the ceiling diagnostic feeds each layer the decoder's own state at that depth.
    """

    memory: Tensor | list[Tensor] | None = None
    mask: Tensor | list[Tensor] | None = None
    handles: list = field(default_factory=list)

    def set(self, memory: Tensor | list[Tensor] | None, mask: Tensor | list[Tensor] | None = None) -> None:
        self.memory = memory
        self.mask = mask

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def attach_cross_attention_hooks(
    layers: Sequence[nn.Module],
    adapters: Sequence[GatedCrossAttention],
    context: BridgeContext | None = None,
) -> BridgeContext:
    """Register forward hooks that add `adapter(hidden, memory)` to each layer's output hidden state.

    Works for layers that return either a Tensor or a tuple whose first element is the hidden state
    (the Hugging Face decoder-layer convention). Layers' own parameters are frozen.
    """
    if len(layers) != len(adapters):
        raise ValueError("one adapter per layer is required")
    ctx = context or BridgeContext()

    def make_hook(adapter: GatedCrossAttention, index: int):
        def hook(_module: nn.Module, _inputs, output):
            if ctx.memory is None:
                return output
            memory, mask = ctx.memory, ctx.mask
            if isinstance(memory, (list, tuple)):
                memory = memory[index]
                if isinstance(mask, (list, tuple)):
                    mask = mask[index]
                elif mask is not None:
                    raise ValueError("a per-layer memory list needs a per-layer mask list")
            if isinstance(output, tuple):
                hidden = adapter(output[0], memory, mask)
                return (hidden,) + tuple(output[1:])
            return adapter(output, memory, mask)

        return hook

    for index, (layer, adapter) in enumerate(zip(layers, adapters, strict=True)):
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        ctx.handles.append(layer.register_forward_hook(make_hook(adapter, index)))
    return ctx


def adapter_parameters(adapters: Iterable[GatedCrossAttention]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for adapter in adapters:
        params.extend(adapter.parameters())
    return params
