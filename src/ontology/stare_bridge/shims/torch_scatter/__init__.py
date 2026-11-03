"""Native-torch stand-in for the three ``torch_scatter`` functions StarE uses.

torch_scatter ships compiled wheels only for specific torch/CUDA/OS triples and has none for the
torch this repo pins on Windows. StarE calls ``scatter_add`` (normalisation, qualifier
coalescing), ``scatter_mean`` (``QUAL_N == "mean"``) and ``scatter_max`` (attention softmax,
off by default). All three reduce to ``index_add_`` / ``scatter_reduce`` along dimension 0,
which is the only dimension StarE scatters over.

The signature mirrors torch_scatter 2.x ``(src, index, dim=-1, out=None, dim_size=None)`` and
tolerates the legacy sixth positional ``fill_value`` that one unused upstream helper passes.
``out`` is not supported because upstream never passes it.
"""

from __future__ import annotations

import torch
from torch import Tensor

__version__ = "0.0.0+hypergraph-eggroll-shim"


def _dim_size(index: Tensor, dim_size: int | None) -> int:
    if dim_size is not None:
        return int(dim_size)
    return int(index.max().item()) + 1 if index.numel() else 0


def _check(src: Tensor, index: Tensor, dim: int, out: Tensor | None) -> None:
    if out is not None:
        raise NotImplementedError("torch_scatter shim: `out` is not supported")
    if dim not in (0, -src.dim()):
        raise NotImplementedError("torch_scatter shim: only dim=0 (the dimension StarE uses)")
    if index.dim() != 1 or index.shape[0] != src.shape[0]:
        raise ValueError("torch_scatter shim: index must be 1-D with len(index) == src.shape[0]")


def scatter_add(
    src: Tensor, index: Tensor, dim: int = -1, out: Tensor | None = None, dim_size: int | None = None, fill_value=None
) -> Tensor:
    _check(src, index, dim, out)
    n = _dim_size(index, dim_size)
    result = torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    if src.numel() == 0:
        return result
    return result.index_add_(0, index.long(), src)


def scatter_mean(
    src: Tensor, index: Tensor, dim: int = -1, out: Tensor | None = None, dim_size: int | None = None, fill_value=None
) -> Tensor:
    _check(src, index, dim, out)
    total = scatter_add(src, index, dim, None, dim_size)
    count = scatter_add(torch.ones((src.shape[0],), dtype=src.dtype, device=src.device), index, dim, None, dim_size)
    return total / count.clamp_min(1).view((-1,) + (1,) * (src.dim() - 1))


def scatter_max(
    src: Tensor, index: Tensor, dim: int = -1, out: Tensor | None = None, dim_size: int | None = None, fill_value=None
) -> tuple[Tensor, None]:
    """Group-wise max; empty groups give 0 (torch_scatter fills them with the lowest value and
    upstream then resets those to 0). The argmax half of the tuple is not computed."""
    _check(src, index, dim, out)
    n = _dim_size(index, dim_size)
    result = torch.full((n,) + tuple(src.shape[1:]), float("-inf"), dtype=src.dtype, device=src.device)
    if src.numel():
        expanded = index.long().view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
        result = result.scatter_reduce(0, expanded, src, reduce="amax", include_self=True)
    result[result == float("-inf")] = 0
    return result, None


__all__ = ["scatter_add", "scatter_max", "scatter_mean"]
