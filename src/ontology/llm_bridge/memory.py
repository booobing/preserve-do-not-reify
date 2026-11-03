"""HypergraphMemory: encoder outputs -> key/value memory for cross-attention (contract C10)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def retrieve_topk(query_emb: Tensor, fact_emb: Tensor, k: int) -> Tensor:
    """Indices [Q, k] of the top-k facts per query by cosine (inputs are L2-normalized)."""
    if fact_emb.shape[0] == 0 or k <= 0:
        return torch.zeros((query_emb.shape[0], 0), dtype=torch.long, device=query_emb.device)
    k = min(int(k), int(fact_emb.shape[0]))
    scores = query_emb @ fact_emb.t()
    return scores.topk(k, dim=-1).indices


class HypergraphMemory(nn.Module):
    """Projects hyperedge and node embeddings into the decoder width and tags their kind.

    memory = LN([W_e z_e + type_e ; W_v z_v + type_v])   shape [B, E+V, d_llm]
    mask   = True for valid slots                          shape [B, E+V]
    """

    def __init__(self, d_hnn: int, d_llm: int) -> None:
        super().__init__()
        self.d_hnn = int(d_hnn)
        self.d_llm = int(d_llm)
        self.edge_proj = nn.Linear(d_hnn, d_llm)
        self.node_proj = nn.Linear(d_hnn, d_llm)
        self.kind_emb = nn.Embedding(2, d_llm)  # 0 = hyperedge slot, 1 = node slot
        self.norm = nn.LayerNorm(d_llm)

    def build(
        self, node_emb: Tensor, edge_emb: Tensor, block_sizes: Sequence[int] | None = None
    ) -> tuple[Tensor, Tensor]:
        """`[all edges ; all distinct nodes]`, or per-fact blocks when `block_sizes` is given.

        The flat layout drops which participant belongs to which fact: a node in three facts gets
        one slot and nothing says which. The v5 probe priced that omission - the same names cost
        0.313 written flat and 0.400 written one line per fact - so the grouped layout emits
        `[edge_i, its participants...]` per fact and repeats a shared participant in every block it
        belongs to, exactly as the text rendering repeats the name (charter v6 §3-a).
        """
        # The encoder trains in fp32 while the frozen decoder runs in fp16/bf16, so the memory
        # casts to its own parameter dtype rather than requiring the caller to match it.
        device = self.edge_proj.weight.device
        dtype = self.edge_proj.weight.dtype
        edges = self.edge_proj(edge_emb.to(device=device, dtype=dtype)) + self.kind_emb.weight[0]
        nodes = self.node_proj(node_emb.to(device=device, dtype=dtype)) + self.kind_emb.weight[1]
        if block_sizes is None:
            slots = torch.cat([edges, nodes], dim=0)
        else:
            sizes = [int(n) for n in block_sizes]
            if len(sizes) != edges.shape[0]:
                raise ValueError(f"block_sizes has {len(sizes)} entries for {edges.shape[0]} edges")
            if sum(sizes) != nodes.shape[0]:
                raise ValueError(f"block_sizes sums to {sum(sizes)} for {nodes.shape[0]} node rows")
            rows: list[Tensor] = []
            cursor = 0
            for index, size in enumerate(sizes):
                rows.append(edges[index : index + 1])
                rows.append(nodes[cursor : cursor + size])
                cursor += size
            slots = torch.cat(rows, dim=0)
        memory = self.norm(slots).unsqueeze(0)
        mask = torch.ones((1, memory.shape[1]), dtype=torch.bool, device=device)
        return memory, mask

    def node_slots(self, node_emb: Tensor) -> Tensor:
        """The node half of the memory, before concatenation - what vocabulary anchoring aims.

        Anchoring targets the node slots only. Edge slots stand for whole facts and have no name in
        the decoder's vocabulary to be pointed at.
        """
        device, dtype = self.node_proj.weight.device, self.node_proj.weight.dtype
        nodes = self.node_proj(node_emb.to(device=device, dtype=dtype)) + self.kind_emb.weight[1]
        return self.norm(nodes)

    def build_batch(self, items: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Pad a list of (node_emb, edge_emb) pairs into [B, L_max, d_llm] with a validity mask."""
        if not items:
            raise ValueError("build_batch needs at least one (node_emb, edge_emb) pair")
        built = [self.build(n, e) for n, e in items]
        length = max(m.shape[1] for m, _ in built)
        device, dtype = built[0][0].device, built[0][0].dtype
        memory = torch.zeros((len(built), length, self.d_llm), device=device, dtype=dtype)
        mask = torch.zeros((len(built), length), dtype=torch.bool, device=device)
        for row, (m, _) in enumerate(built):
            memory[row, : m.shape[1]] = m[0]
            mask[row, : m.shape[1]] = True
        return memory, mask


class SpokeMemory(nn.Module):
    """One memory slot per membership `(fact, participant, role)` (charter-hnn-llm-bridge-v7 §3-a).

        slot(e, v, r) = LN( W_2 GELU( W_1 [z_e ; z_v ; E_role[r]] ) )

    The encoder computes on the incidence explicitly, but `HypergraphMemory` hands the decoder
    unconnected fact and participant vectors, so which participant fills which role of which fact
    never crosses the channel. Here each slot *is* one connection. Answering "which participant fills
    role r of the fact these constraints describe" becomes a single cross-attention lookup instead of
    a fact lookup followed by a participant lookup - the step the ceiling diagnostic could not make.
    A participant shared by two facts gets two different slots, because the fact vector is fused in.
    """

    def __init__(self, d_hnn: int, d_llm: int, num_roles: int, *, hidden: int | None = None) -> None:
        super().__init__()
        self.d_hnn = int(d_hnn)
        self.d_llm = int(d_llm)
        width = int(hidden or 4 * d_hnn)
        self.role_emb = nn.Embedding(int(num_roles), d_hnn)
        self.fuse_in = nn.Linear(3 * d_hnn, width)
        self.fuse_out = nn.Linear(width, d_llm)
        self.norm = nn.LayerNorm(d_llm)

    def build(self, node_emb: Tensor, edge_emb: Tensor, role_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Row i of all three inputs is membership i; returns ([1, M, d_llm], [1, M] mask)."""
        if not node_emb.shape[0] == edge_emb.shape[0] == role_ids.shape[0]:
            raise ValueError(f"membership rows disagree: nodes {node_emb.shape[0]}, "
                             f"edges {edge_emb.shape[0]}, roles {role_ids.shape[0]}")
        device, dtype = self.fuse_in.weight.device, self.fuse_in.weight.dtype
        roles = self.role_emb(role_ids.to(device=device, dtype=torch.long))
        joined = torch.cat([edge_emb.to(device=device, dtype=dtype),
                            node_emb.to(device=device, dtype=dtype), roles], dim=-1)
        slots = self.norm(self.fuse_out(torch.nn.functional.gelu(self.fuse_in(joined))))
        memory = slots.unsqueeze(0)
        mask = torch.ones((1, memory.shape[1]), dtype=torch.bool, device=device)
        return memory, mask
