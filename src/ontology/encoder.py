"""Role-aware hypergraph encoder (contract C5, design §4).

    h_v^0 = LN(E_type[type(v)] + W_x t_v)          h_e^0 = E_rel[rel(e)]
    node -> edge:  m_e = mean_{(v,e,r)} act(phi1(h_v + E_role[r]));  h_e = LN(h_e + phi2([h_e; m_e]))
    edge -> node:  m_v = mean_{(v,e,r)} act(phi3(h_e + E_role[r]));  h_v = LN(h_v + phi4([h_v; m_v]))
    z_v = normalize(W_v h_v), z_e = normalize(W_e h_e)

The same module consumes the native n-ary view (`hyper` arm) and the reified binary view
(`reified_binary` arm); only the IncidenceBatch differs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .hypergraph import IncidenceBatch


def _scatter_mean(values: Tensor, index: Tensor, size: int) -> Tensor:
    out = torch.zeros((size, values.shape[-1]), dtype=values.dtype, device=values.device)
    if values.numel() == 0:
        return out
    out.index_add_(0, index, values)
    degree = torch.zeros((size,), dtype=values.dtype, device=values.device)
    degree.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return out / degree.clamp_min(1.0).unsqueeze(-1)


class RoleAwareMessageLayer(nn.Module):
    def __init__(self, d: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.phi1 = nn.Linear(d, d)
        self.phi2 = nn.Linear(2 * d, d)
        self.phi3 = nn.Linear(d, d)
        self.phi4 = nn.Linear(2 * d, d)
        self.norm_e = nn.LayerNorm(d)
        self.norm_v = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_v: Tensor,
        h_e: Tensor,
        role_emb: Tensor,
        node_idx: Tensor,
        edge_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        num_nodes, num_edges = h_v.shape[0], h_e.shape[0]
        msg_e = F.gelu(self.phi1(h_v[node_idx] + role_emb))
        m_e = _scatter_mean(msg_e, edge_idx, num_edges)
        h_e = self.norm_e(h_e + self.dropout(self.phi2(torch.cat([h_e, m_e], dim=-1))))
        msg_v = F.gelu(self.phi3(h_e[edge_idx] + role_emb))
        m_v = _scatter_mean(msg_v, node_idx, num_nodes)
        h_v = self.norm_v(h_v + self.dropout(self.phi4(torch.cat([h_v, m_v], dim=-1))))
        return h_v, h_e


class RoleAwareHypergraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_node_types: int,
        num_edge_types: int,
        num_roles: int,
        d: int = 64,
        layers: int = 2,
        text_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.d = int(d)
        self.num_layers = int(layers)
        self.type_emb = nn.Embedding(num_node_types, d)
        self.rel_emb = nn.Embedding(num_edge_types, d)
        self.role_emb = nn.Embedding(num_roles, d)
        self.text_proj = nn.Linear(text_dim, d)
        self.input_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList(RoleAwareMessageLayer(d, dropout=dropout) for _ in range(layers))
        self.node_out = nn.Linear(d, d)
        self.edge_out = nn.Linear(d, d)
        self.completion_out = nn.Linear(2 * d, d)

    def forward(self, batch: IncidenceBatch, node_text_emb: Tensor | None = None) -> dict[str, Tensor]:
        device = self.type_emb.weight.device
        num_nodes, num_edges = batch.num_nodes, batch.num_edges
        if num_nodes == 0:
            zeros_v = torch.zeros((0, self.d), device=device)
            zeros_e = torch.zeros((num_edges, self.d), device=device)
            return {"node_emb": zeros_v, "edge_emb": zeros_e, "node_hidden": zeros_v, "edge_hidden": zeros_e}
        h_v = self.type_emb(batch.node_type_ids.to(device))
        if node_text_emb is not None:
            if node_text_emb.shape[0] != num_nodes:
                raise ValueError(f"node_text_emb has {node_text_emb.shape[0]} rows, expected {num_nodes}")
            h_v = h_v + self.text_proj(node_text_emb.to(device))
        h_v = self.input_norm(h_v)
        h_e = self.rel_emb(batch.edge_type_ids.to(device)) if num_edges > 0 else torch.zeros((0, self.d), device=device)
        node_idx = batch.incidence[:, 0].to(device)
        edge_idx = batch.incidence[:, 1].to(device)
        role_emb = self.role_emb(batch.role_ids.to(device))
        for layer in self.layers:
            h_v, h_e = layer(h_v, h_e, role_emb, node_idx, edge_idx)
        node_emb = F.normalize(self.node_out(h_v), dim=-1)
        edge_emb = F.normalize(self.edge_out(h_e), dim=-1) if num_edges > 0 else h_e
        return {"node_emb": node_emb, "edge_emb": edge_emb, "node_hidden": h_v, "edge_hidden": h_e}

    def completion_query(self, hidden: Tensor, role_ids: Tensor) -> Tensor:
        """Representation of 'this fact with role r missing' used to score candidate nodes."""
        role = self.role_emb(role_ids.to(hidden.device))
        return F.normalize(self.completion_out(torch.cat([hidden, role], dim=-1)), dim=-1)


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))
