"""Chapter-5 role-aware hypergraph encoder inside the StarE pipeline
(docs/charter-stare-native-encoder-v1.md).

The StarE bridge already runs upstream StarE's data loader, sampler, Transformer decoder and
filtered evaluation. This module swaps only the ENCODER: instead of StarE's CompGCN-style
convolution, the node table is produced by the same `RoleAwareMessageLayer` the chapter-5
encoder uses (node -> edge -> node, mean aggregation, role embedding added to every message).
The decoder input tokens `[s, r, q1, v1, ...]` and everything downstream are unchanged.

Three graph views of the same statements, one per arm:

    hyper    one hyperedge per statement, typed by its predicate; members are
             (s, role:subject), (o, role:object) and (v_k, role:q_k) for every qualifier pair.
             Every participant, qualifier values included, sends AND receives messages.
    reified  chapter 5's `reify()`: a statement node b_i per statement, and per member a
             2-member edge typed "role edge of rho" with members (b_i, subject), (member, object).
    prune    one 2-member hyperedge per statement (s, o); qualifiers are invisible to the
             encoder (they still reach the decoder as tokens, like in every arm).

Id spaces follow upstream: entities 0..E-1 (0 = pad), relations 0..R-1 (0 = pad; predicates and
qualifier properties share it). Roles: qualifier property id for a qualifier slot, R for
subject, R + 1 for object. Edge types: predicate ids 0..R-1, and R + role for a role edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..encoder import RoleAwareMessageLayer
from .reify_stare import ReificationError, qualifier_pairs

MODES: tuple[str, ...] = ("hyper", "reified", "prune")
NATIVE_ARMS: dict[str, str] = {"native_hyper": "hyper", "native_reified": "reified", "native_prune": "prune"}


@dataclass(frozen=True)
class NativeIncidence:
    mode: str
    num_entities: int  # E, includes the pad id 0
    num_relations: int  # R, includes the pad id 0
    node_idx: np.ndarray  # (M,) node of each membership
    edge_idx: np.ndarray  # (M,) hyperedge of each membership
    role_idx: np.ndarray  # (M,) role id of each membership
    edge_type: np.ndarray  # (num_edges,) edge type id
    blank_predicates: np.ndarray  # (num_blank,) predicate of each statement node (reified only)

    @property
    def role_subject(self) -> int:
        return self.num_relations

    @property
    def role_object(self) -> int:
        return self.num_relations + 1

    @property
    def num_roles(self) -> int:
        return self.num_relations + 2

    @property
    def num_edge_types(self) -> int:
        return self.num_relations + self.num_roles

    @property
    def num_blank(self) -> int:
        return int(self.blank_predicates.shape[0])

    @property
    def num_nodes(self) -> int:
        return self.num_entities + self.num_blank

    @property
    def num_edges(self) -> int:
        return int(self.edge_type.shape[0])

    @property
    def num_memberships(self) -> int:
        return int(self.node_idx.shape[0])

    def summary(self) -> dict[str, int | str]:
        return {
            "mode": self.mode,
            "nodes": self.num_nodes,
            "statement_nodes": self.num_blank,
            "hyperedges": self.num_edges,
            "memberships": self.num_memberships,
            "edge_types": self.num_edge_types,
            "roles": self.num_roles,
        }


def _members(row: Sequence[int], *, num_entities: int, num_relations: int, mode: str) -> tuple[int, list[tuple[int, int]]]:
    if len(row) < 3:
        raise ReificationError(f"statement needs a main triple: {list(row)}")
    subject, predicate, obj = int(row[0]), int(row[1]), int(row[2])
    if 0 in (subject, predicate, obj):
        raise ReificationError(f"pad id inside the main triple: {list(row)}")
    if not (0 < subject < num_entities and 0 < obj < num_entities and 0 < predicate < num_relations):
        raise ReificationError(f"id out of range for E={num_entities}, R={num_relations}: {list(row)}")
    members = [(subject, num_relations), (obj, num_relations + 1)]
    if mode != "prune":
        for prop, value in qualifier_pairs(row):
            if not (0 < prop < num_relations and 0 < value < num_entities):
                raise ReificationError(f"qualifier id out of range: ({prop}, {value}) in {list(row)}")
            members.append((value, prop))
    return predicate, members


def build_native_incidence(
    statements: Sequence[Sequence[int]], *, num_entities: int, num_relations: int, mode: str
) -> NativeIncidence:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if num_entities <= 0 or num_relations <= 0:
        raise ReificationError("num_entities and num_relations must be positive upstream counts")
    nodes: list[int] = []
    edges: list[int] = []
    roles: list[int] = []
    edge_types: list[int] = []
    blanks: list[int] = []
    role_subject, role_object = num_relations, num_relations + 1
    for index, row in enumerate(statements):
        predicate, members = _members(row, num_entities=num_entities, num_relations=num_relations, mode=mode)
        if mode == "reified":
            blank = num_entities + index
            blanks.append(predicate)
            for node, role in members:
                edge = len(edge_types)
                edge_types.append(num_relations + role)  # "role edge of rho"
                nodes.extend((blank, node))
                edges.extend((edge, edge))
                roles.extend((role_subject, role_object))  # reify(): (blank, subject), (member, object)
        else:
            edge = len(edge_types)
            edge_types.append(predicate)
            for node, role in members:
                nodes.append(node)
                edges.append(edge)
                roles.append(role)
    as_i64 = lambda xs: np.asarray(xs, dtype=np.int64)  # noqa: E731
    return NativeIncidence(
        mode=mode, num_entities=int(num_entities), num_relations=int(num_relations),
        node_idx=as_i64(nodes), edge_idx=as_i64(edges), role_idx=as_i64(roles),
        edge_type=as_i64(edge_types), blank_predicates=as_i64(blanks),
    )


class NativeEncoderMixin:
    """Replaces upstream `StarEEncoder.forward_base`; mixed in front of `StarE_Transformer`."""

    def init_native(self, inc: NativeIncidence, *, layers: int, get_param) -> None:
        d = self.emb_dim  # type: ignore[attr-defined]
        device = self.device  # type: ignore[attr-defined]
        self.native_mode = inc.mode
        self.num_real_entities = inc.num_entities
        for name, array in (("native_node_idx", inc.node_idx), ("native_edge_idx", inc.edge_idx),
                            ("native_role_idx", inc.role_idx), ("native_edge_type", inc.edge_type),
                            ("native_blank_predicates", inc.blank_predicates)):
            self.register_buffer(name, torch.as_tensor(array, dtype=torch.long, device=device))  # type: ignore[attr-defined]
        self.native_edge_emb = nn.Embedding(inc.num_edge_types, d)
        self.native_role_emb = nn.Embedding(inc.num_roles, d)
        self.native_input_norm = nn.LayerNorm(d)
        self.native_layers = nn.ModuleList(RoleAwareMessageLayer(d) for _ in range(layers))
        if inc.num_blank:
            self.w_blank = get_param((1, d))
        # Upstream's convolutions are not used by this encoder; drop them so they are not
        # counted as parameters and receive no optimiser state.
        self.conv1 = None
        self.conv2 = None
        self.to(device)  # type: ignore[attr-defined]

    def native_node_init(self) -> Tensor:
        x = self.init_embed  # type: ignore[attr-defined]
        if self.native_blank_predicates.numel():
            blanks = self.native_edge_emb(self.native_blank_predicates) + self.w_blank
            x = torch.cat([x, blanks], dim=0)
        return x

    def forward_base(self, sub, rel, drop1, drop2, quals=None, embed_qualifiers: bool = False, return_mask: bool = False):
        h_v = self.native_input_norm(self.native_node_init())
        h_e = self.native_edge_emb(self.native_edge_type)
        role = self.native_role_emb(self.native_role_idx)
        drops = (drop1, drop2)
        for depth, layer in enumerate(self.native_layers):
            h_v, h_e = layer(h_v, h_e, role, self.native_node_idx, self.native_edge_idx)
            if depth < len(drops):  # same placement as upstream: drop after conv1 and after conv2
                h_v = drops[depth](h_v)
        x = h_v
        r = self.init_rel  # type: ignore[attr-defined]  # decoder relation table, identical in every arm
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        candidates = x[: self.num_real_entities]
        if not embed_qualifiers:
            return sub_emb, rel_emb, candidates
        if quals is None:
            raise ValueError("embed_qualifiers=True needs the qualifier tensor")
        qual_obj_emb = torch.index_select(x, 0, quals[:, 1::2].reshape(-1)).view(sub_emb.shape[0], -1, sub_emb.shape[1])
        qual_rel_emb = torch.index_select(r, 0, quals[:, 0::2].reshape(-1)).view(rel_emb.shape[0], -1, rel_emb.shape[1])
        if not return_mask:
            return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, candidates
        mask = torch.zeros((sub.shape[0], quals.shape[1] + 2), dtype=torch.bool, device=self.device)  # type: ignore[attr-defined]
        mask[:, 2:] = quals == 0
        return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, candidates, mask


_NATIVE_CLASSES: dict[type, type] = {}


def native_class(base: type) -> type:
    if base not in _NATIVE_CLASSES:
        _NATIVE_CLASSES[base] = type("NativeEncoderStarE_Transformer", (NativeEncoderMixin, base), {"model_name": "NativeEncoderStarE_Transformer"})
    return _NATIVE_CLASSES[base]


def build_native_model(up: Any, graph_repr: dict[str, np.ndarray], config: dict[str, Any], inc: NativeIncidence) -> nn.Module:
    """Upstream `StarE_Transformer` (decoder, entity and relation tables) with this encoder.

    `graph_repr` only satisfies the upstream constructor; the native forward never reads it.
    Built as an upstream instance and re-tagged, because upstream's `super(self.__class__, ...)`
    recurses for real subclasses (same reason as `model.build_reified_model`).
    """
    if config["NUM_ENTITIES"] != inc.num_entities or config["NUM_RELATIONS"] != inc.num_relations:
        raise ValueError("config disagrees with the incidence id spaces")
    model = up.StarE_Transformer(graph_repr, config)
    model.__class__ = native_class(up.StarE_Transformer)
    model.init_native(inc, layers=int(config["STAREARGS"]["LAYERS"]), get_param=up.get_param)
    return model


def is_native(model: nn.Module) -> bool:
    return isinstance(model, NativeEncoderMixin)
