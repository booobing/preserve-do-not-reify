"""The pre-registered arms (charter §A-2) and their parameter-matched construction.

- hyper:          RoleAwareHypergraphEncoder on the native n-ary incidence (fact = hyperedge)
- reified_binary: the SAME encoder class/config on the reified binary incidence (fact = blank node)
- flat_text:      no structure; fact text -> shared text tower -> MLP adapter sized to the encoder
- reified_binary_deep (optional steelman, not judged): reified view with 2x layers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contrastive import AlignmentModel, Arm, WorldView
from .encoder import RoleAwareHypergraphEncoder, count_parameters
from .hypergraph import Hypergraph
from .text_encoder import HashingTextEncoder

ARM_NAMES: tuple[str, ...] = ("hyper", "reified_binary", "flat_text", "reified_binary_deep")
JUDGED_ARMS: tuple[str, str] = ("hyper", "reified_binary")


@dataclass(frozen=True)
class ModelConfig:
    d: int = 64
    layers: int = 2
    text_dim: int = 64
    text_buckets: int = 4096
    dropout: float = 0.0
    head_parity: bool = False  # v1 default; charter v2 §V2-B turns it on
    text_tower: str = "hashing"  # "hashing" (v1) | "sentence_transformer" (charter v2 §V2-C)


class HyperArm(Arm):
    name = "hyper"

    def __init__(self, encoder: RoleAwareHypergraphEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def encode_facts(self, view: WorldView, text_tower: nn.Module) -> Tensor:
        batch = view.native_for(self.training)
        out = self.encoder(batch, text_tower(view.node_texts_for(self.training)))
        return out["edge_emb"]  # rows align with view.fact_ids_for(self.training)

    def completion_forward(self, view: WorldView, text_tower: nn.Module, pairs: Sequence[tuple[str, str]]):
        source = view.native_for(self.training)
        drop = [(fact, view.vocab.role_id(role)) for fact, role in pairs]
        batch = source.without_memberships(drop)
        out = self.encoder(batch, text_tower(view.node_texts_for(self.training)))
        rows = [source.edge_index[fact] for fact, _ in pairs]
        role_ids = torch.tensor([view.vocab.role_id(role) for _, role in pairs], dtype=torch.long)
        queries = self.encoder.completion_query(out["edge_hidden"][rows], role_ids)
        return queries, out["node_emb"], source.node_index


class ReifiedArm(Arm):
    """The reified view of the same facts, through the same encoder.

    `head_parity` controls which readout produces the fact embedding. v1 used the node head,
    which left `edge_out` unused and gave the two arms structurally different readouts. With
    parity on (the v2 default) the blank node's hidden state goes through `edge_out`, the same
    head the hypergraph arm uses for its hyperedges, so only the aggregation differs.
    """

    name = "reified_binary"

    def __init__(self, encoder: RoleAwareHypergraphEncoder, *, head_parity: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.head_parity = bool(head_parity)

    def _fact_embeddings(self, out: dict, rows: list[int]) -> Tensor:
        if self.head_parity:
            return F.normalize(self.encoder.edge_out(out["node_hidden"][rows]), dim=-1)
        return out["node_emb"][rows]

    def encode_facts(self, view: WorldView, text_tower: nn.Module) -> Tensor:
        source = view.reified_for(self.training)
        out = self.encoder(source, text_tower(view.reified_node_texts_for(self.training)))
        rows = [source.node_index[Hypergraph.blank_node_id(f)] for f in view.fact_ids_for(self.training)]
        return self._fact_embeddings(out, rows)

    def completion_forward(self, view: WorldView, text_tower: nn.Module, pairs: Sequence[tuple[str, str]]):
        source = view.reified_for(self.training)
        subject_id = view.vocab.role_id("subject")
        object_id = view.vocab.role_id("object")
        drop = []
        for fact, role in pairs:
            edge_id = Hypergraph.role_edge_id(fact, role)
            drop.append((edge_id, subject_id))
            drop.append((edge_id, object_id))
        batch = source.without_memberships(drop)
        out = self.encoder(batch, text_tower(view.reified_node_texts_for(self.training)))
        rows = [source.node_index[Hypergraph.blank_node_id(fact)] for fact, _ in pairs]
        role_ids = torch.tensor([view.vocab.role_id(role) for _, role in pairs], dtype=torch.long)
        # `completion_query` already consumes the blank node's hidden state, which is the same
        # quantity the hyper arm feeds from its hyperedge, so this path needs no parity switch.
        queries = self.encoder.completion_query(out["node_hidden"][rows], role_ids)
        return queries, out["node_emb"], source.node_index


class FlatTextArm(Arm):
    name = "flat_text"

    def __init__(self, *, text_dim: int, d: int, hidden: int) -> None:
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, d))

    def _embed(self, text_tower: nn.Module, texts: Sequence[str]) -> Tensor:
        return F.normalize(self.adapter(text_tower(list(texts))), dim=-1)

    def encode_facts(self, view: WorldView, text_tower: nn.Module) -> Tensor:
        texts = [view.graph.fact_to_text(view.graph.edges[f]) for f in view.fact_ids_for(self.training)]
        return self._embed(text_tower, texts)

    def completion_forward(self, view: WorldView, text_tower: nn.Module, pairs: Sequence[tuple[str, str]]):
        masked = [view.graph.fact_to_masked_text(view.graph.edges[fact], role) for fact, role in pairs]
        queries = self._embed(text_tower, masked)
        node_emb = self._embed(text_tower, view.node_texts)
        return queries, node_emb, view.native.node_index

    def encode_facts_for_training(self, view: WorldView, text_tower: nn.Module) -> Tensor:
        texts = [view.graph.fact_to_text(view.graph.edges[f]) for f in view.fact_ids_for(self.training)]
        return self._embed(text_tower, texts)


def _encoder(view: WorldView, cfg: ModelConfig, *, layers: int | None = None) -> RoleAwareHypergraphEncoder:
    return RoleAwareHypergraphEncoder(
        num_node_types=len(view.vocab.node_types),
        num_edge_types=len(view.vocab.edge_types),
        num_roles=len(view.vocab.roles),
        d=cfg.d,
        layers=cfg.layers if layers is None else layers,
        text_dim=cfg.text_dim,
        dropout=cfg.dropout,
    )


def flat_adapter_hidden(encoder_params: int, *, text_dim: int, d: int) -> int:
    """Hidden width so that Linear(text_dim,h)+Linear(h,d) has ~encoder_params parameters."""
    hidden = (encoder_params - d) / (text_dim + 1 + d)
    return max(1, int(round(hidden)))


def build_arm(name: str, view: WorldView, cfg: ModelConfig) -> Arm:
    if name == "hyper":
        return HyperArm(_encoder(view, cfg))
    if name == "reified_binary":
        return ReifiedArm(_encoder(view, cfg), head_parity=cfg.head_parity)
    if name == "reified_binary_deep":
        arm = ReifiedArm(_encoder(view, cfg, layers=cfg.layers * 2), head_parity=cfg.head_parity)
        arm.name = "reified_binary_deep"
        return arm
    if name == "flat_text":
        reference = count_parameters(_encoder(view, cfg))
        hidden = flat_adapter_hidden(reference, text_dim=cfg.text_dim, d=cfg.d)
        return FlatTextArm(text_dim=cfg.text_dim, d=cfg.d, hidden=hidden)
    raise KeyError(f"unknown arm {name!r}; expected one of {ARM_NAMES}")


def build_text_tower(cfg: ModelConfig, *, seed: int) -> nn.Module:
    if cfg.text_tower == "hashing":
        return HashingTextEncoder(num_buckets=cfg.text_buckets, text_dim=cfg.text_dim, seed=seed)
    if cfg.text_tower == "sentence_transformer":
        from .text_encoder import CachedSentenceTransformerTower

        return CachedSentenceTransformerTower(text_dim=cfg.text_dim)
    raise KeyError(f"unknown text tower {cfg.text_tower!r}")


def build_model(name: str, view: WorldView, cfg: ModelConfig, *, seed: int) -> AlignmentModel:
    torch.manual_seed(seed)
    text_tower = build_text_tower(cfg, seed=seed)
    arm = build_arm(name, view, cfg)
    return AlignmentModel(arm, text_tower)


def parameter_report(view: WorldView, cfg: ModelConfig, arms: Sequence[str] = JUDGED_ARMS + ("flat_text",)) -> dict[str, int]:
    report: dict[str, int] = {}
    for name in arms:
        model = build_model(name, view, cfg, seed=0)
        report[name] = count_parameters(model)
    return report
