"""CLIP-style alignment between a structure arm and a text tower (contract C7).

    loss = 0.5 * (CE(sim/tau, arange) + CE(sim^T/tau, arange)) + completion CE
    tau = exp(-logit_scale), logit_scale initialised to log(1/0.07)

The `Arm` interface is what the three pre-registered arms implement (see baselines.py):
`hyper`, `reified_binary` (same encoder, reified incidence) and `flat_text` (no structure).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .hypergraph import Hypergraph, IncidenceBatch
from .schema import OntologySchema, Vocab
from .synthetic import World
from .text_encoder import HashingTextEncoder


@dataclass
class WorldView:
    """Precomputed, device-agnostic tensors/views of one World shared by all arms."""

    schema: OntologySchema
    vocab: Vocab
    graph: Hypergraph
    fact_ids: list[str]
    native: IncidenceBatch
    reified_graph: Hypergraph
    reified: IncidenceBatch
    candidates_by_type: dict[str, list[str]]
    node_texts: list[str]  # aligned with native.node_ids
    reified_node_texts: list[str]  # aligned with reified.node_ids
    fact_index: dict[str, int] = field(default_factory=dict)
    # Inductive protocol (charter v2 §V2-A): when set, training reads these views instead, so
    # held-out facts contribute no hyperedge while the node set stays the same.
    train_native: IncidenceBatch | None = None
    train_reified: IncidenceBatch | None = None
    train_graph: Hypergraph | None = None
    train_reified_graph: Hypergraph | None = None
    train_fact_ids: list[str] = field(default_factory=list)
    train_fact_index: dict[str, int] = field(default_factory=dict)
    train_node_texts: list[str] = field(default_factory=list)
    train_reified_node_texts: list[str] = field(default_factory=list)

    @property
    def inductive(self) -> bool:
        return self.train_native is not None

    def _use_train(self, training: bool) -> bool:
        return bool(training and self.train_native is not None)

    def native_for(self, training: bool) -> IncidenceBatch:
        return self.train_native if self._use_train(training) else self.native

    def reified_for(self, training: bool) -> IncidenceBatch:
        return self.train_reified if self._use_train(training) else self.reified

    def node_texts_for(self, training: bool) -> list[str]:
        return self.train_node_texts if self._use_train(training) else self.node_texts

    def reified_node_texts_for(self, training: bool) -> list[str]:
        return self.train_reified_node_texts if self._use_train(training) else self.reified_node_texts

    def fact_ids_for(self, training: bool) -> list[str]:
        return self.train_fact_ids if self._use_train(training) else self.fact_ids

    def fact_index_for(self, training: bool) -> dict[str, int]:
        return self.train_fact_index if self._use_train(training) else self.fact_index

    def graph_for(self, training: bool) -> Hypergraph:
        return self.train_graph if self._use_train(training) else self.graph

    @classmethod
    def from_world(cls, world: World) -> WorldView:
        graph = world.hypergraph
        vocab = world.schema.vocab()
        reified_graph = graph.reify()
        native = graph.to_incidence(vocab)
        reified = reified_graph.to_incidence(vocab)
        fact_ids = graph.edge_ids()
        return cls(
            schema=world.schema,
            vocab=vocab,
            graph=graph,
            fact_ids=fact_ids,
            native=native,
            reified_graph=reified_graph,
            reified=reified,
            candidates_by_type={t: [n.id for n in graph.nodes_of_type(t)] for t in world.schema.entity_types},
            node_texts=[graph.nodes[n].text or graph.nodes[n].name for n in native.node_ids],
            reified_node_texts=[reified_graph.nodes[n].text or reified_graph.nodes[n].name for n in reified.node_ids],
            fact_index={fact_id: idx for idx, fact_id in enumerate(fact_ids)},
        )

    def to(self, device: torch.device | str) -> WorldView:
        self.native = self.native.to(device)
        self.reified = self.reified.to(device)
        if self.train_native is not None:
            self.train_native = self.train_native.to(device)
        if self.train_reified is not None:
            self.train_reified = self.train_reified.to(device)
        return self

    def role_type(self, fact_id: str, role: str) -> str:
        return self.schema.relation(self.graph.edges[fact_id].relation).role_type(role)


class Arm(nn.Module):
    """Structure side of the alignment model. Subclasses own their parameters."""

    name: str = "arm"

    def encode_facts(self, view: WorldView, text_tower: nn.Module) -> Tensor:
        """[F, d] normalized embeddings aligned with view.fact_ids."""
        raise NotImplementedError

    def completion_forward(
        self, view: WorldView, text_tower: nn.Module, pairs: Sequence[tuple[str, str]]
    ) -> tuple[Tensor, Tensor, dict[str, int]]:
        """Returns (queries [M,d], node_emb [V,d], node_index) for the masked (fact_id, role) pairs."""
        raise NotImplementedError


def info_nce(a: Tensor, b: Tensor, logit_scale: Tensor) -> Tensor:
    logits = logit_scale.exp() * a @ b.t()
    targets = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


class AlignmentModel(nn.Module):
    def __init__(self, arm: Arm, text_tower: HashingTextEncoder, *, init_temperature: float = 0.07) -> None:
        super().__init__()
        self.arm = arm
        self.text_tower = text_tower
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def encode_text(self, texts: Sequence[str]) -> Tensor:
        return F.normalize(self.text_tower(list(texts)), dim=-1)

    def encode_facts(self, view: WorldView) -> Tensor:
        return self.arm.encode_facts(view, self.text_tower)

    def completion_forward(self, view: WorldView, pairs: Sequence[tuple[str, str]]):
        return self.arm.completion_forward(view, self.text_tower, pairs)

    def completion_loss(self, view: WorldView, pairs: Sequence[tuple[str, str]]) -> Tensor:
        queries, node_emb, node_index = self.completion_forward(view, pairs)
        losses = []
        scale = self.logit_scale.exp()
        for row, (fact_id, role) in enumerate(pairs):
            candidate_ids = view.candidates_by_type[view.role_type(fact_id, role)]
            cand = node_emb[[node_index[c] for c in candidate_ids]]
            target = candidate_ids.index(view.graph.edges[fact_id].member(role))
            logits = scale * (cand @ queries[row]).unsqueeze(0)
            losses.append(F.cross_entropy(logits, torch.tensor([target], device=logits.device)))
        return torch.stack(losses).mean()


@dataclass
class TrainConfig:
    steps: int = 200
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-4
    completion_weight: float = 1.0
    seed: int = 1
    log_every: int = 10


def _training_texts(world: World) -> dict[str, list[str]]:
    train_ids = set(world.splits["train"])
    texts: dict[str, list[str]] = {fact_id: [] for fact_id in train_ids}
    for sentence in world.sentences:
        if sentence.edge_id in train_ids:
            texts[sentence.edge_id].append(sentence.text)
    for query in world.queries:
        if query.edge_id in train_ids and query.kind == "retrieval":
            texts[query.edge_id].append(query.text)
    return texts


def train_alignment(
    model: AlignmentModel,
    view: WorldView,
    world: World,
    config: TrainConfig,
    *,
    device: torch.device | str = "cpu",
) -> list[dict[str, Any]]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model.to(device)
    view.to(device)
    texts_by_fact = _training_texts(world)
    train_facts = [f for f in world.splits["train"] if texts_by_fact.get(f)]
    if not train_facts:
        raise ValueError("no training facts with texts")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        batch_size = min(config.batch_size, len(train_facts))
        picks = rng.choice(len(train_facts), size=batch_size, replace=False)
        batch_facts = [train_facts[int(i)] for i in picks]
        batch_texts = [texts_by_fact[f][int(rng.integers(0, len(texts_by_fact[f])))] for f in batch_facts]
        fact_emb = model.encode_facts(view)
        fact_index = view.fact_index_for(model.arm.training)
        rows = torch.tensor([fact_index[f] for f in batch_facts], device=fact_emb.device)
        text_emb = model.encode_text(batch_texts)
        loss_align = info_nce(fact_emb[rows], text_emb, model.logit_scale)
        pairs = []
        for fact_id in batch_facts:
            roles = view.graph.edges[fact_id].roles()
            pairs.append((fact_id, roles[int(rng.integers(0, len(roles)))]))
        loss_completion = model.completion_loss(view, pairs)
        loss = loss_align + config.completion_weight * loss_completion
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(math.log(1.0), math.log(100.0))
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "loss_align": float(loss_align.item()),
                    "loss_completion": float(loss_completion.item()),
                    "temperature": float(1.0 / model.logit_scale.exp().item()),
                    "elapsed_s": time.perf_counter() - started,
                }
            )
    model.eval()
    return history
