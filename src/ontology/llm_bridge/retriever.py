"""The retriever every Paper 2 arm shares (charter-hnn-llm-bridge-v1, "고정 요인").

All five arms must see the same k facts for the same question, otherwise a difference in answers
could come from retrieval rather than from how the context reaches the decoder. So retrieval runs
once here, on a Paper 1 alignment model, and the arms differ only in what they do with the result:
three serialize it to text, one turns it into cross-attention memory, one ignores it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..baselines import ModelConfig, build_model
from ..contrastive import AlignmentModel, TrainConfig, WorldView, train_alignment
from ..synthetic import World


@dataclass
class RetrievalResult:
    fact_ids: list[str]
    scores: list[float]


class FactRetriever:
    """A trained Paper 1 `hyper` tower, frozen and reused as the shared retriever."""

    def __init__(self, model: AlignmentModel, view: WorldView, *, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.view = view
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.model.arm.eval()
        with torch.no_grad():
            self.fact_emb = self.model.encode_facts(view).detach()
        self.fact_ids = list(view.fact_ids)
        self._index = {fact_id: i for i, fact_id in enumerate(self.fact_ids)}
        self._encoded: dict[str, Tensor] | None = None
        self._query_cache: dict[str, Tensor] = {}

    @classmethod
    def train(
        cls,
        world: World,
        view: WorldView,
        *,
        seed: int,
        steps: int = 1500,
        model_cfg: ModelConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> FactRetriever:
        cfg = model_cfg or ModelConfig()
        model = build_model("hyper", view, cfg, seed=seed)
        train_alignment(model, view, world, TrainConfig(steps=steps, seed=seed), device=device)
        return cls(model, view, device=device)

    @torch.no_grad()
    def top_k(self, question: str, k: int) -> RetrievalResult:
        query = self._query_cache.get(question)
        if query is None:
            query = self.model.encode_text([question]).to(self.fact_emb.device).detach()
            self._query_cache[question] = query
        scores = (query @ self.fact_emb.t()).squeeze(0)
        k = min(int(k), scores.shape[0])
        top = scores.topk(k)
        return RetrievalResult(
            fact_ids=[self.fact_ids[int(i)] for i in top.indices.tolist()],
            scores=[float(s) for s in top.values.tolist()],
        )

    @torch.no_grad()
    def _encode_once(self) -> dict[str, Tensor]:
        """Encoder output for the whole graph, computed once.

        The tower is frozen after training, so this does not change between calls; recomputing it
        per question would dominate the run without altering a single number.
        """
        if self._encoded is None:
            self._encoded = {
                key: value.detach()
                for key, value in self.model.arm.encoder(
                    self.view.native, self.model.text_tower(self.view.node_texts)
                ).items()
            }
        return self._encoded

    @torch.no_grad()
    def memory_inputs(self, fact_ids: Sequence[str]) -> tuple[Tensor, Tensor]:
        """(node_emb, edge_emb) for the retrieved facts and the entities participating in them.

        The hnn_xattn arm reads exactly the facts the text arms serialize, so the two sides carry
        the same evidence and differ only in the channel it travels through.
        """
        out = self._encode_once()
        edge_rows = [self.view.native.edge_index[f] for f in fact_ids]
        node_ids: list[str] = []
        for fact_id in fact_ids:
            for node_id, _role in self.view.graph.edges[fact_id].members:
                if node_id not in node_ids:
                    node_ids.append(node_id)
        node_rows = [self.view.native.node_index[n] for n in node_ids]
        return out["node_emb"][node_rows], out["edge_emb"][edge_rows]

    @torch.no_grad()
    def memory_inputs_grouped(self, fact_ids: Sequence[str]) -> tuple[Tensor, Tensor, list[int]]:
        """(node_emb, edge_emb, block_sizes) with participants ordered and repeated per fact.

        Same facts and same entities as `memory_inputs`; only the layout differs. A node in two
        retrieved facts appears in both blocks, which is what carries fact membership across the
        channel (charter v6 §3-a).
        """
        out = self._encode_once()
        edge_rows = [self.view.native.edge_index[f] for f in fact_ids]
        node_rows: list[int] = []
        block_sizes: list[int] = []
        for fact_id in fact_ids:
            members = self.view.graph.edges[fact_id].members
            node_rows.extend(self.view.native.node_index[n] for n, _role in members)
            block_sizes.append(len(members))
        return out["node_emb"][node_rows], out["edge_emb"][edge_rows], block_sizes

    @torch.no_grad()
    def memory_inputs_spokes(self, fact_ids: Sequence[str]) -> tuple[Tensor, Tensor, Tensor]:
        """(node_emb [M, d], edge_emb [M, d], role_ids [M]) with row i = membership i (charter v7 §3-a).

        Same facts, same entities and the same frozen embeddings as `memory_inputs`; every row now
        carries its own fact and role, so the connection itself is what crosses the channel.
        """
        out = self._encode_once()
        node_rows, edge_rows, roles = [], [], []
        for fact_id in fact_ids:
            edge_row = self.view.native.edge_index[fact_id]
            for node_id, role in self.view.graph.edges[fact_id].members:
                node_rows.append(self.view.native.node_index[node_id])
                edge_rows.append(edge_row)
                roles.append(self.view.vocab.role_id(role))
        role_ids = torch.tensor(roles, dtype=torch.long)
        return out["node_emb"][node_rows], out["edge_emb"][edge_rows], role_ids

    def memory_block_order(self, fact_ids: Sequence[str]) -> list[str]:
        """Node ids behind the grouped node slots, in slot order, repeats included."""
        order: list[str] = []
        for fact_id in fact_ids:
            order.extend(n for n, _role in self.view.graph.edges[fact_id].members)
        return order

    def memory_node_order(self, fact_ids: Sequence[str]) -> list[str]:
        """The node ids behind the node slots of `memory_inputs`, in slot order.

        The alignment loss has to name which memory slots belong to the gold fact, and memory is
        laid out as [one slot per fact ; one slot per distinct participant]. This exposes the
        second half's order rather than making callers re-derive it and drift out of step.
        """
        node_ids: list[str] = []
        for fact_id in fact_ids:
            for node_id, _role in self.view.graph.edges[fact_id].members:
                if node_id not in node_ids:
                    node_ids.append(node_id)
        return node_ids

    @torch.no_grad()
    def soft_prompt_inputs(self, fact_ids: Sequence[str]) -> tuple[Tensor, list[Tensor]]:
        """(edge_emb [F, d], one [n_i, d] block of member embeddings per fact) for the soft arm."""
        out = self._encode_once()
        edge_rows = [self.view.native.edge_index[f] for f in fact_ids]
        blocks = []
        for fact_id in fact_ids:
            rows = [self.view.native.node_index[n] for n, _role in self.view.graph.edges[fact_id].members]
            blocks.append(out["node_emb"][rows])
        return out["edge_emb"][edge_rows], blocks

    def retrieval_hit_rate(self, questions: Sequence[str], gold_fact_ids: Sequence[str], k: int) -> float:
        """How often the gold fact is in the shared top-k. Reported so an arm's failure can be
        separated from a retrieval failure that hit every arm equally."""
        if not questions:
            return 0.0
        hits = sum(gold in self.top_k(q, k).fact_ids for q, gold in zip(questions, gold_fact_ids, strict=True))
        return hits / len(questions)
