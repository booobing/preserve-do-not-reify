"""Evaluation metrics (contract C8): retrieval recall@k / MRR and n-ary completion accuracy."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

from .contrastive import AlignmentModel, WorldView
from .synthetic import Query, World


def retrieval_metrics(
    query_emb: Tensor, fact_emb: Tensor, gold_idx: Sequence[int], *, ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    n = int(query_emb.shape[0])
    if n == 0:
        out = {f"recall@{k}": 0.0 for k in ks}
        out.update({"mrr": 0.0, "n": 0.0})
        return out
    scores = query_emb @ fact_emb.t()  # [Q, F]
    gold = torch.tensor(list(gold_idx), device=scores.device).unsqueeze(1)
    gold_scores = scores.gather(1, gold)
    # pessimistic rank: ties count against the gold fact (collapsed embeddings cannot score 1.0)
    ranks = 1 + (scores > gold_scores).sum(dim=1).float() + ((scores == gold_scores).sum(dim=1).float() - 1.0)
    out = {f"recall@{k}": float((ranks <= k).float().mean().item()) for k in ks}
    out["mrr"] = float((1.0 / ranks).mean().item())
    out["n"] = float(n)
    return out


def _unique_fact_groups(pairs: Sequence[tuple[str, str]], max_per_group: int | None = None) -> list[list[int]]:
    """Split pair indices into groups where every fact appears at most once (one mask per fact/forward).

    `max_per_group` caps how many facts are masked in the same forward. Masking many at once is
    cheaper but perturbs the graph the candidates are scored against, so large corpora set a cap
    (e.g. 128 of 31k facts perturbs 0.4% of the graph instead of 17%).
    """
    groups: list[list[int]] = []
    used: list[set[str]] = []
    for idx, (fact, _) in enumerate(pairs):
        for g, seen in zip(groups, used, strict=True):
            if fact not in seen and (max_per_group is None or len(g) < max_per_group):
                g.append(idx)
                seen.add(fact)
                break
        else:
            groups.append([idx])
            used.append({fact})
    return groups


@torch.no_grad()
def completion_metrics(
    model: AlignmentModel,
    view: WorldView,
    queries: Sequence[Query],
    *,
    ks: tuple[int, ...] = (1, 3),
    max_masks_per_forward: int | None = None,
    with_mrr: bool = False,
) -> dict[str, float]:
    pairs = [(q.edge_id, q.masked_role or "") for q in queries]
    if not pairs:
        out = {f"acc@{k}": 0.0 for k in ks}
        out["n"] = 0.0
        if with_mrr:
            out["mrr"] = 0.0
        return out
    hits = {k: 0 for k in ks}
    reciprocal = 0.0
    for group in _unique_fact_groups(pairs, max_masks_per_forward):
        sub_pairs = [pairs[i] for i in group]
        q_emb, node_emb, node_index = model.completion_forward(view, sub_pairs)
        for row, (fact, role) in enumerate(sub_pairs):
            candidate_ids = view.candidates_by_type[view.role_type(fact, role)]
            cand = node_emb[[node_index[c] for c in candidate_ids]]
            scores = cand @ q_emb[row]
            target = candidate_ids.index(view.graph.edges[fact].member(role))
            rank = 1 + int((scores > scores[target]).sum().item()) + int((scores == scores[target]).sum().item()) - 1
            reciprocal += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1
    out = {f"acc@{k}": hits[k] / len(pairs) for k in ks}
    out["n"] = float(len(pairs))
    if with_mrr:
        out["mrr"] = reciprocal / len(pairs)
    return out


@torch.no_grad()
def evaluate_split(model: AlignmentModel, view: WorldView, world: World, split: str) -> dict[str, Any]:
    model.eval()
    retrieval_queries = world.queries_for(split, kind="retrieval")
    fact_emb = model.encode_facts(view)
    if retrieval_queries:
        query_emb = model.encode_text([q.text for q in retrieval_queries])
        gold = [view.fact_index[q.edge_id] for q in retrieval_queries]
        retrieval = retrieval_metrics(query_emb, fact_emb, gold)
    else:
        retrieval = retrieval_metrics(torch.zeros((0, fact_emb.shape[-1])), fact_emb, [])
    completion = completion_metrics(model, view, world.queries_for(split, kind="completion"))
    return {"retrieval": retrieval, "completion": completion}
