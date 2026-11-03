"""Pure conversion between StarE harness rows and HAHE inputs, plus the shared evaluation index.

A StarE row is ``[s, r, o, q1, v1, ..., 0, 0]`` in StarE's id spaces (id 0 = pad in both the entity
and the relation table). HAHE wants a vocabulary ``[PAD], [MASK], relations..., entities...`` and
examples whose qualifiers are grouped by attribute (attributes and values sorted, as HAHE's reader
does). The evaluation side mirrors ``loops/evaluation.py`` of StarE exactly:

- queries: object of every test row (HAHE mask position 2) and subject of every test row (position 0;
  StarE asks the same question through the reciprocal row);
- filter: every other entity known to complete the same (subject or object, relation, raw qualifier
  tuple) across train + valid + test;
- candidates: entities that appear as a subject or object somewhere in train + valid + test;
- rank: ``1 + argsort(argsort(-score))[target]`` after setting filtered and excluded entries to -1e6.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

PAD, MASK = "[PAD]", "[MASK]"
OBJECT_POSITION, SUBJECT_POSITION = 2, 0
EXCLUDED_SCORE = -1_000_000.0


@dataclass(frozen=True)
class HaheVocab:
    tokens: tuple[str, ...]  # index = HAHE token id
    num_relations: int  # R without pad
    num_entities: int  # E without pad

    @property
    def entity_offset(self) -> int:
        return 2 + self.num_relations

    def lines(self) -> list[str]:
        return list(self.tokens)


def build_vocab(e2id: Mapping[str, int], r2id: Mapping[str, int]) -> HaheVocab:
    """Relations then entities, each in StarE id order (pad '__na__' dropped), so that
    HAHE token = 2 + (rid - 1) for a relation and entity_offset + (eid - 1) for an entity."""
    rels = [t for t, _ in sorted(r2id.items(), key=lambda kv: kv[1]) if r2id[t] != 0]
    ents = [t for t, _ in sorted(e2id.items(), key=lambda kv: kv[1]) if e2id[t] != 0]
    if [r2id[t] for t in rels] != list(range(1, len(rels) + 1)):
        raise ValueError("relation ids must be 1..R without gaps")
    if [e2id[t] for t in ents] != list(range(1, len(ents) + 1)):
        raise ValueError("entity ids must be 1..E without gaps")
    return HaheVocab(tokens=(PAD, MASK, *rels, *ents), num_relations=len(rels), num_entities=len(ents))


def qualifier_pairs(row: Sequence[int]) -> list[tuple[int, int]]:
    tail = list(row[3:])
    return [(int(tail[k]), int(tail[k + 1])) for k in range(0, len(tail) - 1, 2) if tail[k] != 0 or tail[k + 1] != 0]


@dataclass(frozen=True)
class Statement:
    arity: int
    head: str
    relation: str
    tail: str
    auxiliary: tuple[tuple[str, tuple[str, ...]], ...]  # sorted attributes with sorted values


def row_to_statement(row: Sequence[int], id2e: Sequence[str], id2r: Sequence[str]) -> Statement:
    s, r, o = int(row[0]), int(row[1]), int(row[2])
    if 0 in (s, r, o):
        raise ValueError(f"pad inside the main triple: {list(row)}")
    groups: dict[str, list[str]] = collections.defaultdict(list)
    pairs = qualifier_pairs(row)
    for q, v in pairs:
        if q == 0 or v == 0:
            raise ValueError(f"half a qualifier pair in {list(row)}")
        groups[id2r[q]].append(id2e[v])
    aux = tuple((attr, tuple(sorted(groups[attr]))) for attr in sorted(groups))
    return Statement(arity=2 + len(pairs), head=id2e[s], relation=id2r[r], tail=id2e[o], auxiliary=aux)


def max_arity(statements: Iterable[Statement]) -> int:
    return max(st.arity for st in statements)


def candidate_entities(rows: Iterable[Sequence[int]], num_entities_with_pad: int) -> np.ndarray:
    """Boolean mask over StarE entity ids: True for entities seen as subject or object anywhere
    (StarE's ENT_POS_FILTERED with CORRUPTION_POSITIONS [0, 2]); the pad id is never a candidate."""
    seen = np.zeros(num_entities_with_pad, dtype=bool)
    for row in rows:
        seen[int(row[0])] = True
        seen[int(row[2])] = True
    seen[0] = False
    return seen


def filter_index(rows: Iterable[Sequence[int]]) -> dict[tuple, set[int]]:
    """StarE's evaluation index with reciprocals: ('o', s, r, quals) -> objects and
    ('s', o, r, quals) -> subjects, keyed on the raw (padded) qualifier tuple."""
    index: dict[tuple, set[int]] = collections.defaultdict(set)
    for row in rows:
        s, r, o = int(row[0]), int(row[1]), int(row[2])
        quals = tuple(int(x) for x in row[3:])
        index[("o", s, r, quals)].add(o)
        index[("s", o, r, quals)].add(s)
    return index


def stare_rank(scores: np.ndarray, target: int, known: Iterable[int], candidates: np.ndarray) -> int:
    """StarE's rank for one query. `scores` is indexed by StarE entity id (length E with pad).
    The target must be a candidate: StarE would push a non-candidate target to -1e6 and rank it
    among ties, which cannot happen when candidates come from the same rows as the queries."""
    if not candidates[target]:
        raise ValueError(f"target {target} is not a candidate entity")
    pred = np.array(scores, dtype=np.float64, copy=True)
    irrelevant = ~candidates.copy()
    for k in known:
        if k != target:
            irrelevant[k] = True
    pred[irrelevant] = EXCLUDED_SCORE
    order = np.argsort(-pred, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return int(ranks[target])


def hahe_native_rank(scores: np.ndarray, target: int, known: Iterable[int]) -> int:
    """HAHE's own rule for comparison: all entities are candidates, filtered, ties pessimistic."""
    pred = np.array(scores, dtype=np.float64, copy=True)
    for k in known:
        if k != target:
            pred[k] = -np.inf
    pred[0] = -np.inf
    return int((pred >= pred[target]).sum())


def summarise(ranks: Sequence[int]) -> dict[str, float]:
    r = np.asarray(ranks, dtype=np.float64)
    out = {"count": float(len(r)), "mr": float(r.mean()), "mrr": float((1.0 / r).mean())}
    for k in (1, 3, 5, 10):
        out[f"hits_at {k}"] = float((r <= k).mean())
    return out


def mean_of_sides(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    """StarE's `_mean_metrics_`: average object- and subject-prediction metrics."""
    return {k: (left[k] + right[k]) / 2.0 if k != "count" else left[k] for k in left}
