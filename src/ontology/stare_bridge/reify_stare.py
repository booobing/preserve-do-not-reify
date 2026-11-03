"""Graph-side reification of StarE integer statements (charter §4-2, decision 0-1 (가)).

Input rows are what the upstream loader produces: ``[s, r, o, qr1, qv1, ..., 0, 0]`` padded to
``MAX_QPAIRS`` with id 0. For each statement ``i`` of the graph split this builds

    blank b_i = E + i
    (b_i, role:subject, s)      role:subject = R
    (b_i, role:object,  o)      role:object  = R + 1
    (b_i, qr_k, qv_k)           one per non-padding qualifier pair, reusing qr_k's relation id

which is chapter 5's ``Hypergraph.reify()`` in StarE's id spaces: the blank node carries the
statement's predicate only (through its initial embedding, see ``model.py``), members hang off it
by role, repeated qualifier properties are kept exactly as the hyper arm keeps them.

Everything here is numpy and pure so the counting invariants are unit-testable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

ROLE_SUBJECT_OFFSET = 0
ROLE_OBJECT_OFFSET = 1
NEW_RELATIONS = 2


class ReificationError(ValueError):
    """A statement row cannot be reified (padding inside the main triple, half a qualifier pair)."""


@dataclass(frozen=True)
class ReifiedGraph:
    triples: np.ndarray  # (T, 3) int64 rows [blank, role_relation, member]
    blank_predicates: np.ndarray  # (n_statements,) predicate id of each blank, statement order
    num_real_entities: int  # E, includes the upstream pad id 0
    num_relations_base: int  # R, upstream count (includes pad id 0; predicates and qualifier properties share it)

    @property
    def num_blank(self) -> int:
        return int(self.blank_predicates.shape[0])

    @property
    def role_subject(self) -> int:
        return self.num_relations_base + ROLE_SUBJECT_OFFSET

    @property
    def role_object(self) -> int:
        return self.num_relations_base + ROLE_OBJECT_OFFSET

    @property
    def num_relations(self) -> int:
        """Relation count the reified arm hands to StarE (R + 2); inverses are added by upstream."""
        return self.num_relations_base + NEW_RELATIONS

    @property
    def num_nodes(self) -> int:
        return self.num_real_entities + self.num_blank

    def blank_id(self, statement_index: int) -> int:
        return self.num_real_entities + int(statement_index)


def qualifier_pairs(row: Sequence[int]) -> list[tuple[int, int]]:
    """Non-padding (property, value) pairs of one padded statement row."""
    pairs: list[tuple[int, int]] = []
    tail = list(row[3:])
    if len(tail) % 2:
        raise ReificationError(f"qualifier tail must have even length, got {len(tail)}: {list(row)}")
    for k in range(0, len(tail), 2):
        prop, value = int(tail[k]), int(tail[k + 1])
        if prop == 0 and value == 0:
            continue
        if prop == 0 or value == 0:
            raise ReificationError(f"half a qualifier pair ({prop}, {value}) in {list(row)}")
        pairs.append((prop, value))
    return pairs


def expected_role_edges(statements: Sequence[Sequence[int]]) -> int:
    """Σ(2 + kept qualifier pairs): the number of role edges before inverses (charter §2)."""
    return sum(2 + len(qualifier_pairs(row)) for row in statements)


def reify_statements(
    statements: Sequence[Sequence[int]], *, num_entities: int, num_relations: int
) -> ReifiedGraph:
    if num_entities <= 0 or num_relations <= 0:
        raise ReificationError("num_entities and num_relations must be positive upstream counts")
    rows: list[tuple[int, int, int]] = []
    predicates: list[int] = []
    role_subject = num_relations + ROLE_SUBJECT_OFFSET
    role_object = num_relations + ROLE_OBJECT_OFFSET
    for index, row in enumerate(statements):
        if len(row) < 3:
            raise ReificationError(f"statement needs a main triple: {list(row)}")
        subject, predicate, obj = int(row[0]), int(row[1]), int(row[2])
        if 0 in (subject, predicate, obj):
            raise ReificationError(f"pad id inside the main triple: {list(row)}")
        if not (0 < subject < num_entities and 0 < obj < num_entities and 0 < predicate < num_relations):
            raise ReificationError(f"id out of range for E={num_entities}, R={num_relations}: {list(row)}")
        blank = num_entities + index
        rows.append((blank, role_subject, subject))
        rows.append((blank, role_object, obj))
        for prop, value in qualifier_pairs(row):
            if not (0 < prop < num_relations and 0 < value < num_entities):
                raise ReificationError(f"qualifier id out of range: ({prop}, {value}) in {list(row)}")
            rows.append((blank, prop, value))
        predicates.append(predicate)
    triples = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    return ReifiedGraph(
        triples=triples,
        blank_predicates=np.asarray(predicates, dtype=np.int64),
        num_real_entities=int(num_entities),
        num_relations_base=int(num_relations),
    )


def make_model_config(config: Mapping[str, Any], reified: ReifiedGraph) -> dict[str, Any]:
    """Config for constructing the reified-arm model: StarE's triple mode (no qualifier
    aggregation, no W_q) over the role-edge graph, with the enlarged relation table."""
    out = dict(config)
    out["STAREARGS"] = dict(config["STAREARGS"])
    out["STATEMENT_LEN"] = 3
    out["NUM_RELATIONS"] = reified.num_relations
    out["NUM_ENTITIES"] = reified.num_real_entities
    return out


def make_data_config(config: Mapping[str, Any], reified: ReifiedGraph) -> dict[str, Any]:
    """Config for reciprocals / sampler / evaluation in the reified arm: statements stay
    qualifier-aware (STATEMENT_LEN unchanged), only the relation count grows so reciprocal
    relation ids (r + NUM_RELATIONS) land in the enlarged table."""
    out = dict(config)
    out["STAREARGS"] = dict(config["STAREARGS"])
    out["NUM_RELATIONS"] = reified.num_relations
    out["NUM_ENTITIES"] = reified.num_real_entities
    return out


def graph_summary(reified: ReifiedGraph) -> dict[str, int]:
    return {
        "statement_nodes": reified.num_blank,
        "role_edges": int(reified.triples.shape[0]),
        "role_edges_with_inverses": int(reified.triples.shape[0]) * 2,
        "nodes_total": reified.num_nodes,
        "relations_total": reified.num_relations,
    }
