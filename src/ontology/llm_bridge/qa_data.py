"""Paper 2 QA split built from the same synthetic world (charter-hnn-llm-bridge-v1, P5 data contract).

Each item: question text, gold answer string (entity name), gold fact id(s), the masked role, the
number of constraints the question states (>=3 -> multi-constraint bucket), the language, and the
split the fact belongs to. Axiom probes are exported separately as (question, forbidden_answer,
axiom) so the violation rate can be measured without an LLM in the loop.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..hypergraph import Hyperedge
from ..synthetic import World

MULTI_CONSTRAINT_MIN = 3


@dataclass
class QAItem:
    id: str
    question: str
    answer: str
    answer_node_id: str
    gold_edge_ids: list[str]
    relation: str
    masked_role: str
    n_constraints: int
    multi_constraint: bool
    lang: str
    split: str


@dataclass
class AxiomProbeItem:
    id: str
    question: str
    forbidden_answer: str
    required_answer: str | None
    axiom: str
    relation: str
    lang: str


def build_qa_items(world: World) -> list[QAItem]:
    graph = world.hypergraph
    items: list[QAItem] = []
    for query in world.queries:
        if query.kind != "completion" or not query.masked_role or not query.answer_node_id:
            continue
        edge = graph.edges[query.edge_id]
        n_constraints = edge.arity - 1
        items.append(
            QAItem(
                id=f"qa-{query.id}",
                question=query.text,
                answer=graph.nodes[query.answer_node_id].name,
                answer_node_id=query.answer_node_id,
                gold_edge_ids=[edge.id],
                relation=edge.relation,
                masked_role=query.masked_role,
                n_constraints=n_constraints,
                multi_constraint=n_constraints >= MULTI_CONSTRAINT_MIN,
                lang=query.lang,
                split=world.edge_split(edge.id),
            )
        )
    return items


def build_axiom_probe_items(world: World) -> list[AxiomProbeItem]:
    """Questions whose answer is forced by an axiom; answering with the probe's action violates it."""
    graph = world.hypergraph
    items: list[AxiomProbeItem] = []
    for probe in world.axiom_probes:
        edge = Hyperedge.from_dict(probe.edge)
        axiom = next(a for a in world.schema.axioms if a.name == probe.axiom)
        then_role, then_cond = next(iter(axiom.then.items()))
        required = then_cond.get("name_eq")
        context = ", ".join(
            f"{role}={graph.nodes[n].name}" for n, role in edge.members if role != then_role
        )
        items.append(
            AxiomProbeItem(
                id=f"probe-{probe.id}",
                question=f"In the {edge.relation} where {context}, what is the {then_role}?",
                forbidden_answer=graph.nodes[edge.member(then_role)].name,
                required_answer=str(required) if required is not None else None,
                axiom=probe.axiom,
                relation=edge.relation,
                lang="en",
            )
        )
    return items


def export_qa_dataset(world: World, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = build_qa_items(world)
    probes = build_axiom_probe_items(world)
    for split in ("train", "val", "test"):
        rows = [asdict(i) for i in items if i.split == split]
        with (out / f"qa_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "axiom_probes.jsonl").open("w", encoding="utf-8") as handle:
        for probe in probes:
            handle.write(json.dumps(asdict(probe), ensure_ascii=False) + "\n")
    summary = {
        "n_items": len(items),
        "n_multi_constraint": sum(i.multi_constraint for i in items),
        "n_probes": len(probes),
        "per_split": {s: sum(i.split == s for i in items) for s in ("train", "val", "test")},
    }
    (out / "qa_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
