"""Data -> Hypergraph extractors and extraction fidelity (contract C4).

- StructuredRecordExtractor: rows/JSON records with a {relation: {role: column}} mapping.
- TemplateTextExtractor: deterministic slot-grammar reverse matching over sentence templates.
- LLMExtractor: protocol only (v2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .hypergraph import Hyperedge, Hypergraph, Node, edge_key
from .schema import OntologySchema
from .templates import NAME_PATTERN, SENTENCE_TEMPLATES


def _node_id(node_type: str, name: str) -> str:
    return f"{node_type}:{name}"


class EntityCatalog:
    """Optional (type, name) -> attrs/text lookup so extracted nodes carry attributes for axioms."""

    def __init__(self, entries: Iterable[Mapping[str, Any]] = ()) -> None:
        self._entries: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in entries:
            self._entries[(str(entry["type"]), str(entry["name"]))] = {
                "attrs": dict(entry.get("attrs", {})),
                "text": str(entry.get("text", "")),
            }

    def node(self, node_type: str, name: str) -> Node:
        info = self._entries.get((node_type, name), {"attrs": {}, "text": ""})
        return Node(id=_node_id(node_type, name), type=node_type, name=name, text=info["text"] or name, attrs=dict(info["attrs"]))


class LLMExtractor(Protocol):
    """v2 extension point: an LLM-backed extractor must satisfy this protocol (no implementation in v1)."""

    def extract(self, texts: Sequence[str]) -> Hypergraph: ...


@dataclass
class StructuredRecordExtractor:
    schema: OntologySchema
    mapping: Mapping[str, Mapping[str, str]] | None = None  # {relation: {role: column}}
    catalog: EntityCatalog | None = None
    relation_column: str = "relation"

    def _column(self, relation: str, role: str) -> str:
        if self.mapping and relation in self.mapping:
            return self.mapping[relation].get(role, role)
        return role

    def extract(self, records: Iterable[Mapping[str, Any]]) -> Hypergraph:
        graph = Hypergraph()
        catalog = self.catalog or EntityCatalog()
        for index, record in enumerate(records):
            relation_name = str(record[self.relation_column])
            relation = self.schema.relation(relation_name)
            members: list[tuple[str, str]] = []
            for role in relation.roles:
                value = record.get(self._column(relation_name, role.name))
                if value is None or str(value) == "":
                    raise ValueError(f"record {index} missing role {role.name!r} for {relation_name!r}")
                node = catalog.node(role.type, str(value))
                graph.add_node(node)
                members.append((node.id, role.name))
            graph.add_edge(
                Hyperedge(
                    id=f"{relation_name}#rec{index:05d}",
                    relation=relation_name,
                    members=tuple(members),
                    provenance=f"record:{index}",
                )
            )
        graph.validate(self.schema)
        return graph


@dataclass
class _CompiledTemplate:
    relation: str
    lang: str
    template_id: int
    regex: re.Pattern[str]
    roles: tuple[str, ...]


def compile_templates(
    templates: Mapping[str, Mapping[str, Sequence[str]]], *, name_pattern: str = NAME_PATTERN
) -> list[_CompiledTemplate]:
    compiled: list[_CompiledTemplate] = []
    for relation, by_lang in templates.items():
        for lang, items in by_lang.items():
            for template_id, template in enumerate(items):
                roles = tuple(re.findall(r"\{(\w+)\}", template))
                pattern = "^"
                cursor = 0
                for match in re.finditer(r"\{(\w+)\}", template):
                    pattern += re.escape(template[cursor : match.start()])
                    pattern += f"(?P<{match.group(1)}>{name_pattern})"
                    cursor = match.end()
                pattern += re.escape(template[cursor:]) + "$"
                compiled.append(
                    _CompiledTemplate(
                        relation=relation,
                        lang=lang,
                        template_id=template_id,
                        regex=re.compile(pattern),
                        roles=roles,
                    )
                )
    return compiled


@dataclass
class TemplateTextExtractor:
    schema: OntologySchema
    templates: Mapping[str, Mapping[str, Sequence[str]]] | None = None
    catalog: EntityCatalog | None = None

    def __post_init__(self) -> None:
        self._compiled = compile_templates(self.templates or SENTENCE_TEMPLATES)

    def parse(self, text: str) -> tuple[str, dict[str, str]] | None:
        cleaned = text.strip()
        for item in self._compiled:
            match = item.regex.match(cleaned)
            if match:
                return item.relation, {role: match.group(role) for role in item.roles}
        return None

    def extract(self, texts: Sequence[str]) -> Hypergraph:
        graph = Hypergraph()
        catalog = self.catalog or EntityCatalog()
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        count = 0
        for index, text in enumerate(texts):
            parsed = self.parse(text)
            if parsed is None:
                continue
            relation_name, slots = parsed
            relation = self.schema.relation(relation_name)
            members: list[tuple[str, str]] = []
            for role in relation.roles:
                node = catalog.node(role.type, slots[role.name])
                graph.add_node(node)
                members.append((node.id, role.name))
            key = (relation_name, tuple(sorted(members)))
            if key in seen:  # several sentences describe one fact
                continue
            seen.add(key)
            graph.add_edge(
                Hyperedge(
                    id=f"{relation_name}#txt{count:05d}",
                    relation=relation_name,
                    members=tuple(members),
                    text=text,
                    provenance=f"text:{index}",
                )
            )
            count += 1
        graph.validate(self.schema)
        return graph


def extraction_fidelity(pred: Hypergraph, gold: Hypergraph) -> dict[str, float]:
    """Hyperedge-level P/R/F1 over identity-independent keys (contract C4)."""
    pred_keys = {edge_key(e, pred.nodes) for e in pred.edges.values()}
    gold_keys = {edge_key(e, gold.nodes) for e in gold.edges.values()}
    tp = len(pred_keys & gold_keys)
    precision = tp / len(pred_keys) if pred_keys else 0.0
    recall = tp / len(gold_keys) if gold_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_edges": float(len(pred_keys)),
        "gold_edges": float(len(gold_keys)),
    }
