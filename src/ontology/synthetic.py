"""Deterministic synthetic ontology world (contract C3).

One world = schema + gold hypergraph + structured records + natural-language sentences (en/ko)
+ queries (retrieval / completion) + axiom-violation probes + fact-level train/val/test splits.
Paper 1 trains/evaluates the encoder on it; Paper 2 draws its QA split from the same world.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hypergraph import Hyperedge, Hypergraph, Node
from .schema import Axiom, OntologySchema, build_schema
from .templates import (  # noqa: F401 - re-exported for backwards compatibility
    COMPLETION_QUESTION_TEMPLATES,
    NAME_PATTERN,
    RETRIEVAL_QUERY_TEMPLATES,
    SENTENCE_TEMPLATES,
)

ACTIONS: tuple[str, ...] = ("EmergencyStop", "Inspect", "Recalibrate", "ReplacePart", "Monitor")
SENSOR_KINDS: tuple[str, ...] = ("temperature", "vibration", "pressure", "current")
EQUIPMENT_FAMILIES: tuple[str, ...] = ("pump", "compressor", "conveyor", "press")
SEVERITIES: tuple[str, ...] = ("low", "medium", "critical")
SHIFTS: tuple[str, ...] = ("day", "night")


def industrial_maintenance_schema() -> OntologySchema:
    """Domain v1 from docs/hnn-ontology-design-v1.md §7."""
    axioms = (
        Axiom(
            name="z3_vibration_emergency",
            relation="DefectEvent",
            if_={"zone": {"name_eq": "Z3"}, "sensor_secondary": {"attr_eq": ["kind", "vibration"]}},
            then={"action": {"name_eq": "EmergencyStop"}},
        ),
        Axiom(
            name="critical_code_escalates_to_stop",
            relation="Escalation",
            if_={"error_code": {"attr_eq": ["severity", "critical"]}},
            then={"action": {"name_eq": "EmergencyStop"}},
        ),
    )
    return build_schema(
        name="industrial_maintenance_v1",
        entity_types=["Equipment", "Sensor", "Zone", "Technician", "ErrorCode", "Action"],
        relation_types={
            "DefectEvent": {
                "equipment": "Equipment",
                "sensor_primary": "Sensor",
                "sensor_secondary": "Sensor",
                "zone": "Zone",
                "action": "Action",
            },
            "Maintenance": {
                "technician": "Technician",
                "equipment": "Equipment",
                "zone": "Zone",
                "error_code": "ErrorCode",
            },
            "Escalation": {"error_code": "ErrorCode", "technician": "Technician", "action": "Action"},
        },
        axioms=axioms,
    )


@dataclass(frozen=True)
class WorldConfig:
    seed: int = 1
    n_equipment: int = 12
    n_sensors: int = 10
    n_zones: int = 4
    n_technicians: int = 8
    n_error_codes: int = 10
    n_facts: Mapping[str, int] = field(
        default_factory=lambda: {"DefectEvent": 60, "Maintenance": 40, "Escalation": 30}
    )
    languages: tuple[str, ...] = ("en", "ko")
    max_sentence_templates: int = 3
    retrieval_queries_per_fact: int = 2
    completion_roles_per_fact: int = 2
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    n_axiom_probes: int = 10

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["n_facts"] = dict(self.n_facts)
        return payload


WORLD_PRESETS: dict[str, WorldConfig] = {
    "smoke": WorldConfig(
        seed=1,
        n_equipment=5,
        n_sensors=5,
        n_zones=3,
        n_technicians=4,
        n_error_codes=5,
        n_facts={"DefectEvent": 14, "Maintenance": 10, "Escalation": 8},
        retrieval_queries_per_fact=1,
        completion_roles_per_fact=1,
        n_axiom_probes=4,
    ),
    "v1": WorldConfig(
        seed=1,
        n_equipment=24,
        n_sensors=16,
        n_zones=5,
        n_technicians=12,
        n_error_codes=16,
        n_facts={"DefectEvent": 180, "Maintenance": 120, "Escalation": 90},
        retrieval_queries_per_fact=2,
        completion_roles_per_fact=2,
        n_axiom_probes=20,
    ),
}


@dataclass
class Sentence:
    edge_id: str
    lang: str
    template_id: int
    text: str


@dataclass
class Query:
    id: str
    edge_id: str
    kind: str  # "retrieval" | "completion"
    text: str
    lang: str
    masked_role: str | None = None
    answer_node_id: str | None = None


@dataclass
class AxiomProbe:
    id: str
    axiom: str
    edge: dict[str, Any]  # Hyperedge.to_dict() of a fact that violates the axiom
    text: str


@dataclass
class World:
    config: WorldConfig
    schema: OntologySchema
    hypergraph: Hypergraph
    records: list[dict[str, Any]]
    sentences: list[Sentence]
    queries: list[Query]
    axiom_probes: list[AxiomProbe]
    splits: dict[str, list[str]]  # train/val/test -> edge ids
    entity_catalog: list[dict[str, Any]]  # node dicts (type/name/attrs) for extractors

    def edge_split(self, edge_id: str) -> str:
        for name, ids in self.splits.items():
            if edge_id in ids:
                return name
        raise KeyError(edge_id)

    def sentences_for(self, split: str) -> list[Sentence]:
        ids = set(self.splits[split])
        return [s for s in self.sentences if s.edge_id in ids]

    def queries_for(self, split: str, kind: str | None = None) -> list[Query]:
        ids = set(self.splits[split])
        return [q for q in self.queries if q.edge_id in ids and (kind is None or q.kind == kind)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "schema": self.schema.to_dict(),
            "hypergraph": self.hypergraph.to_dict(),
            "records": list(self.records),
            "sentences": [asdict(s) for s in self.sentences],
            "queries": [asdict(q) for q in self.queries],
            "axiom_probes": [asdict(p) for p in self.axiom_probes],
            "splits": {k: list(v) for k, v in self.splits.items()},
            "entity_catalog": list(self.entity_catalog),
        }

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


# ---------------------------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------------------------


def _node_id(node_type: str, name: str) -> str:
    return f"{node_type}:{name}"


def _node_text(node_type: str, name: str, attrs: Mapping[str, Any]) -> str:
    if node_type == "Equipment":
        return f"{name} ({attrs['family']})"
    if node_type == "Sensor":
        return f"{name} ({attrs['kind']} sensor)"
    if node_type == "Zone":
        return f"{name} (zone)"
    if node_type == "Technician":
        return f"{name} ({attrs['shift']} shift technician)"
    if node_type == "ErrorCode":
        return f"{name} ({attrs['severity']} error code)"
    if node_type == "Action":
        return f"{name} (action)"
    return name


def _make_entities(config: WorldConfig, rng: np.random.Generator) -> list[Node]:
    """Entities plus latent regularities (design §7): equipment has a home zone and attached sensors,
    technicians cover zones. These live in `attrs` only (never in node text), so they are learnable
    from other facts but not readable from the entity descriptions."""
    nodes: list[Node] = []

    def add(node_type: str, name: str, attrs: dict[str, Any]) -> None:
        nodes.append(Node(id=_node_id(node_type, name), type=node_type, name=name, text=_node_text(node_type, name, attrs), attrs=attrs))

    zone_names = [f"Z{i + 1}" for i in range(config.n_zones)]
    sensor_names = [f"S-{i + 1:02d}" for i in range(config.n_sensors)]
    for i in range(config.n_equipment):
        attached = [sensor_names[int(j)] for j in rng.choice(len(sensor_names), size=min(3, len(sensor_names)), replace=False)]
        add(
            "Equipment",
            f"EQ-{i + 1:02d}",
            {"family": str(rng.choice(EQUIPMENT_FAMILIES)), "home_zone": zone_names[i % len(zone_names)], "sensors": sorted(attached)},
        )
    for i, name in enumerate(sensor_names):
        # cycle kinds so every kind (incl. vibration) exists even in small worlds
        add("Sensor", name, {"kind": SENSOR_KINDS[i % len(SENSOR_KINDS)]})
    for name in zone_names:
        add("Zone", name, {})
    for i in range(config.n_technicians):
        covered = sorted({zone_names[i % len(zone_names)], zone_names[(i + 1) % len(zone_names)]})
        add("Technician", f"T-{i + 1:02d}", {"shift": str(rng.choice(SHIFTS)), "zones": covered})
    for i in range(config.n_error_codes):
        add("ErrorCode", f"E-{101 + i}", {"severity": SEVERITIES[i % len(SEVERITIES)]})
    for name in ACTIONS:
        add("Action", name, {})
    return nodes


def _pick(rng: np.random.Generator, items: list[Node]) -> Node:
    return items[int(rng.integers(0, len(items)))]


REGULARITY_P = 0.9  # probability that a fact follows its latent regularity (else uniform)
KIND_TO_ACTION = {"temperature": "Inspect", "vibration": "ReplacePart", "pressure": "Recalibrate", "current": "Monitor"}
SEVERITY_TO_ACTION = {"low": "Monitor", "medium": "Inspect"}  # critical -> EmergencyStop by axiom


def _by_name(items: list[Node], names: list[str]) -> list[Node]:
    wanted = set(names)
    return [n for n in items if n.name in wanted]


def _sample_fact(
    relation: str,
    graph: Hypergraph,
    rng: np.random.Generator,
    by_type: Mapping[str, list[Node]],
) -> tuple[tuple[str, str], ...]:
    actions = by_type["Action"]
    stop = next(a for a in actions if a.name == "EmergencyStop")
    regular = bool(rng.random() < REGULARITY_P)
    if relation == "DefectEvent":
        equipment = _pick(rng, by_type["Equipment"])
        sensors = by_type["Sensor"]
        attached = _by_name(sensors, list(equipment.attrs.get("sensors", [])))
        pool = attached if (regular and len(attached) >= 2) else sensors
        primary = _pick(rng, pool)
        secondary = _pick(rng, [s for s in pool if s.id != primary.id])
        zone = _by_name(by_type["Zone"], [equipment.attrs["home_zone"]])[0] if regular else _pick(rng, by_type["Zone"])
        if zone.name == "Z3" and secondary.attrs.get("kind") == "vibration":
            action = stop
        elif regular:
            action = _by_name(actions, [KIND_TO_ACTION[primary.attrs["kind"]]])[0]
        else:
            action = _pick(rng, actions)
        return (
            (equipment.id, "equipment"),
            (primary.id, "sensor_primary"),
            (secondary.id, "sensor_secondary"),
            (zone.id, "zone"),
            (action.id, "action"),
        )
    if relation == "Maintenance":
        equipment = _pick(rng, by_type["Equipment"])
        zone = _by_name(by_type["Zone"], [equipment.attrs["home_zone"]])[0] if regular else _pick(rng, by_type["Zone"])
        covering = [t for t in by_type["Technician"] if zone.name in t.attrs.get("zones", [])]
        technician = _pick(rng, covering) if (regular and covering) else _pick(rng, by_type["Technician"])
        return (
            (technician.id, "technician"),
            (equipment.id, "equipment"),
            (zone.id, "zone"),
            (_pick(rng, by_type["ErrorCode"]).id, "error_code"),
        )
    if relation == "Escalation":
        code = _pick(rng, by_type["ErrorCode"])
        severity = code.attrs.get("severity")
        if severity == "critical":
            action = stop
        elif regular:
            action = _by_name(actions, [SEVERITY_TO_ACTION[severity]])[0]
        else:
            action = _pick(rng, actions)
        return (
            (code.id, "error_code"),
            (_pick(rng, by_type["Technician"]).id, "technician"),
            (action.id, "action"),
        )
    raise KeyError(relation)


def _fill(template: str, graph: Hypergraph, edge: Hyperedge, extra: Mapping[str, str] | None = None) -> str:
    values = {role: graph.nodes[node_id].name for node_id, role in edge.members}
    if extra:
        values.update(extra)
    return template.format(**values)


def _template_roles(template: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\{(\w+)\}", template))


def _unique_under_roles(graph: Hypergraph, edge: Hyperedge, roles: tuple[str, ...]) -> bool:
    signature = tuple(edge.member(role) for role in roles)
    for other in graph.edges.values():
        if other.id == edge.id or other.relation != edge.relation:
            continue
        if tuple(other.member(role) for role in roles) == signature:
            return False
    return True


def generate_world(config: WorldConfig) -> World:
    schema = industrial_maintenance_schema()
    rng = np.random.default_rng(config.seed)
    graph = Hypergraph()
    entities = _make_entities(config, rng)
    for node in entities:
        graph.add_node(node)
    by_type = {t: graph.nodes_of_type(t) for t in schema.entity_types}

    # facts (unique keys) --------------------------------------------------------------
    seen_keys: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    records: list[dict[str, Any]] = []
    for relation, count in config.n_facts.items():
        produced = 0
        attempts = 0
        while produced < count:
            attempts += 1
            if attempts > count * 50:
                raise RuntimeError(f"could not sample {count} unique {relation} facts; enlarge the entity pools")
            members = _sample_fact(relation, graph, rng, by_type)
            key = (relation, members)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            edge = Hyperedge(id=f"{relation}#{produced:04d}", relation=relation, members=members, provenance="synthetic")
            graph.add_edge(edge)
            records.append({"relation": relation, **{role: graph.nodes[n].name for n, role in members}})
            produced += 1
    graph.validate(schema)

    # sentences ------------------------------------------------------------------------
    sentences: list[Sentence] = []
    for edge in graph.edges.values():
        for lang in config.languages:
            templates = SENTENCE_TEMPLATES[edge.relation][lang][: config.max_sentence_templates]
            for template_id, template in enumerate(templates):
                sentences.append(Sentence(edge_id=edge.id, lang=lang, template_id=template_id, text=_fill(template, graph, edge)))
        edge.text = _fill(SENTENCE_TEMPLATES[edge.relation]["en"][0], graph, edge)

    # retrieval queries ----------------------------------------------------------------
    queries: list[Query] = []
    for edge in graph.edges.values():
        candidates: list[tuple[str, str]] = []
        for lang in config.languages:
            for template in RETRIEVAL_QUERY_TEMPLATES[edge.relation][lang]:
                if _unique_under_roles(graph, edge, _template_roles(template)):
                    candidates.append((lang, template))
        order = rng.permutation(len(candidates)) if candidates else []
        for pick in list(order)[: config.retrieval_queries_per_fact]:
            lang, template = candidates[int(pick)]
            queries.append(
                Query(
                    id=f"q{len(queries):05d}",
                    edge_id=edge.id,
                    kind="retrieval",
                    text=_fill(template, graph, edge),
                    lang=lang,
                )
            )
        if not candidates:  # fall back to the full canonical sentence in a question form
            queries.append(
                Query(
                    id=f"q{len(queries):05d}",
                    edge_id=edge.id,
                    kind="retrieval",
                    text="Which fact is this: " + graph.fact_to_text(edge) + "?",
                    lang="en",
                )
            )

    # completion queries ---------------------------------------------------------------
    for edge in graph.edges.values():
        roles = list(edge.roles())
        order = rng.permutation(len(roles))
        for pick in list(order)[: config.completion_roles_per_fact]:
            masked_role = roles[int(pick)]
            context = ", ".join(
                f"{role}={graph.nodes[n].name}" for n, role in edge.members if role != masked_role
            )
            lang = str(config.languages[int(pick) % len(config.languages)])
            text = COMPLETION_QUESTION_TEMPLATES[lang].format(relation=edge.relation, context=context, masked_role=masked_role)
            queries.append(
                Query(
                    id=f"q{len(queries):05d}",
                    edge_id=edge.id,
                    kind="completion",
                    text=text,
                    lang=lang,
                    masked_role=masked_role,
                    answer_node_id=edge.member(masked_role),
                )
            )

    # axiom probes (facts that violate an axiom) ----------------------------------------
    probes: list[AxiomProbe] = []
    stop = next(a for a in by_type["Action"] if a.name == "EmergencyStop")
    non_stop = [a for a in by_type["Action"] if a.id != stop.id]
    vibration = [s for s in by_type["Sensor"] if s.attrs.get("kind") == "vibration"]
    critical = [c for c in by_type["ErrorCode"] if c.attrs.get("severity") == "critical"]
    z3 = [z for z in by_type["Zone"] if z.name == "Z3"]
    for i in range(config.n_axiom_probes):
        if i % 2 == 0 and vibration and z3:
            secondary = _pick(rng, vibration)
            primary = _pick(rng, [s for s in by_type["Sensor"] if s.id != secondary.id])
            members = (
                (_pick(rng, by_type["Equipment"]).id, "equipment"),
                (primary.id, "sensor_primary"),
                (secondary.id, "sensor_secondary"),
                (z3[0].id, "zone"),
                (_pick(rng, non_stop).id, "action"),
            )
            edge = Hyperedge(id=f"probe#{i:03d}", relation="DefectEvent", members=members, provenance="axiom_probe")
            probes.append(AxiomProbe(id=edge.id, axiom="z3_vibration_emergency", edge=edge.to_dict(), text=_fill(SENTENCE_TEMPLATES["DefectEvent"]["en"][0], graph, edge)))
        elif critical:
            members = (
                (_pick(rng, critical).id, "error_code"),
                (_pick(rng, by_type["Technician"]).id, "technician"),
                (_pick(rng, non_stop).id, "action"),
            )
            edge = Hyperedge(id=f"probe#{i:03d}", relation="Escalation", members=members, provenance="axiom_probe")
            probes.append(AxiomProbe(id=edge.id, axiom="critical_code_escalates_to_stop", edge=edge.to_dict(), text=_fill(SENTENCE_TEMPLATES["Escalation"]["en"][0], graph, edge)))

    # splits (fact-level, per relation, deterministic) -----------------------------------
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for relation in schema.relation_names:
        ids = [e.id for e in graph.edges.values() if e.relation == relation]
        perm = [ids[int(i)] for i in rng.permutation(len(ids))]
        n = len(perm)
        n_val = max(1, int(round(n * config.split[1]))) if n >= 3 else 0
        n_test = max(1, int(round(n * config.split[2]))) if n >= 3 else 0
        splits["test"].extend(perm[:n_test])
        splits["val"].extend(perm[n_test : n_test + n_val])
        splits["train"].extend(perm[n_test + n_val :])

    catalog = [{"type": n.type, "name": n.name, "attrs": dict(n.attrs), "text": n.text} for n in entities]
    return World(
        config=config,
        schema=schema,
        hypergraph=graph,
        records=records,
        sentences=sentences,
        queries=queries,
        axiom_probes=probes,
        splits=splits,
        entity_catalog=catalog,
    )


def load_preset(name: str, *, seed: int | None = None) -> World:
    if name not in WORLD_PRESETS:
        raise KeyError(f"unknown world preset {name!r}; expected one of {sorted(WORLD_PRESETS)}")
    config = WORLD_PRESETS[name]
    if seed is not None:
        config = WorldConfig(**{**config.to_dict(), "seed": int(seed)})
    return generate_world(config)
