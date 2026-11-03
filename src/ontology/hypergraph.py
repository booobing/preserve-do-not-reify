"""Hypergraph container, invariants, incidence export, reification (contract C2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch import Tensor

from .schema import (
    REIFIED_ROLES,
    RESERVED_BLANK_TYPE,
    ROLE_EDGE_PREFIX,
    Axiom,
    OntologySchema,
    Vocab,
    evaluate_condition,
)


class HypergraphInvariantError(ValueError):
    """Raised when a hypergraph violates the C2 invariants against a schema."""


@dataclass
class Node:
    id: str
    type: str
    name: str
    text: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "name": self.name, "text": self.text, "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Node:
        return cls(
            id=str(payload["id"]),
            type=str(payload["type"]),
            name=str(payload["name"]),
            text=str(payload.get("text", "")),
            attrs=dict(payload.get("attrs", {})),
        )


@dataclass
class Hyperedge:
    id: str
    relation: str
    members: tuple[tuple[str, str], ...]  # (node_id, role)
    text: str = ""
    provenance: str = ""

    @property
    def arity(self) -> int:
        return len(self.members)

    def member(self, role: str) -> str:
        for node_id, member_role in self.members:
            if member_role == role:
                return node_id
        raise KeyError(f"hyperedge {self.id!r} has no member with role {role!r}")

    def roles(self) -> tuple[str, ...]:
        return tuple(role for _, role in self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "relation": self.relation,
            "members": [[node_id, role] for node_id, role in self.members],
            "text": self.text,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Hyperedge:
        return cls(
            id=str(payload["id"]),
            relation=str(payload["relation"]),
            members=tuple((str(n), str(r)) for n, r in payload["members"]),
            text=str(payload.get("text", "")),
            provenance=str(payload.get("provenance", "")),
        )


@dataclass
class IncidenceBatch:
    """Tensor view of a hypergraph (contract C2 `to_incidence`)."""

    node_ids: list[str]
    edge_ids: list[str]
    node_type_ids: Tensor  # [V] long
    edge_type_ids: Tensor  # [E] long
    incidence: Tensor  # [nnz, 2] long: (node_idx, edge_idx), sorted by (edge_idx, node_idx)
    role_ids: Tensor  # [nnz] long
    node_index: dict[str, int]
    edge_index: dict[str, int]

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edge_ids)

    @property
    def nnz(self) -> int:
        return int(self.incidence.shape[0])

    def to(self, device: torch.device | str) -> IncidenceBatch:
        return IncidenceBatch(
            node_ids=self.node_ids,
            edge_ids=self.edge_ids,
            node_type_ids=self.node_type_ids.to(device),
            edge_type_ids=self.edge_type_ids.to(device),
            incidence=self.incidence.to(device),
            role_ids=self.role_ids.to(device),
            node_index=self.node_index,
            edge_index=self.edge_index,
        )

    def without_memberships(self, drop: Iterable[tuple[str, int]]) -> IncidenceBatch:
        """Return a copy with the given (edge_id, role_id) memberships removed (used for completion)."""
        drop_set = set(drop)
        if not drop_set:
            return self
        keep_list = [True] * self.nnz
        edge_idx = self.incidence[:, 1].tolist()
        roles = self.role_ids.tolist()
        for position, (e_idx, role_id) in enumerate(zip(edge_idx, roles, strict=True)):
            if (self.edge_ids[e_idx], role_id) in drop_set:
                keep_list[position] = False
        keep = torch.tensor(keep_list, dtype=torch.bool, device=self.incidence.device)
        return IncidenceBatch(
            node_ids=self.node_ids,
            edge_ids=self.edge_ids,
            node_type_ids=self.node_type_ids,
            edge_type_ids=self.edge_type_ids,
            incidence=self.incidence[keep],
            role_ids=self.role_ids[keep],
            node_index=self.node_index,
            edge_index=self.edge_index,
        )


def edge_key(edge: Hyperedge, nodes: Mapping[str, Node]) -> tuple[str, frozenset[tuple[str, str, str]]]:
    """Identity-independent key used by extraction fidelity: (relation, {(type, name, role)})."""
    return (
        edge.relation,
        frozenset((nodes[node_id].type, nodes[node_id].name, role) for node_id, role in edge.members),
    )


class Hypergraph:
    """Ordered container of typed nodes and role-labelled hyperedges."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Hyperedge] = {}

    # ---- construction ---------------------------------------------------------------------
    def add_node(self, node: Node, *, merge: bool = True) -> Node:
        if node.id in self.nodes:
            if not merge:
                raise HypergraphInvariantError(f"duplicate node id {node.id!r}")
            existing = self.nodes[node.id]
            if existing.type != node.type or existing.name != node.name:
                raise HypergraphInvariantError(f"node id {node.id!r} re-added with different type/name")
            if node.text and not existing.text:
                existing.text = node.text
            for key, value in node.attrs.items():
                existing.attrs.setdefault(key, value)
            return existing
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Hyperedge) -> Hyperedge:
        if edge.id in self.edges:
            raise HypergraphInvariantError(f"duplicate hyperedge id {edge.id!r}")
        if edge.arity == 0:
            raise HypergraphInvariantError(f"hyperedge {edge.id!r} is empty")
        for node_id, _role in edge.members:
            if node_id not in self.nodes:
                raise HypergraphInvariantError(f"hyperedge {edge.id!r} references unknown node {node_id!r}")
        self.edges[edge.id] = edge
        return edge

    def __len__(self) -> int:
        return len(self.edges)

    def __iter__(self) -> Iterator[Hyperedge]:
        return iter(self.edges.values())

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def node_ids(self) -> list[str]:
        return list(self.nodes)

    def edge_ids(self) -> list[str]:
        return list(self.edges)

    def nodes_of_type(self, node_type: str) -> list[Node]:
        return [node for node in self.nodes.values() if node.type == node_type]

    def edges_of_node(self, node_id: str) -> list[Hyperedge]:
        return [edge for edge in self.edges.values() if any(n == node_id for n, _ in edge.members)]

    def copy(self) -> Hypergraph:
        return Hypergraph.from_dict(self.to_dict())

    # ---- invariants -----------------------------------------------------------------------
    def validate(self, schema: OntologySchema) -> None:
        entity_types = set(schema.entity_types)
        for node in self.nodes.values():
            if node.type not in entity_types and node.type != RESERVED_BLANK_TYPE:
                raise HypergraphInvariantError(f"node {node.id!r} has unknown type {node.type!r}")
        for edge in self.edges.values():
            if edge.arity == 0:
                raise HypergraphInvariantError(f"hyperedge {edge.id!r} is empty")
            for node_id, _ in edge.members:
                if node_id not in self.nodes:
                    raise HypergraphInvariantError(f"hyperedge {edge.id!r} references unknown node {node_id!r}")
            if edge.relation.startswith(ROLE_EDGE_PREFIX):
                self._validate_reified_edge(edge)
                continue
            relation = schema.relation(edge.relation) if edge.relation in schema.relation_names else None
            if relation is None:
                raise HypergraphInvariantError(f"hyperedge {edge.id!r} has unknown relation {edge.relation!r}")
            roles = edge.roles()
            if len(set(roles)) != len(roles):
                raise HypergraphInvariantError(f"hyperedge {edge.id!r} has duplicate roles")
            declared = set(relation.role_names)
            unknown = set(roles) - declared
            if unknown:
                raise HypergraphInvariantError(
                    f"hyperedge {edge.id!r} uses roles {sorted(unknown)} that relation "
                    f"{relation.name!r} does not declare"
                )
            missing = set(relation.required_role_names) - set(roles)
            if missing:
                raise HypergraphInvariantError(
                    f"hyperedge {edge.id!r} is missing required roles {sorted(missing)}"
                )
            for node_id, role in edge.members:
                expected = relation.role_type(role)
                actual = self.nodes[node_id].type
                if actual != expected:
                    raise HypergraphInvariantError(
                        f"hyperedge {edge.id!r} role {role!r} expects {expected!r}, node {node_id!r} is {actual!r}"
                    )

    def _validate_reified_edge(self, edge: Hyperedge) -> None:
        if edge.arity != 2 or set(edge.roles()) != set(REIFIED_ROLES):
            raise HypergraphInvariantError(f"reified edge {edge.id!r} must have roles {REIFIED_ROLES}")
        subject = self.nodes[edge.member("subject")]
        if subject.type != RESERVED_BLANK_TYPE:
            raise HypergraphInvariantError(f"reified edge {edge.id!r} subject must be a blank node")

    # ---- axioms ---------------------------------------------------------------------------
    def axiom_applies(self, axiom: Axiom, edge: Hyperedge) -> bool:
        if edge.relation != axiom.relation:
            return False
        return all(
            evaluate_condition(cond, name=self.nodes[edge.member(role)].name, attrs=self.nodes[edge.member(role)].attrs)
            for role, cond in axiom.if_.items()
        )

    def axiom_satisfied(self, axiom: Axiom, edge: Hyperedge) -> bool:
        if not self.axiom_applies(axiom, edge):
            return True
        return all(
            evaluate_condition(cond, name=self.nodes[edge.member(role)].name, attrs=self.nodes[edge.member(role)].attrs)
            for role, cond in axiom.then.items()
        )

    def axiom_violations(self, schema: OntologySchema) -> list[tuple[str, str]]:
        violations: list[tuple[str, str]] = []
        for edge in self.edges.values():
            if edge.relation.startswith(ROLE_EDGE_PREFIX):
                continue
            for axiom in schema.axioms:
                if not self.axiom_satisfied(axiom, edge):
                    violations.append((edge.id, axiom.name))
        return violations

    # ---- tensor export --------------------------------------------------------------------
    def to_incidence(self, vocab: Vocab) -> IncidenceBatch:
        node_ids = self.node_ids()
        edge_ids = self.edge_ids()
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        edge_index = {edge_id: idx for idx, edge_id in enumerate(edge_ids)}
        node_type_ids = torch.tensor(
            [vocab.node_type_id(self.nodes[n].type) for n in node_ids], dtype=torch.long
        ).reshape(-1)
        edge_type_ids = torch.tensor(
            [vocab.edge_type_id(self.edges[e].relation) for e in edge_ids], dtype=torch.long
        ).reshape(-1)
        pairs: list[tuple[int, int, int]] = []
        for edge_id in edge_ids:
            edge = self.edges[edge_id]
            e_idx = edge_index[edge_id]
            for node_id, role in edge.members:
                pairs.append((e_idx, node_index[node_id], vocab.role_id(role)))
        pairs.sort()
        if pairs:
            incidence = torch.tensor([[n, e] for e, n, _ in pairs], dtype=torch.long)
            role_ids = torch.tensor([r for _, _, r in pairs], dtype=torch.long)
        else:
            incidence = torch.zeros((0, 2), dtype=torch.long)
            role_ids = torch.zeros((0,), dtype=torch.long)
        return IncidenceBatch(
            node_ids=node_ids,
            edge_ids=edge_ids,
            node_type_ids=node_type_ids,
            edge_type_ids=edge_type_ids,
            incidence=incidence,
            role_ids=role_ids,
            node_index=node_index,
            edge_index=edge_index,
        )

    # ---- reification ----------------------------------------------------------------------
    def reify(self) -> Hypergraph:
        """Lossless binary view: each n-ary edge -> blank node + n role edges (contract C2)."""
        out = Hypergraph()
        for node in self.nodes.values():
            out.add_node(Node.from_dict(node.to_dict()))
        for edge in self.edges.values():
            blank_id = f"__b__{edge.id}"
            # The blank node carries the relation name as its text (the rdf:type triple of classic
            # reification) and nothing else, so the binary view has the same information as the
            # n-ary view: relation identity + role-labelled members. No sentence text leaks in.
            out.add_node(Node(id=blank_id, type=RESERVED_BLANK_TYPE, name=edge.id, text=edge.relation))
            for node_id, role in edge.members:
                out.add_edge(
                    Hyperedge(
                        id=Hypergraph.role_edge_id(edge.id, role),
                        relation=f"{ROLE_EDGE_PREFIX}{role}",
                        members=((blank_id, "subject"), (node_id, "object")),
                        provenance=edge.id,
                    )
                )
        return out

    @staticmethod
    def blank_node_id(edge_id: str) -> str:
        return f"__b__{edge_id}"

    @staticmethod
    def role_edge_id(edge_id: str, role: str) -> str:
        return f"{edge_id}::{role}"

    # ---- serialization --------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "hyperedges": [edge.to_dict() for edge in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Hypergraph:
        graph = cls()
        for node in payload.get("nodes", []):
            graph.add_node(Node.from_dict(node), merge=False)
        for edge in payload.get("hyperedges", []):
            graph.add_edge(Hyperedge.from_dict(edge))
        return graph

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json_text(cls, text: str) -> Hypergraph:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_json_file(cls, path: str | Path) -> Hypergraph:
        return cls.from_json_text(Path(path).read_text(encoding="utf-8"))

    # ---- text views -----------------------------------------------------------------------
    def fact_to_text(self, edge: Hyperedge) -> str:
        """Canonical serialization `Relation(role=Name, ...)` used by the flat_text arm and RAG."""
        parts = ", ".join(f"{role}={self.nodes[node_id].name}" for node_id, role in edge.members)
        return f"{edge.relation}({parts})"

    def fact_to_masked_text(self, edge: Hyperedge, masked_role: str) -> str:
        parts = ", ".join(
            f"{role}=?" if role == masked_role else f"{role}={self.nodes[node_id].name}"
            for node_id, role in edge.members
        )
        return f"{edge.relation}({parts})"

    def fact_to_triples_text(self, edge: Hyperedge) -> str:
        """Reified triple serialization used by the GraphRAG-style baseline."""
        blank = Hypergraph.blank_node_id(edge.id)
        lines = [f"({blank}, rdf:type, {edge.relation})"]
        lines.extend(f"({blank}, {role}, {self.nodes[node_id].name})" for node_id, role in edge.members)
        return "\n".join(lines)

    def fact_to_hyperedge_json(self, edge: Hyperedge) -> str:
        """HyperGraphRAG-style JSON serialization used by the text-serialized hypergraph baseline."""
        payload = {
            "type": edge.relation,
            "participants": [
                {"role": role, "node": self.nodes[node_id].name, "node_type": self.nodes[node_id].type}
                for node_id, role in edge.members
            ],
        }
        return json.dumps(payload, ensure_ascii=False)
