"""Ontology schema contract (docs/hnn-ontology-contracts-v1.md C1).

An ontology here is: entity types, n-ary relation types with named/typed roles, and axioms
(hyperedge-level rules). It is deliberately JSON-native rather than OWL/RDF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

RESERVED_BLANK_TYPE = "__blank__"
ROLE_EDGE_PREFIX = "__role_edge__"
REIFIED_ROLES = ("subject", "object")


class SchemaError(ValueError):
    """Raised when an ontology schema violates the C1 invariants."""


@dataclass(frozen=True)
class Role:
    """A named, typed slot of a relation.

    `optional` exists because real n-ary data does not have a fixed arity per relation: two
    Wikidata statements with the same predicate can carry different qualifier properties. The
    synthetic world leaves every role required, so its invariants are unchanged.
    """

    name: str
    type: str
    optional: bool = False


@dataclass(frozen=True)
class RelationType:
    name: str
    roles: tuple[Role, ...]

    @property
    def arity(self) -> int:
        return len(self.roles)

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(role.name for role in self.roles)

    @property
    def required_role_names(self) -> tuple[str, ...]:
        return tuple(role.name for role in self.roles if not role.optional)

    @property
    def min_arity(self) -> int:
        return len(self.required_role_names)

    def role_type(self, role_name: str) -> str:
        for role in self.roles:
            if role.name == role_name:
                return role.type
        raise KeyError(f"relation {self.name!r} has no role {role_name!r}")


# Conditions are small JSON objects evaluated against a node:
#   {"name_eq": "Z3"} | {"name_in": ["Z1","Z3"]} | {"attr_eq": ["kind", "vibration"]}
Condition = Mapping[str, Any]


def evaluate_condition(condition: Condition, *, name: str, attrs: Mapping[str, Any]) -> bool:
    for key, value in condition.items():
        if key == "name_eq":
            if name != value:
                return False
        elif key == "name_in":
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set, frozenset)):
                raise SchemaError("name_in expects a list of names")
            if name not in set(value):
                return False
        elif key == "attr_eq":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise SchemaError("attr_eq expects [attribute, value]")
            attr_key, attr_value = value
            if attrs.get(attr_key) != attr_value:
                return False
        else:
            raise SchemaError(f"unknown condition operator {key!r}")
    return True


@dataclass(frozen=True)
class Axiom:
    name: str
    relation: str
    if_: Mapping[str, Condition]
    then: Mapping[str, Condition]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relation": self.relation,
            "if": {role: dict(cond) for role, cond in self.if_.items()},
            "then": {role: dict(cond) for role, cond in self.then.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Axiom:
        return cls(
            name=str(payload["name"]),
            relation=str(payload["relation"]),
            if_=dict(payload.get("if", {})),
            then=dict(payload.get("then", {})),
        )


@dataclass(frozen=True)
class Vocab:
    """Shared id spaces for node types / edge types / roles, including reified reserved entries.

    Both the `hyper` and `reified_binary` arms use the same Vocab so embedding tables have the
    same size and the parameter counts match.
    """

    node_types: tuple[str, ...]
    edge_types: tuple[str, ...]
    roles: tuple[str, ...]

    def node_type_id(self, name: str) -> int:
        return self.node_types.index(name)

    def edge_type_id(self, name: str) -> int:
        return self.edge_types.index(name)

    def role_id(self, name: str) -> int:
        return self.roles.index(name)


@dataclass(frozen=True)
class OntologySchema:
    name: str
    entity_types: tuple[str, ...]
    relation_types: tuple[RelationType, ...]
    axioms: tuple[Axiom, ...] = field(default_factory=tuple)

    # ---- validation -----------------------------------------------------------------------
    def validate(self) -> None:
        if not self.name:
            raise SchemaError("schema name must be non-empty")
        if len(set(self.entity_types)) != len(self.entity_types):
            raise SchemaError("entity_types contain duplicates")
        if RESERVED_BLANK_TYPE in self.entity_types:
            raise SchemaError(f"{RESERVED_BLANK_TYPE!r} is reserved and cannot be an entity type")
        names = [relation.name for relation in self.relation_types]
        if len(set(names)) != len(names):
            raise SchemaError("relation_types contain duplicate names")
        for relation in self.relation_types:
            if relation.name.startswith(ROLE_EDGE_PREFIX):
                raise SchemaError(f"relation name {relation.name!r} uses the reserved prefix")
            if relation.arity < 2:
                raise SchemaError(f"relation {relation.name!r} must have arity >= 2")
            if relation.min_arity < 2:
                raise SchemaError(f"relation {relation.name!r} must keep at least 2 required roles")
            role_names = relation.role_names
            if len(set(role_names)) != len(role_names):
                raise SchemaError(f"relation {relation.name!r} has duplicate role names")
            for role in relation.roles:
                if role.type not in self.entity_types:
                    raise SchemaError(
                        f"relation {relation.name!r} role {role.name!r} has unknown type {role.type!r}"
                    )
        for axiom in self.axioms:
            try:
                relation = self.relation(axiom.relation)
            except KeyError as exc:
                raise SchemaError(f"axiom {axiom.name!r} references unknown relation") from exc
            for role_name in list(axiom.if_) + list(axiom.then):
                if role_name not in relation.role_names:
                    raise SchemaError(f"axiom {axiom.name!r} references unknown role {role_name!r}")
            for cond in list(axiom.if_.values()) + list(axiom.then.values()):
                evaluate_condition(cond, name="", attrs={})  # raises on unknown operator

    # ---- lookups --------------------------------------------------------------------------
    def relation(self, name: str) -> RelationType:
        for relation in self.relation_types:
            if relation.name == name:
                return relation
        raise KeyError(f"unknown relation {name!r}")

    @property
    def relation_names(self) -> tuple[str, ...]:
        return tuple(relation.name for relation in self.relation_types)

    def role_names(self) -> tuple[str, ...]:
        seen: list[str] = []
        for relation in self.relation_types:
            for role in relation.roles:
                if role.name not in seen:
                    seen.append(role.name)
        return tuple(seen)

    def vocab(self) -> Vocab:
        """Vocab shared by the native and reified views (contract C2/C5)."""
        base_roles = self.role_names()
        roles = tuple(base_roles) + tuple(r for r in REIFIED_ROLES if r not in base_roles)
        node_types = tuple(self.entity_types) + (RESERVED_BLANK_TYPE,)
        edge_types = tuple(self.relation_names) + tuple(f"{ROLE_EDGE_PREFIX}{role}" for role in base_roles)
        return Vocab(node_types=node_types, edge_types=edge_types, roles=roles)

    # ---- serialization --------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity_types": list(self.entity_types),
            "relation_types": [
                {
                    "name": r.name,
                    "roles": [
                        {"name": role.name, "type": role.type, **({"optional": True} if role.optional else {})}
                        for role in r.roles
                    ],
                }
                for r in self.relation_types
            ],
            "axioms": [axiom.to_dict() for axiom in self.axioms],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OntologySchema:
        relation_types = tuple(
            RelationType(
                name=str(item["name"]),
                roles=tuple(
                    Role(
                        name=str(role["name"]),
                        type=str(role["type"]),
                        optional=bool(role.get("optional", False)),
                    )
                    for role in item["roles"]
                ),
            )
            for item in payload.get("relation_types", [])
        )
        schema = cls(
            name=str(payload["name"]),
            entity_types=tuple(str(t) for t in payload.get("entity_types", [])),
            relation_types=relation_types,
            axioms=tuple(Axiom.from_dict(a) for a in payload.get("axioms", [])),
        )
        schema.validate()
        return schema

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json_text(cls, text: str) -> OntologySchema:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_json_file(cls, path: str | Path) -> OntologySchema:
        return cls.from_json_text(Path(path).read_text(encoding="utf-8"))


def build_schema(
    name: str,
    entity_types: Iterable[str],
    relation_types: Mapping[str, Mapping[str, str]],
    axioms: Iterable[Axiom] = (),
    optional_roles: Mapping[str, Iterable[str]] | None = None,
) -> OntologySchema:
    """Convenience constructor: relation_types = {relation: {role: entity_type}} (ordered)."""
    optional = {rel: set(names) for rel, names in (optional_roles or {}).items()}
    relations = tuple(
        RelationType(
            name=rel,
            roles=tuple(
                Role(name=role, type=etype, optional=role in optional.get(rel, ()))
                for role, etype in roles.items()
            ),
        )
        for rel, roles in relation_types.items()
    )
    schema = OntologySchema(
        name=name,
        entity_types=tuple(entity_types),
        relation_types=relations,
        axioms=tuple(axioms),
    )
    schema.validate()
    return schema
