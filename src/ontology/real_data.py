"""WD50K real-data loader (charter-hnn-ontology-encoder-v2 §V2-D).

WD50K is a set of Wikidata statements: a primary triple plus qualifier pairs. A statement maps
onto one hyperedge without any reshaping, which is the point - the dataset is n-ary in the wild,
not n-ary because we made it so:

    Q12174,P2293,Q18052481,P4  ->  hyperedge typed P2293 with members
                                   (Q12174, role=subject), (Q18052481, role=object),
                                   (P4's value, role=P4)

The reified view is then exactly what Wikidata itself does internally with statement nodes, so
the comparison is between two representations that both exist in practice.

Files come from the StarE release; `fetch_wd50k` downloads them once into `data/wd50k_*/` and
every later call reads from disk, so judged runs need no network.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .hypergraph import Hyperedge, Hypergraph, Node
from .schema import OntologySchema, RelationType, Role

WD50K_BASE = "https://raw.githubusercontent.com/migalkin/StarE/master/data/clean"
WD50K_VARIANTS = ("wd50k", "wd50k_33", "wd50k_66", "wd50k_100")
SPLITS = ("train", "valid", "test")
DEFAULT_ROOT = Path("data")

SUBJECT_ROLE = "subject"
OBJECT_ROLE = "object"
ENTITY_TYPE = "Entity"


@dataclass
class Statement:
    """One WD50K row: a primary triple plus zero or more (qualifier property, value) pairs."""

    subject: str
    predicate: str
    object: str
    qualifiers: tuple[tuple[str, str], ...]

    @property
    def arity(self) -> int:
        return 2 + len(self.qualifiers)

    @classmethod
    def parse(cls, line: str) -> Statement:
        parts = [p.strip() for p in line.strip().split(",") if p.strip()]
        if len(parts) < 3:
            raise ValueError(f"statement needs at least a triple, got {line!r}")
        if (len(parts) - 3) % 2 != 0:
            raise ValueError(f"qualifiers must come in property/value pairs: {line!r}")
        quals = tuple((parts[i], parts[i + 1]) for i in range(3, len(parts), 2))
        return cls(subject=parts[0], predicate=parts[1], object=parts[2], qualifiers=quals)


def fetch_wd50k(variant: str = "wd50k_100", root: Path | str = DEFAULT_ROOT, *, allow_download: bool = True) -> Path:
    """Return the local directory for a variant, downloading the three split files once."""
    if variant not in WD50K_VARIANTS:
        raise KeyError(f"unknown WD50K variant {variant!r}; expected one of {WD50K_VARIANTS}")
    target = Path(root) / variant
    target.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = target / f"{split}.txt"
        if path.exists():
            continue
        if not allow_download:
            raise FileNotFoundError(f"{path} is missing and downloads are disabled")
        url = f"{WD50K_BASE}/{variant}/statements/{split}.txt"
        if not url.startswith("https://raw.githubusercontent.com/"):  # pragma: no cover - constant
            raise ValueError(f"refusing to fetch from {url!r}")
        request = urllib.request.Request(url, headers={"User-Agent": "hypergraph-eggroll research"})  # noqa: S310
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - https host checked above
            path.write_bytes(response.read())
    return target


def read_statements(path: Path | str) -> list[Statement]:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [Statement.parse(line) for line in lines if line.strip()]


def qualifier_properties(statements: Iterable[Statement]) -> list[str]:
    seen: list[str] = []
    for statement in statements:
        for prop, _ in statement.qualifiers:
            if prop not in seen:
                seen.append(prop)
    return sorted(seen)


def build_schema(statements: Sequence[Statement], name: str = "wd50k") -> OntologySchema:
    """One relation type per predicate; roles are subject, object, and the qualifier properties.

    WD50K has no entity typing, so every node is `Entity`. Roles carry the semantics instead,
    which is what the role-aware encoder consumes.
    """
    by_predicate: dict[str, list[str]] = {}
    for statement in statements:
        roles = by_predicate.setdefault(statement.predicate, [SUBJECT_ROLE, OBJECT_ROLE])
        for prop, _ in statement.qualifiers:
            if prop not in roles:
                roles.append(prop)
    relations = tuple(
        RelationType(
            name=pred,
            roles=tuple(
                # subject/object are the primary triple and always present; qualifier properties
                # vary from statement to statement, so they are optional roles.
                Role(name=r, type=ENTITY_TYPE, optional=r not in (SUBJECT_ROLE, OBJECT_ROLE))
                for r in roles
            ),
        )
        for pred, roles in sorted(by_predicate.items())
    )
    schema = OntologySchema(name=name, entity_types=(ENTITY_TYPE,), relation_types=relations)
    schema.validate()
    return schema


def build_hypergraph(statements: Sequence[Statement], *, prefix: str = "s") -> Hypergraph:
    graph = Hypergraph()

    def node(entity_id: str) -> str:
        node_id = f"{ENTITY_TYPE}:{entity_id}"
        if node_id not in graph.nodes:
            graph.add_node(Node(id=node_id, type=ENTITY_TYPE, name=entity_id, text=entity_id))
        return node_id

    for index, statement in enumerate(statements):
        members = [(node(statement.subject), SUBJECT_ROLE), (node(statement.object), OBJECT_ROLE)]
        used = {SUBJECT_ROLE, OBJECT_ROLE}
        for prop, value in statement.qualifiers:
            if prop in used:  # a repeated qualifier property cannot be a second role on one edge
                continue
            used.add(prop)
            members.append((node(value), prop))
        graph.add_edge(
            Hyperedge(
                id=f"{prefix}{index:06d}",
                relation=statement.predicate,
                members=tuple(members),
                provenance="wd50k",
            )
        )
    return graph


@dataclass
class RealDataset:
    name: str
    schema: OntologySchema
    hypergraph: Hypergraph
    splits: dict[str, list[str]]

    def stats(self) -> dict[str, float]:
        arities = [e.arity for e in self.hypergraph.edges.values()]
        return {
            "facts": float(len(arities)),
            "entities": float(self.hypergraph.num_nodes),
            "relations": float(len(self.schema.relation_types)),
            "mean_arity": sum(arities) / len(arities) if arities else 0.0,
            "max_arity": float(max(arities)) if arities else 0.0,
            "n_ary_fraction": sum(a > 2 for a in arities) / len(arities) if arities else 0.0,
            **{f"{split}_facts": float(len(ids)) for split, ids in self.splits.items()},
        }


def load_wd50k(
    variant: str = "wd50k_100",
    root: Path | str = DEFAULT_ROOT,
    *,
    allow_download: bool = True,
    max_facts_per_split: int | None = None,
) -> RealDataset:
    """Load one variant into a single hypergraph plus split membership over fact ids."""
    directory = fetch_wd50k(variant, root, allow_download=allow_download)
    per_split: dict[str, list[Statement]] = {}
    for split in SPLITS:
        statements = read_statements(directory / f"{split}.txt")
        if max_facts_per_split is not None:
            statements = statements[:max_facts_per_split]
        per_split[split] = statements

    ordered: list[Statement] = []
    splits: dict[str, list[str]] = {}
    for split in SPLITS:
        start = len(ordered)
        ordered.extend(per_split[split])
        splits[split] = [f"s{i:06d}" for i in range(start, len(ordered))]
    splits["val"] = splits.pop("valid")  # match the synthetic world's naming

    schema = build_schema(ordered, name=variant)
    graph = build_hypergraph(ordered)
    graph.validate(schema)
    return RealDataset(name=variant, schema=schema, hypergraph=graph, splits=splits)
