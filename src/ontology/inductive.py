"""Inductive protocol (charter-hnn-ontology-encoder-v2 §V2-A).

v1 was transductive: every fact's hyperedge sat in the graph during training, and only the
text pairing and the masked member were held out. Here the training graph keeps the full node
set - every entity is known - but carries hyperedges for train-split facts only, so a held-out
fact contributes no structure until evaluation.

Both structural arms get the same treatment: the native view drops held-out hyperedges, and the
reified view drops their blank nodes and role edges, which is the same information removed twice
over.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .hypergraph import Hypergraph, Node
from .schema import Vocab


def subgraph_with_facts(graph: Hypergraph, keep_fact_ids: Iterable[str]) -> Hypergraph:
    """All nodes, but only the named hyperedges."""
    keep = set(keep_fact_ids)
    missing = keep - set(graph.edges)
    if missing:
        raise KeyError(f"unknown fact ids: {sorted(missing)[:5]}")
    out = Hypergraph()
    for node in graph.nodes.values():
        out.add_node(Node.from_dict(node.to_dict()), merge=False)
    for fact_id in graph.edge_ids():  # preserve the original order
        if fact_id in keep:
            edge = graph.edges[fact_id]
            out.add_edge(type(edge)(**{**edge.to_dict(), "members": tuple(edge.members)}))
    return out


def build_training_views(view, train_fact_ids: Sequence[str], vocab: Vocab | None = None):
    """Attach train-only native/reified incidence batches to a WorldView, in place.

    Returns the same view so callers can chain. `view.inductive` becomes True and the arms then
    read the train views while `self.training` is set.
    """
    vocab = vocab or view.vocab
    train_ids = [f for f in view.fact_ids if f in set(train_fact_ids)]
    train_graph = subgraph_with_facts(view.graph, train_ids)
    train_reified_graph = train_graph.reify()
    view.train_native = train_graph.to_incidence(vocab)
    view.train_reified = train_reified_graph.to_incidence(vocab)
    view.train_graph = train_graph
    view.train_reified_graph = train_reified_graph
    view.train_fact_ids = list(view.train_native.edge_ids)
    view.train_fact_index = {f: i for i, f in enumerate(view.train_fact_ids)}
    view.train_node_texts = [
        train_graph.nodes[n].text or train_graph.nodes[n].name for n in view.train_native.node_ids
    ]
    view.train_reified_node_texts = [
        train_reified_graph.nodes[n].text or train_reified_graph.nodes[n].name
        for n in view.train_reified.node_ids
    ]
    return view


def coverage_report(view) -> dict[str, float]:
    """What the inductive split actually removed, for the run summary."""
    if not view.inductive:
        return {"inductive": 0.0}
    return {
        "inductive": 1.0,
        "facts_total": float(view.native.num_edges),
        "facts_in_training_graph": float(view.train_native.num_edges),
        "held_out_facts": float(view.native.num_edges - view.train_native.num_edges),
        "nodes_total": float(view.native.num_nodes),
        "nodes_in_training_graph": float(view.train_native.num_nodes),
        "memberships_removed": float(view.native.nnz - view.train_native.nnz),
    }
