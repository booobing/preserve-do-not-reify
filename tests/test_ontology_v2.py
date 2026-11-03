"""Contracts for the Paper 1 v2 factors (charter-hnn-ontology-encoder-v2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.ontology.baselines import ModelConfig, build_model, parameter_report
from src.ontology.contrastive import TrainConfig, WorldView, train_alignment
from src.ontology.inductive import build_training_views, coverage_report, subgraph_with_facts
from src.ontology.real_data import Statement, build_hypergraph, build_schema, load_wd50k
from src.ontology.run_encoder_benchmark_v2 import FACTORS, judge, main
from src.ontology.schema import SchemaError
from src.ontology.schema import build_schema as build_ontology_schema
from src.ontology.synthetic import load_preset

DATA_ROOT = Path("data")


@pytest.fixture(scope="module")
def smoke_world():
    world = load_preset("smoke")
    return world, WorldView.from_world(world)


# ---- V2-A inductive ---------------------------------------------------------------------------


def test_subgraph_keeps_all_nodes_and_only_named_facts(smoke_world) -> None:
    world, _ = smoke_world
    train_ids = world.splits["train"]
    sub = subgraph_with_facts(world.hypergraph, train_ids)
    assert sub.num_nodes == world.hypergraph.num_nodes
    assert set(sub.edge_ids()) == set(train_ids)
    sub.validate(world.schema)
    with pytest.raises(KeyError):
        subgraph_with_facts(world.hypergraph, ["nope"])


def test_training_views_hide_held_out_structure(smoke_world) -> None:
    world, _ = smoke_world
    view = build_training_views(WorldView.from_world(world), world.splits["train"])
    assert view.inductive
    held_out = set(world.splits["val"]) | set(world.splits["test"])
    assert not held_out & set(view.train_native.edge_ids)
    assert set(view.train_native.edge_ids) == set(world.splits["train"])
    # the node set is unchanged, so only structure is removed
    assert view.train_native.num_nodes == view.native.num_nodes
    assert view.train_native.nnz < view.native.nnz
    # the reified training view drops the blank nodes of held-out facts too
    blanks = {n for n in view.train_reified.node_ids if n.startswith("__b__")}
    assert blanks == {f"__b__{f}" for f in world.splits["train"]}
    report = coverage_report(view)
    assert report["held_out_facts"] == len(held_out)
    assert coverage_report(WorldView.from_world(world)) == {"inductive": 0.0}


@pytest.mark.parametrize("arm", ["hyper", "reified_binary"])
def test_arms_read_training_views_only_while_training(smoke_world, arm: str) -> None:
    world, _ = smoke_world
    view = build_training_views(WorldView.from_world(world), world.splits["train"])
    cfg = ModelConfig(d=16, layers=1, text_dim=16, text_buckets=256)
    model = build_model(arm, view, cfg, seed=1)
    model.arm.train()
    train_emb = model.encode_facts(view)
    assert train_emb.shape[0] == len(world.splits["train"])
    model.arm.eval()
    eval_emb = model.encode_facts(view)
    assert eval_emb.shape[0] == view.native.num_edges


def test_inductive_training_runs_and_indexes_rows_correctly(smoke_world) -> None:
    world, _ = smoke_world
    view = build_training_views(WorldView.from_world(world), world.splits["train"])
    cfg = ModelConfig(d=16, layers=1, text_dim=16, text_buckets=256)
    model = build_model("hyper", view, cfg, seed=1)
    history = train_alignment(model, view, world, TrainConfig(steps=8, batch_size=4, seed=1), device="cpu")
    assert history[-1]["loss"] < history[0]["loss"]


# ---- V2-B head parity -------------------------------------------------------------------------


def test_head_parity_changes_the_readout_but_not_the_parameter_count(smoke_world) -> None:
    _, view = smoke_world
    off = ModelConfig(d=16, layers=1, text_dim=16, text_buckets=256, head_parity=False)
    on = ModelConfig(d=16, layers=1, text_dim=16, text_buckets=256, head_parity=True)
    assert parameter_report(view, off, arms=("hyper", "reified_binary")) == parameter_report(
        view, on, arms=("hyper", "reified_binary")
    )
    a = build_model("reified_binary", view, off, seed=2)
    b = build_model("reified_binary", view, on, seed=2)
    assert not a.arm.head_parity and b.arm.head_parity
    with torch.no_grad():
        assert not torch.allclose(a.encode_facts(view), b.encode_facts(view))
    # parity routes the fact embedding through edge_out, the head the hyper arm uses
    with torch.no_grad():
        out = b.arm.encoder(view.reified, b.text_tower(view.reified_node_texts))
        rows = [view.reified.node_index[f"__b__{f}"] for f in view.fact_ids]
        expected = torch.nn.functional.normalize(b.arm.encoder.edge_out(out["node_hidden"][rows]), dim=-1)
        assert torch.allclose(b.encode_facts(view), expected, atol=1e-5)


# ---- V2-D real data ---------------------------------------------------------------------------


def test_statement_parsing_and_arity() -> None:
    s = Statement.parse("Q12174,P2293,Q18052481,P4,Q99")
    assert s.subject == "Q12174" and s.predicate == "P2293" and s.object == "Q18052481"
    assert s.qualifiers == (("P4", "Q99"),) and s.arity == 3
    assert Statement.parse("A,B,C").arity == 2
    with pytest.raises(ValueError):
        Statement.parse("A,B")
    with pytest.raises(ValueError):
        Statement.parse("A,B,C,D")  # dangling qualifier property


def test_statements_become_hyperedges_with_qualifier_roles() -> None:
    statements = [Statement.parse("Q1,P10,Q2,P5,Q3"), Statement.parse("Q1,P10,Q4")]
    schema = build_schema(statements, name="tiny")
    graph = build_hypergraph(statements)
    graph.validate(schema)
    first, second = list(graph.edges.values())
    assert first.relation == "P10" and dict((r, n) for n, r in first.members).keys() >= {"subject", "object", "P5"}
    assert second.arity == 2  # the same relation may appear without qualifiers
    relation = schema.relation("P10")
    assert relation.required_role_names == ("subject", "object")
    assert "P5" in relation.role_names and relation.role_type("P5") == "Entity"


def test_optional_roles_are_a_schema_concept_not_a_loophole() -> None:
    schema = build_ontology_schema(
        "opt", ["A"], {"R": {"x": "A", "y": "A", "z": "A"}}, optional_roles={"R": ["z"]}
    )
    assert schema.relation("R").required_role_names == ("x", "y")
    with pytest.raises(SchemaError):  # cannot make a relation effectively unary
        build_ontology_schema("bad", ["A"], {"R": {"x": "A", "y": "A"}}, optional_roles={"R": ["y"]})


@pytest.mark.skipif(not (DATA_ROOT / "wd50k_100" / "train.txt").exists(), reason="WD50K not fetched")
def test_wd50k_loads_offline_and_is_fully_n_ary() -> None:
    dataset = load_wd50k("wd50k_100", allow_download=False)
    stats = dataset.stats()
    assert stats["n_ary_fraction"] == 1.0
    assert stats["mean_arity"] > 3.0
    assert stats["train_facts"] == 22738 and stats["test_facts"] == 5297
    ids = set(dataset.splits["train"]) | set(dataset.splits["val"]) | set(dataset.splits["test"])
    assert ids == set(dataset.hypergraph.edge_ids())
    assert not (set(dataset.splits["train"]) & set(dataset.splits["test"]))
    dataset.hypergraph.validate(dataset.schema)


# ---- CLI --------------------------------------------------------------------------------------


def test_factor_table_is_single_factor_against_the_v1_baseline() -> None:
    base = FACTORS["v1_baseline"]
    for name in ("v2a_inductive", "v2b_head", "v2c_text"):
        differences = [k for k in base if FACTORS[name][k] != base[k]]
        assert len(differences) == 1, (name, differences)
    assert FACTORS["v2_combined"]["real"] is False
    assert FACTORS["v2d_real"]["real"] is True


def test_v2_judgment_uses_the_factor_metric_pair() -> None:
    def row(arm, seed, recall, acc, mrr=0.0, acc10=0.0):
        return {
            "arm": arm, "seed": seed,
            "retrieval": {"recall@1": 0.0, "recall@5": recall, "recall@10": 1.0, "mrr": 0.0, "n": 10.0},
            "completion": {"acc@1": acc, "acc@3": 1.0, "acc@10": acc10, "mrr": mrr, "n": 10.0},
        }

    seeds = [1, 2, 3, 4, 5]
    win = [row("hyper", s, 0.9, 0.5) for s in seeds] + [row("reified_binary", s, 0.5, 0.3) for s in seeds]
    assert judge(win, "v2a_inductive")["verdict"] == "success"
    half = [row("hyper", s, 0.9, 0.3) for s in seeds] + [row("reified_binary", s, 0.5, 0.3) for s in seeds]
    assert judge(half, "v2a_inductive")["verdict"] == "partial"
    real_rows = (
        [row("hyper", s, 0.0, 0.0, mrr=0.3, acc10=0.5) for s in seeds]
        + [row("reified_binary", s, 0.0, 0.0, mrr=0.2, acc10=0.4) for s in seeds]
    )
    verdict = judge(real_rows, "v2d_real")
    assert verdict["metrics"] == ["completion.mrr", "completion.acc@10"]
    assert verdict["verdict"] == "success"


def test_v2_cli_smoke_writes_artifacts(workspace_tmp_dir) -> None:
    out = workspace_tmp_dir("ontology_v2_cli_smoke")
    code = main([
        "--factor", "v2a_inductive", "--seeds", "1", "--steps", "4", "--batch-size", "4",
        "--d", "16", "--layers", "1", "--text-dim", "16", "--text-buckets", "256",
        "--world-preset", "smoke", "--device", "cpu", "--out", str(out),
    ])
    assert code == 0
    factor_dir = out / "v2a_inductive"
    config = json.loads((factor_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["inductive"]["inductive"] == 1.0
    assert config["params_by_arm"]["hyper"] == config["params_by_arm"]["reified_binary"]
    rows = json.loads((factor_dir / "summary.json").read_text(encoding="utf-8"))
    assert {r["arm"] for r in rows} == {"hyper", "reified_binary"}
    assert all(r["inductive"]["inductive"] == 1.0 for r in rows)
    judgment = json.loads((factor_dir / "judgment.json").read_text(encoding="utf-8"))
    assert judgment["verdict"] == "incomplete"
    assert judgment["metrics"] == ["retrieval.recall@5", "completion.acc@1"]


def test_v2e_judges_hyper_against_the_depth_steelman() -> None:
    """charter v2e: the strengthened control is a *different arm*, not a looser rule.

    The depth objection ("the gap is effective depth, not representation") was half-closed - v1
    measured depth without head parity, v2b parity without depth. This factor combines them, so its
    control has to be the deep arm while every other factor keeps the original judged pair.
    """
    from src.ontology.run_encoder_benchmark_v2 import FACTORS, judge

    assert FACTORS["v2e_depth_parity"]["control_arm"] == "reified_binary_deep"
    assert FACTORS["v2e_depth_parity"]["head_parity"] is True
    for name, spec in FACTORS.items():
        if name != "v2e_depth_parity":
            assert "control_arm" not in spec, f"{name} must keep the original judged pair"

    def row(arm, seed, recall, acc):
        return {"arm": arm, "seed": seed,
                "retrieval": {"recall@5": recall}, "completion": {"acc@1": acc}}

    seeds = [1, 2, 3, 4, 5]
    rows = [row("hyper", s, 0.86, 0.50) for s in seeds]
    rows += [row("reified_binary_deep", s, 0.73, 0.43) for s in seeds]
    rows += [row("reified_binary", s, 0.99, 0.99) for s in seeds]  # must be ignored here
    report = judge(rows, "v2e_depth_parity")
    assert report["control"] == "reified_binary_deep"
    assert report["verdict"] == "success"

    # the same rows judged as v2b_head compare against the shallow arm instead, and lose
    assert judge(rows, "v2b_head")["control"] == "reified_binary"
    assert judge(rows, "v2b_head")["verdict"] == "fail"
