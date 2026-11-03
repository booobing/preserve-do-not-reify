"""Contract tests for src/ontology/stare_bridge/native_encoder.py (docs/charter-stare-native-encoder-v1.md)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ontology.stare_bridge import native_encoder as ne
from src.ontology.stare_bridge import runtime as rt
from src.ontology.stare_bridge import upstream as ups
from src.ontology.stare_bridge.reify_stare import ReificationError

REPO = Path(__file__).resolve().parents[1]
E, R = 10, 5
ROWS = [
    [1, 2, 3, 0, 0, 0, 0],
    [4, 1, 5, 2, 6, 0, 0],
    [7, 3, 8, 4, 9, 4, 2],  # repeated qualifier property kept, as upstream keeps it
]
SUBJ, OBJ = R, R + 1


def _members_of(inc: ne.NativeIncidence, edge: int) -> list[tuple[int, int]]:
    mask = inc.edge_idx == edge
    return list(zip(inc.node_idx[mask].tolist(), inc.role_idx[mask].tolist(), strict=True))


def test_hyper_view_one_edge_per_statement_with_all_participants():
    inc = ne.build_native_incidence(ROWS, num_entities=E, num_relations=R, mode="hyper")
    assert (inc.num_edges, inc.num_memberships, inc.num_blank, inc.num_nodes) == (3, 2 + 3 + 4, 0, E)
    assert inc.edge_type.tolist() == [2, 1, 3]
    assert _members_of(inc, 0) == [(1, SUBJ), (3, OBJ)]
    assert _members_of(inc, 1) == [(4, SUBJ), (5, OBJ), (6, 2)]
    assert _members_of(inc, 2) == [(7, SUBJ), (8, OBJ), (9, 4), (2, 4)]
    assert (inc.num_roles, inc.num_edge_types) == (R + 2, R + R + 2)


def test_prune_view_hides_qualifiers_from_the_encoder():
    inc = ne.build_native_incidence(ROWS, num_entities=E, num_relations=R, mode="prune")
    assert (inc.num_edges, inc.num_memberships) == (3, 6)
    assert set(inc.role_idx.tolist()) == {SUBJ, OBJ}
    assert _members_of(inc, 2) == [(7, SUBJ), (8, OBJ)]


def test_reified_view_matches_chapter5_reify():
    inc = ne.build_native_incidence(ROWS, num_entities=E, num_relations=R, mode="reified")
    assert (inc.num_blank, inc.num_nodes) == (3, E + 3)
    assert inc.blank_predicates.tolist() == [2, 1, 3]
    assert inc.num_edges == 2 + 3 + 4 and inc.num_memberships == 2 * inc.num_edges
    # edge types are "role edge of rho" = R + role; members are (blank, subject), (member, object)
    assert inc.edge_type[:2].tolist() == [R + SUBJ, R + OBJ]
    assert _members_of(inc, 0) == [(E, SUBJ), (1, OBJ)]
    assert _members_of(inc, 8) == [(E + 2, SUBJ), (2, OBJ)] and inc.edge_type[8] == R + 4
    assert inc.edge_type.max() < inc.num_edge_types and inc.role_idx.max() < inc.num_roles
    blanks = inc.node_idx[inc.node_idx >= E]
    assert set(blanks.tolist()) == {E, E + 1, E + 2}


def test_views_reject_bad_rows_and_unknown_modes():
    for bad in ([[1, 0, 3]], [[1, 2, 30]], [[1, 2, 3, 2, 40]], [[1, 2, 3, 4, 0]]):
        for mode in ("hyper", "reified"):
            with pytest.raises(ReificationError):
                ne.build_native_incidence(bad, num_entities=E, num_relations=R, mode=mode)
    with pytest.raises(ValueError):
        ne.build_native_incidence(ROWS, num_entities=E, num_relations=R, mode="clique")


def test_arm_names_map_to_modes():
    assert ne.NATIVE_ARMS == {"native_hyper": "hyper", "native_reified": "reified", "native_prune": "prune"}


# ---- judge script generalisation ---------------------------------------------------------------


def _judge():
    spec = importlib.util.spec_from_file_location("judge_native", REPO / ".harness/scripts/judge_stare_input_representation.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["judge_native"] = module
    spec.loader.exec_module(module)
    return module


def _summary(root: Path, arm: str, seed: int, mrr: float, h10: float):
    d = root / arm / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    metrics = {"mrr": mrr, "hits_at 1": mrr * 0.7, "hits_at 3": mrr, "hits_at 10": h10, "mr": 100.0}
    (d / "summary.json").write_text(json.dumps({"epochs_done": 401, "last_eval": {"epoch": 401, "metrics": metrics},
                                                "best_eval": {"epoch": 401, "metrics": {"mrr": mrr}}}), encoding="utf-8")


def test_judge_any_pair_names_the_verdict_and_files_by_arm(tmp_path):
    j = _judge()
    for seed in range(1, 6):
        _summary(tmp_path, "native_hyper", seed, 0.66, 0.78)
        _summary(tmp_path, "native_reified", seed, 0.64, 0.76)
        _summary(tmp_path, "native_prune", seed, 0.66 + (0.01 if seed % 2 else -0.01), 0.78)
    a = j.build_report(tmp_path, epochs=401, base="native_hyper", other="native_reified")
    assert a["verdict"] == "native_hyper_wins" and "reproduction" not in a
    assert a["wins_native_reified_native_hyper_ties"]["mrr"] == (0, 5, 0)
    j.write_outputs(a, tmp_path)
    assert (tmp_path / "judgment_native_reified_vs_native_hyper.json").exists()
    assert not (tmp_path / "judgment.json").exists(), "the default file name is kept for the original pair only"
    b = j.build_report(tmp_path, epochs=401, base="native_hyper", other="native_prune")
    assert b["verdict"] == "no_consistent_order"


def test_judge_default_pair_reproduces_the_committed_stare_verdict():
    j = _judge()
    root = REPO / "outputs" / "stare_input_representation_v1" / "judged"
    if not (root / "judgment.json").exists():
        pytest.skip("committed StarE judged outputs are not present")
    report = j.build_report(root, epochs=401)
    committed = json.loads((root / "judgment.json").read_text(encoding="utf-8"))
    assert report["verdict"] == committed["verdict"] == "hyper_wins"
    for metric in ("mrr", "hits_at 10"):
        assert list(report["wins_reified_hyper_ties"][metric]) == list(committed["wins_reified_hyper_ties"][metric])
        assert report["paired_reified_minus_hyper"][metric]["mean"] == pytest.approx(committed["paired_reified_minus_hyper"][metric]["mean"])
    assert report["reproduction"]["within_tolerance"] is True


# ---- integration: needs the pinned checkout and torch-geometric --------------------------------

HAVE_PYG = importlib.util.find_spec("torch_geometric") is not None
HAVE_CLONE = (ups.REPO_ROOT / "third_party" / "StarE").exists()
needs_upstream = pytest.mark.skipif(not (HAVE_PYG and HAVE_CLONE), reason="pinned StarE checkout or torch-geometric missing")


@needs_upstream
def test_native_arms_build_score_real_entities_and_train_one_step():
    up = ups.load_upstream()
    config = up.default_config()
    config["DATASET"] = "wd50k_100"
    config["DEVICE"] = torch.device("cpu")
    data = up.load_data(config)
    train = list(data["train"])
    n_ent, n_rel = data["n_entities"], data["n_relations"]
    config["NUM_ENTITIES"], config["NUM_RELATIONS"] = n_ent, n_rel
    graph_repr = up.DataManager.get_alternative_graph_repr(train, config)
    sub = torch.tensor([train[0][0], train[1][0]])
    rel = torch.tensor([train[0][1], train[1][1]])
    quals = torch.tensor([train[0][3:], train[1][3:]])
    params = {}
    for mode in ne.MODES:
        inc = ne.build_native_incidence(train, num_entities=n_ent, num_relations=n_rel, mode=mode)
        torch.manual_seed(0)
        model = ne.build_native_model(up, graph_repr, config, inc)
        assert ne.is_native(model) and model.conv1 is None and model.conv2 is None
        params[mode] = sum(p.numel() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            scores = model(sub, rel, quals)
        assert scores.shape == (2, n_ent), "candidates are the real entities only"
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.native_layers[0].phi1.weight.detach().clone()
        target = torch.zeros(2, n_ent)
        target[0, train[0][2]] = 1.0
        target[1, train[1][2]] = 1.0
        model.loss(model(sub, rel, quals), target).backward()
        opt.step()
        assert not torch.equal(before, model.native_layers[0].phi1.weight.detach()), f"{mode}: encoder receives gradient"
    d = config["EMBEDDING_DIM"]
    assert params["hyper"] == params["prune"], "same modules, only the graph differs"
    assert params["reified"] == params["hyper"] + d, "the reified arm adds only the shared statement-node offset"
    inc = ne.build_native_incidence(train, num_entities=n_ent, num_relations=n_rel, mode="hyper")
    assert inc.num_memberships == 2 * len(train) + 32439, "charter §2: 32,439 kept qualifier pairs in train"


@needs_upstream
def test_runner_native_arm_smoke_and_resume_on_cpu(tmp_path):
    from src.ontology.stare_bridge import runner

    out = tmp_path / "native_reified_seed0"
    common = ["--arm", "native_reified", "--seed", "0", "--out", str(out), "--device", "cpu", "--eval-every", "1",
              "--chunk", "1", "--eval-limit", "4", "--max-steps-per-epoch", "2", "--no-lock"]
    assert runner.main(["--epochs", "1", *common]) == 0
    assert runner.main(["--epochs", "2", "--resume", *common]) == 0
    rows = rt.read_jsonl(out / "metrics.jsonl")
    assert [r["epoch"] for r in rows] == [1, 2] and all("eval" in r for r in rows)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["graph"]["mode"] == "reified" and summary["graph"]["statement_nodes"] == 22738
    assert summary["arm"] == "native_reified" and np.isfinite(summary["last_eval"]["metrics"]["mrr"])
