"""Contract tests for src/ontology/stare_bridge (docs/charter-stare-input-representation-v1.md §9).

Pure tests always run. Tests that need the pinned checkout (third_party/StarE) and torch-geometric
are skipped when either is missing, so the suite stays green on a machine without the clone.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ontology.stare_bridge import reify_stare as rs
from src.ontology.stare_bridge import runtime as rt
from src.ontology.stare_bridge import upstream as ups

REPO = Path(__file__).resolve().parents[1]
SHIMS = REPO / "src" / "ontology" / "stare_bridge" / "shims"
E, R = 10, 5  # toy id spaces: entities 1..9, relations 1..4 (0 is the upstream pad id)
ROWS = [
    [1, 2, 3, 0, 0, 0, 0],  # no qualifiers
    [4, 1, 5, 2, 6, 0, 0],  # one qualifier
    [7, 3, 8, 4, 9, 4, 2],  # two qualifiers with the same property, kept as upstream keeps them
]


def _load_shim(name: str):
    spec = importlib.util.spec_from_file_location(f"shim_{name}", SHIMS / name / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---- reification -------------------------------------------------------------------------------


def test_reify_counts_ids_and_role_edges():
    g = rs.reify_statements(ROWS, num_entities=E, num_relations=R)
    assert (g.num_blank, g.num_real_entities, g.num_nodes) == (3, E, E + 3)
    assert (g.role_subject, g.role_object, g.num_relations) == (R, R + 1, R + 2)
    assert g.triples.shape == (rs.expected_role_edges(ROWS), 3) == (2 + 3 + 4, 3)
    assert g.blank_predicates.tolist() == [2, 1, 3]
    assert sorted(set(g.triples[:, 0].tolist())) == [E, E + 1, E + 2]
    first = g.triples[g.triples[:, 0] == E]
    assert first.tolist() == [[E, R, 1], [E, R + 1, 3]]
    third = g.triples[g.triples[:, 0] == E + 2]
    assert third.tolist() == [[E + 2, R, 7], [E + 2, R + 1, 8], [E + 2, 4, 9], [E + 2, 4, 2]]
    assert g.triples[:, 2].max() < E, "members are real entities, never blanks"
    assert g.triples[:, 1].max() < R + 2
    assert g.blank_id(2) == E + 2


def test_reify_rejects_padding_in_main_triple_half_pairs_and_out_of_range_ids():
    for bad in ([[1, 0, 3]], [[0, 2, 3]], [[1, 2, 3, 4, 0]], [[1, 2, 3, 0, 9]], [[1, 2, 30]], [[1, 9, 3]], [[1, 2, 3, 2, 40]]):
        with pytest.raises(rs.ReificationError):
            rs.reify_statements(bad, num_entities=E, num_relations=R)
    with pytest.raises(rs.ReificationError):
        rs.reify_statements([[1, 2, 3, 4]], num_entities=E, num_relations=R)  # odd qualifier tail


def test_reify_is_deterministic_and_follows_statement_order():
    a = rs.reify_statements(ROWS, num_entities=E, num_relations=R)
    b = rs.reify_statements(ROWS, num_entities=E, num_relations=R)
    np.testing.assert_array_equal(a.triples, b.triples)
    swapped = rs.reify_statements(ROWS[::-1], num_entities=E, num_relations=R)
    assert swapped.blank_predicates.tolist() == [3, 1, 2]
    assert swapped.triples[0].tolist() == [E, R, 7]


def test_configs_change_only_statement_len_and_id_counts():
    base = {"STATEMENT_LEN": -1, "NUM_RELATIONS": R, "NUM_ENTITIES": E, "MAX_QPAIRS": 7, "STAREARGS": {"LAYERS": 2}}
    g = rs.reify_statements(ROWS, num_entities=E, num_relations=R)
    model_cfg, data_cfg = rs.make_model_config(base, g), rs.make_data_config(base, g)
    assert model_cfg["STATEMENT_LEN"] == 3 and data_cfg["STATEMENT_LEN"] == -1
    assert model_cfg["NUM_RELATIONS"] == data_cfg["NUM_RELATIONS"] == R + 2
    assert model_cfg["NUM_ENTITIES"] == data_cfg["NUM_ENTITIES"] == E
    assert model_cfg["MAX_QPAIRS"] == data_cfg["MAX_QPAIRS"] == 7
    assert base["STATEMENT_LEN"] == -1 and base["NUM_RELATIONS"] == R
    model_cfg["STAREARGS"]["LAYERS"] = 1
    assert base["STAREARGS"]["LAYERS"] == 2, "STAREARGS must be copied, not shared"
    assert rs.graph_summary(g) == {
        "statement_nodes": 3, "role_edges": 9, "role_edges_with_inverses": 18, "nodes_total": E + 3, "relations_total": R + 2,
    }


# ---- shims -------------------------------------------------------------------------------------


def test_scatter_shim_matches_explicit_loops():
    sc = _load_shim("torch_scatter")
    gen = torch.Generator().manual_seed(0)
    src = torch.randn(12, 3, generator=gen)
    idx = torch.tensor([0, 2, 2, 1, 0, 3, 3, 3, 1, 0, 2, 5])
    n = 7
    add = sc.scatter_add(src, idx, dim=0, dim_size=n)
    mean = sc.scatter_mean(src, idx, dim=0, dim_size=n)
    mx, _ = sc.scatter_max(src, idx, dim=0, dim_size=n)
    assert add.shape == mean.shape == mx.shape == (n, 3)
    for i in range(n):
        rows = src[idx == i]
        if len(rows):
            assert torch.allclose(add[i], rows.sum(0))
            assert torch.allclose(mean[i], rows.mean(0))
            assert torch.allclose(mx[i], rows.max(0).values)
        else:
            assert torch.all(add[i] == 0) and torch.all(mean[i] == 0) and torch.all(mx[i] == 0)
    deg = sc.scatter_add(torch.ones(5), torch.tensor([0, 0, 1, 4, 4]), dim=0, dim_size=5)
    assert deg.tolist() == [2, 1, 0, 0, 2], "1-D path used by upstream compute_norm"
    assert sc.scatter_add(src, idx, 0, None, n, 0).shape == (n, 3), "legacy 6-positional call shape"
    assert sc.scatter_add(src, idx, dim=0).shape[0] == int(idx.max()) + 1, "dim_size inferred when omitted"
    assert sc.scatter_add(torch.ones(5), torch.tensor([0, 0, 1, 4, 4])).tolist() == [2, 1, 0, 0, 2], "1-D default dim"
    for bad_dim in (1, -1):  # torch_scatter's default dim=-1 means the last axis; the shim only scatters along axis 0
        with pytest.raises(NotImplementedError):
            sc.scatter_add(src, idx, dim=bad_dim)
    with pytest.raises(ValueError):
        sc.scatter_add(src, idx[:3], dim=0, dim_size=n)


def test_wandb_stub_is_inert():
    wb = _load_shim("wandb")
    assert wb.init(project="x") is None and wb.log({"a": 1}) is None and wb.finish() is None
    assert wb.config == {} and "hypergraph-eggroll-shim" in wb.__version__


# ---- runtime -----------------------------------------------------------------------------------


def test_trim_eval_keeps_the_same_prefix_of_both_halves():
    arr = np.arange(20).reshape(10, 2)  # 5 direct rows then their 5 reciprocal twins
    trimmed = rt.trim_eval(arr, 2)
    assert trimmed.tolist() == arr[[0, 1, 5, 6]].tolist()
    assert rt.trim_eval(arr, None) is arr and rt.trim_eval(arr, 0) is arr
    assert rt.trim_eval(arr, 99).shape == arr.shape
    with pytest.raises(ValueError):
        rt.trim_eval(np.zeros((3, 2)), 1)


def test_long_run_lock_refuses_a_second_holder_and_releases_only_its_own_token(tmp_path):
    lock = tmp_path / "LONG_RUN.lock"
    token = rt.acquire_long_run_lock({"arm": "hyper"}, path=lock)
    assert json.loads(lock.read_text(encoding="utf-8"))["arm"] == "hyper"
    with pytest.raises(rt.LockHeld):
        rt.acquire_long_run_lock({"arm": "reified"}, path=lock)
    assert rt.release_long_run_lock("not-mine", path=lock) is False and lock.exists()
    stolen = rt.acquire_long_run_lock({"arm": "reified"}, path=lock, steal=True)
    assert rt.release_long_run_lock(token, path=lock) is False, "the old token no longer owns the lock"
    assert rt.release_long_run_lock(stolen, path=lock) is True and not lock.exists()
    assert rt.release_long_run_lock(stolen, path=lock) is False


def test_checkpoint_round_trip_leaves_no_temp_file(tmp_path):
    path = tmp_path / "state.pt"
    rt.atomic_torch_save({"epochs_done": 7, "w": torch.arange(3.0), "rng": np.random.get_state()}, path)
    assert path.exists() and not path.with_name("state.pt.tmp").exists()
    back = rt.load_checkpoint(path)
    assert back["epochs_done"] == 7 and torch.equal(back["w"], torch.arange(3.0))
    assert back["rng"][0] == "MT19937"


def test_restore_accepts_rng_states_from_any_device():
    """The first GPU resume failed with 'RNG state must be a torch.ByteTensor' because the
    checkpoint had been mapped onto cuda. restore() must move RNG states back to cpu."""
    from src.ontology.stare_bridge import runner

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
    state = {"model": model, "optimizer": optimizer, "scheduler": scheduler}
    torch.manual_seed(123)
    expected = torch.get_rng_state().clone()
    payload = runner.checkpoint_payload(state, epochs_done=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload["rng"]["torch"] = payload["rng"]["torch"].to(device)
    if payload["rng"]["cuda"] is not None:
        payload["rng"]["cuda"] = [s.to(device) for s in payload["rng"]["cuda"]]
    torch.manual_seed(999)
    assert runner.restore(state, payload) == 4
    assert torch.equal(torch.get_rng_state(), expected)


def test_epoch_budget_rule_from_charter_5b():
    assert rt.epoch_budget(20.0) == 135
    assert rt.epoch_budget(5.0) == 401
    assert rt.epoch_budget(120.0) == 50
    with pytest.raises(ValueError):
        rt.epoch_budget(0.0)


def test_jsonl_append_then_read(tmp_path):
    path = tmp_path / "m.jsonl"
    assert rt.read_jsonl(path) == []
    rt.append_jsonl(path, {"epoch": 1, "x": np.float64(0.5)})
    rt.append_jsonl(path, {"epoch": 2})
    assert [r["epoch"] for r in rt.read_jsonl(path)] == [1, 2]


def test_read_jsonl_skips_zero_filled_and_half_written_lines(tmp_path):
    """The 2026-10-08 reset left metrics.jsonl with a NUL-filled tail and a cut line."""
    path = tmp_path / "m.jsonl"
    path.write_bytes(b'{"epoch": 1}\n{"epoch": 2}\n{"epoch": 3, "train_lo' + b"\x00" * 64 + b"\n")
    assert [r["epoch"] for r in rt.read_jsonl(path)] == [1, 2]


def test_prune_jsonl_keeps_each_epoch_up_to_the_checkpoint_once(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_bytes(
        b'{"epoch": 1}\n{"epoch": 2}\n{"epoch": 3}\n{"epoch": 4}\n{"epoch": 3}\n{"epoch": 5, "x"' + b"\x00" * 8 + b"\n"
    )
    assert rt.prune_jsonl(path, max_epoch=3) == (3, 3)
    assert [r["epoch"] for r in rt.read_jsonl(path)] == [1, 2, 3]
    assert not path.with_name("m.jsonl.tmp").exists()
    assert rt.prune_jsonl(tmp_path / "missing.jsonl", max_epoch=3) == (0, 0)


def test_restore_refuses_a_checkpoint_with_non_finite_tensors():
    from src.ontology.stare_bridge import runner

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = {"model": model, "optimizer": optimizer, "scheduler": None}
    payload = runner.checkpoint_payload(state, epochs_done=1)
    payload["model"]["weight"][0, 0] = float("nan")
    assert runner.check_finite(payload["model"]) == ["weight"]
    with pytest.raises(RuntimeError, match="non-finite"):
        runner.restore(state, payload)


def test_fresh_start_refuses_an_out_dir_that_already_has_metrics(tmp_path):
    from src.ontology.stare_bridge import runner

    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.main(["--arm", "hyper", "--epochs", "1", "--out", str(out), "--device", "cpu", "--no-lock"])


# ---- lock file / upstream helpers --------------------------------------------------------------


def test_lock_file_pins_a_commit_and_mirrors_the_repo_data():
    lock = ups.read_lock()
    assert len(lock["commit"]) == 40 and lock["root"] == "third_party/StarE" and lock["license"] == "MIT"
    assert set(lock["repo_mirror_sha256"]) == {f"data/wd50k_100/{s}.txt" for s in ("train", "valid", "test")}
    assert ups.file_mismatches(ups.REPO_ROOT, lock["repo_mirror_sha256"]) == [], "repo data must stay byte-identical"
    assert set(lock["files_sha256"].values()) == set(lock["repo_mirror_sha256"].values())


def test_file_mismatches_reports_missing_and_changed_files(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"abc")
    good = ups.sha256_of(tmp_path / "a.txt")
    assert ups.file_mismatches(tmp_path, {"a.txt": good}) == []
    assert ups.file_mismatches(tmp_path, {"a.txt": "0" * 64, "b.txt": good}) == ["sha256 mismatch: a.txt", "missing: b.txt"]


# ---- integration: needs the pinned checkout and torch-geometric --------------------------------

HAVE_PYG = importlib.util.find_spec("torch_geometric") is not None
HAVE_CLONE = (ups.REPO_ROOT / "third_party" / "StarE").exists()
needs_upstream = pytest.mark.skipif(not (HAVE_PYG and HAVE_CLONE), reason="pinned StarE checkout or torch-geometric missing")


@pytest.fixture(scope="module")
def upstream():
    return ups.load_upstream()


@needs_upstream
def test_upstream_is_on_the_pinned_commit_and_uses_the_shims(upstream):
    assert upstream.commit == ups.read_lock()["commit"]
    assert upstream.shims == {"torch_scatter": "shim", "wandb": "shim"}
    config = upstream.default_config()
    assert config["MODEL_NAME"] == "stare_transformer" and config["STATEMENT_LEN"] == -1 and config["MAX_QPAIRS"] == 15


@needs_upstream
def test_real_data_reification_matches_the_charter_counts(upstream):
    config = upstream.default_config()
    config["DATASET"] = "wd50k_100"
    data = upstream.load_data(config)
    assert (len(data["train"]), len(data["valid"]), len(data["test"])) == (22738, 3279, 5297)
    assert (data["n_entities"], data["n_relations"]) == (18792, 279)
    g = rs.reify_statements(data["train"], num_entities=18792, num_relations=279)
    assert (g.num_blank, g.triples.shape[0]) == (22738, 77915)  # charter §2
    assert rs.expected_role_edges(data["train"] + data["valid"]) == 89208


@needs_upstream
def test_both_arms_build_and_the_reified_arm_scores_only_real_entities(upstream):
    """CPU, a few seconds: parameter counts, score shape, one optimisation step."""
    from src.ontology.stare_bridge.model import (
        build_hyper_model,
        build_reified_model,
        count_parameters,
        is_reified,
    )

    config = upstream.default_config()
    config["DATASET"] = "wd50k_100"
    config["DEVICE"] = torch.device("cpu")
    data = upstream.load_data(config)
    train = list(data["train"])
    n_ent, n_rel = data["n_entities"], data["n_relations"]
    config["NUM_ENTITIES"], config["NUM_RELATIONS"] = n_ent, n_rel
    torch.manual_seed(0)
    hyper = build_hyper_model(upstream, upstream.DataManager.get_alternative_graph_repr(train, config), config)
    g = rs.reify_statements(train, num_entities=n_ent, num_relations=n_rel)
    model_cfg = rs.make_model_config(config, g)
    torch.manual_seed(0)
    reified = build_reified_model(upstream, upstream.DataManager.get_graph_repr(g.triples.tolist(), model_cfg), model_cfg, g)
    assert is_reified(reified) and not is_reified(hyper)
    d = config["EMBEDDING_DIM"]
    p_hyper, p_reified = count_parameters(hyper), count_parameters(reified)
    assert p_hyper == 5_068_016
    assert p_reified == p_hyper - 2 * d * d + 2 * 2 * d + d, "no W_q in two layers; +2 relations x 2 directions; +w_blank"
    assert reified.blank_init().shape == (g.num_blank, d)
    sub = torch.tensor([train[0][0], train[1][0]])
    rel = torch.tensor([train[0][1], train[1][1]])
    quals = torch.tensor([train[0][3:], train[1][3:]])
    for model in (hyper, reified):
        model.eval()
        with torch.no_grad():
            scores = model(sub, rel, quals)
        assert scores.shape == (2, n_ent), "candidates are the real entities only"
    reified.train()
    opt = torch.optim.Adam(reified.parameters(), lr=1e-3)
    before = reified.w_blank.detach().clone()
    target = torch.zeros(2, n_ent)
    target[0, train[0][2]] = 1.0
    target[1, train[1][2]] = 1.0
    loss = reified.loss(reified(sub, rel, quals), target)
    loss.backward()
    opt.step()
    assert not torch.equal(before, reified.w_blank.detach())
    with pytest.raises(ValueError):
        build_reified_model(upstream, {"edge_index": np.zeros((2, 0)), "edge_type": np.zeros(0)}, config, g)


@needs_upstream
def test_runner_smoke_and_resume_on_cpu(tmp_path):
    """One tiny epoch, then resume to two: metrics append, checkpoint restores, config records it."""
    from src.ontology.stare_bridge import runner

    out = tmp_path / "hyper_seed0"
    common = ["--arm", "hyper", "--seed", "0", "--out", str(out), "--device", "cpu", "--eval-every", "1",
              "--chunk", "1", "--eval-limit", "4", "--max-steps-per-epoch", "2", "--no-lock", "--step-sleep-ms", "1"]
    assert runner.main(["--epochs", "1", *common]) == 0
    rows = rt.read_jsonl(out / "metrics.jsonl")
    assert [r["epoch"] for r in rows] == [1] and rows[0]["steps"] == 2 and "eval" in rows[0]
    assert rows[0]["sleep_s"] == pytest.approx(0.001), "one sleep between two steps"
    assert json.loads((out / "run_config.json").read_text())["step_sleep_ms"] == 1.0
    assert set(rows[0]["eval"]["metrics"]) >= {"mrr", "hits_at 1", "hits_at 10"}
    assert rows[0]["eval"]["queries"] == 8, "4 direct + 4 reciprocal statements"
    assert (out / "state.pt").exists() and json.loads((out / "run_config.json").read_text())["resumed"] is False
    assert runner.main(["--epochs", "2", "--resume", *common]) == 0
    rows = rt.read_jsonl(out / "metrics.jsonl")
    assert [r["epoch"] for r in rows] == [1, 2]
    run_config = json.loads((out / "run_config.json").read_text())
    assert run_config["resumed"] is True and run_config["resumed_from_epoch"] == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["epochs_done"] == 2 and summary["last_eval"]["epoch"] == 2 and summary["params"] == 5_068_016
    assert summary["eval_split"] == "valid" and summary["graph"]["statements_in_graph"] == 22738
