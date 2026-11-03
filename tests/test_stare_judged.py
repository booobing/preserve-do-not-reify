"""Contract tests for .harness/scripts/judge_stare_input_representation.py (charter §7-a).

Pure: toy summary.json files in a temp tree, no torch, no upstream.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("judge_stare", REPO / ".harness/scripts/judge_stare_input_representation.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["judge_stare"] = module
    spec.loader.exec_module(module)
    return module


def _write(root: Path, arm: str, seed: int, *, epochs_done: int, last_epoch: int, mrr: float, h10: float, best_mrr: float | None = None):
    d = root / arm / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    metrics = {"mrr": mrr, "hits_at 1": mrr * 0.7, "hits_at 3": mrr * 1.1, "hits_at 10": h10, "mr": 100.0 / mrr}
    payload = {
        "epochs_done": epochs_done,
        "last_eval": {"epoch": last_epoch, "metrics": metrics},
        "best_eval": {"epoch": last_epoch, "metrics": {"mrr": best_mrr if best_mrr is not None else mrr}},
    }
    (d / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _tree(root: Path, hyper: list[tuple[float, float]], reified: list[tuple[float, float]], epochs: int = 401):
    for seed, (m, h) in enumerate(hyper, start=1):
        _write(root, "hyper", seed, epochs_done=epochs, last_epoch=epochs, mrr=m, h10=h)
    for seed, (m, h) in enumerate(reified, start=1):
        _write(root, "reified", seed, epochs_done=epochs, last_epoch=epochs, mrr=m, h10=h)


def test_seed_wins_counts_strict_wins_and_ties():
    j = _mod()
    assert j.seed_wins([0.5, 0.4, 0.3, 0.2, 0.1], [0.4, 0.4, 0.2, 0.3, 0.0]) == (3, 1, 1)
    with pytest.raises(ValueError):
        j.seed_wins([0.1, 0.2], [0.1])


def test_verdict_needs_both_metrics_at_four_of_five_and_ties_do_not_count():
    j = _mod()
    assert j.verdict({"mrr": (4, 1, 0), "hits_at 10": (5, 0, 0)}) == "reified_wins"
    assert j.verdict({"mrr": (0, 5, 0), "hits_at 10": (1, 4, 0)}) == "hyper_wins"
    assert j.verdict({"mrr": (4, 1, 0), "hits_at 10": (3, 1, 1)}) == "no_consistent_order"
    assert j.verdict({"mrr": (3, 2, 0), "hits_at 10": (3, 2, 0)}) == "no_consistent_order"


def test_paired_stats_sign_flip_floor_for_five_seeds_and_ci_contains_mean():
    j = _mod()
    s = j.paired_stats([0.10, 0.08, 0.12, 0.09, 0.11])
    assert s["n"] == 5 and s["p_signflip"] == pytest.approx(2 / 32)
    assert s["ci95_low"] < s["mean"] < s["ci95_high"] and s["ci95_low"] > 0
    assert s["p_t"] is not None and s["p_t"] < 0.01
    mixed = j.paired_stats([0.02, -0.01, 0.0, 0.01, -0.02])
    assert mixed["p_signflip"] == 1.0 and mixed["ci95_low"] < 0 < mixed["ci95_high"]


def test_reproduction_check_flags_a_port_far_from_the_paper():
    j = _mod()
    assert j.reproduction_check([0.66, 0.64, 0.65, 0.65, 0.66])["within_tolerance"] is True
    out = j.reproduction_check([0.50, 0.52, 0.49, 0.51, 0.50])
    assert out["within_tolerance"] is False and out["delta"] == pytest.approx(0.504 - 0.654)


def test_build_report_reified_wins_and_writes_outputs(tmp_path):
    j = _mod()
    _tree(tmp_path, hyper=[(0.60, 0.75), (0.61, 0.76), (0.62, 0.77), (0.63, 0.78), (0.64, 0.79)],
          reified=[(0.66, 0.80), (0.67, 0.81), (0.60, 0.70), (0.69, 0.83), (0.70, 0.84)])
    report = j.build_report(tmp_path, epochs=401)
    assert report["verdict"] == "reified_wins"
    assert report["wins_reified_hyper_ties"]["mrr"] == (4, 1, 0)
    assert report["complete_pairs"] == 5 and report["missing"] == []
    j.write_outputs(report, tmp_path)
    assert json.loads((tmp_path / "judgment.json").read_text(encoding="utf-8"))["verdict"] == "reified_wins"
    assert (tmp_path / "summary.csv").read_text(encoding="utf-8").count("\n") == 6


def test_build_report_hyper_wins_and_no_order(tmp_path):
    j = _mod()
    _tree(tmp_path, hyper=[(0.66, 0.80)] * 5, reified=[(0.60, 0.75)] * 5)
    assert j.build_report(tmp_path, epochs=401)["verdict"] == "hyper_wins"
    other = tmp_path / "other"
    _tree(other, hyper=[(0.66, 0.80), (0.60, 0.75), (0.66, 0.80), (0.60, 0.75), (0.66, 0.80)],
          reified=[(0.60, 0.75), (0.66, 0.80), (0.60, 0.75), (0.66, 0.80), (0.60, 0.75)])
    assert j.build_report(other, epochs=401)["verdict"] == "no_consistent_order"


def test_incomplete_runs_block_the_verdict_but_allow_a_partial_table(tmp_path):
    j = _mod()
    _tree(tmp_path, hyper=[(0.60, 0.75)] * 5, reified=[(0.66, 0.80)] * 4)
    _write(tmp_path, "reified", 5, epochs_done=200, last_epoch=200, mrr=0.5, h10=0.6)  # still running
    with pytest.raises(SystemExit):
        j.build_report(tmp_path, epochs=401)
    report = j.build_report(tmp_path, epochs=401, partial=True)
    assert report["verdict"] is None and report["complete_pairs"] == 4
    assert report["missing"] == ["reified/seed_5 (epochs_done=200)"]


def test_final_metrics_requires_the_last_eval_at_the_target_epoch():
    j = _mod()
    ok = {"epochs_done": 401, "last_eval": {"epoch": 401, "metrics": {"mrr": 0.5}}}
    assert j.final_metrics(ok, epochs=401) == {"mrr": 0.5}
    assert j.final_metrics({"epochs_done": 401, "last_eval": {"epoch": 400, "metrics": {"mrr": 0.5}}}, epochs=401) is None
    assert j.final_metrics({"epochs_done": 76, "last_eval": {"epoch": 76, "metrics": {"mrr": 0.5}}}, epochs=401) is None


def test_other_arm_from_its_own_tree_is_judged_at_its_own_final_epoch(tmp_path):
    """charter-hahe-same-harness-v1 §4: native_hyper at 401 epochs in one tree, hahe at 300 in another."""
    j = _mod()
    base_root, other_root = tmp_path / "native", tmp_path / "hahe"
    for seed in range(1, 6):
        _write(base_root, "native_hyper", seed, epochs_done=401, last_epoch=401, mrr=0.66, h10=0.78)
        _write(other_root, "hahe", seed, epochs_done=300, last_epoch=300, mrr=0.60 + 0.01 * seed, h10=0.70)
    report = j.build_report(base_root, epochs=401, base="native_hyper", other="hahe", other_root=other_root, other_epochs=300)
    assert report["verdict"] == "native_hyper_wins" and report["other_epochs"] == 300
    j.write_outputs(report, base_root)
    assert (other_root / "judgment_hahe_vs_native_hyper.json").exists() and not (base_root / "judgment_hahe_vs_native_hyper.json").exists()
    at_401 = j.build_report(base_root, epochs=401, base="native_hyper", other="hahe", other_root=other_root)
    assert at_401["verdict"] is None and len(at_401["missing"]) == 5, "hahe has no evaluation at epoch 401"
    assert "other_root" not in j.build_report(base_root, epochs=401, base="native_hyper", other="native_hyper", partial=True)
