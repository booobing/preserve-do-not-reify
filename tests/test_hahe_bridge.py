"""Contract tests for src/ontology/hahe_bridge (docs/charter-hahe-same-harness-v1.md §9).

Pure tests always run. Tests that need the pinned checkouts (third_party/HAHE, third_party/StarE)
and torch-geometric are skipped when any is missing.
"""

from __future__ import annotations

import collections
import importlib.util
import json

import numpy as np
import pytest
import torch

from src.ontology.hahe_bridge import data as hd
from src.ontology.stare_bridge import runtime as rt
from src.ontology.stare_bridge import upstream as ups

E2ID = {"__na__": 0, "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5}
R2ID = {"__na__": 0, "P1": 1, "P2": 2, "P3": 3}
ID2E = {i: t for t, i in E2ID.items()}
ID2R = {i: t for t, i in R2ID.items()}
ROWS = [
    [1, 1, 2, 3, 5, 2, 4, 3, 3, 0, 0],  # qualifiers out of order and repeated property P3
    [1, 1, 3, 3, 5, 2, 4, 3, 3, 0, 0],  # same (s, r, quals) as row 0, another object
    [4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # no qualifiers
]


# ---- conversion --------------------------------------------------------------------------------


def test_vocab_puts_relations_then_entities_in_stare_id_order():
    v = hd.build_vocab(E2ID, R2ID)
    assert v.tokens == ("[PAD]", "[MASK]", "P1", "P2", "P3", "Q1", "Q2", "Q3", "Q4", "Q5")
    assert (v.num_relations, v.num_entities, v.entity_offset) == (3, 5, 5)
    assert v.tokens[v.entity_offset + (4 - 1)] == ID2E[4], "HAHE token of StarE entity id 4"


def test_vocab_rejects_gaps_in_ids():
    with pytest.raises(ValueError):
        hd.build_vocab({"__na__": 0, "Q1": 1, "Q3": 3}, R2ID)


def test_row_to_statement_groups_and_sorts_qualifiers_like_hahe_reader():
    st = hd.row_to_statement(ROWS[0], ID2E, ID2R)
    assert (st.arity, st.head, st.relation, st.tail) == (5, "Q1", "P1", "Q2")
    assert st.auxiliary == (("P2", ("Q4",)), ("P3", ("Q3", "Q5")))
    assert hd.row_to_statement(ROWS[2], ID2E, ID2R).auxiliary == ()
    with pytest.raises(ValueError):
        hd.row_to_statement([0, 1, 2, 0, 0], ID2E, ID2R)
    with pytest.raises(ValueError):
        hd.row_to_statement([1, 1, 2, 3, 0], ID2E, ID2R)


def test_candidates_are_entities_seen_as_subject_or_object_never_pad():
    cand = hd.candidate_entities(ROWS, len(E2ID))
    assert cand.tolist() == [False, True, True, True, True, False], "Q5 only appears as a qualifier value"


def test_filter_index_keys_on_raw_qualifiers_with_both_directions():
    idx = hd.filter_index(ROWS)
    quals = tuple(ROWS[0][3:])
    assert idx[("o", 1, 1, quals)] == {2, 3}
    assert idx[("s", 2, 1, quals)] == {1} and idx[("s", 1, 2, (0,) * 8)] == {4}


# ---- ranking -----------------------------------------------------------------------------------


def _stare_formula(pred_row: np.ndarray, target: int, label_row: np.ndarray, ignored: np.ndarray) -> int:
    """loops/evaluation.py compute(): label minus target, ignored entities forced irrelevant, double argsort."""
    pred = torch.tensor(pred_row[None, :].copy())
    irrelevant = torch.tensor(label_row[None, :].copy())
    irrelevant[0, target] = 0
    irrelevant[:, torch.tensor(ignored)] = 1
    pred[irrelevant.bool()] = -1000000
    ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[0, target]
    return int(ranks)


def test_stare_rank_matches_the_upstream_formula_on_random_scores():
    rng = np.random.default_rng(0)
    n = 60
    for _ in range(200):
        scores = rng.normal(size=n)
        candidates = rng.random(n) > 0.3
        candidates[0] = False
        target = int(rng.choice(np.flatnonzero(candidates)))
        known = set(rng.choice(n, size=4, replace=False).tolist()) | {target}
        label = np.zeros(n)
        label[list(known)] = 1.0
        ignored = np.flatnonzero(~candidates)
        assert hd.stare_rank(scores, target, known, candidates) == _stare_formula(scores, target, label, ignored)


def test_stare_rank_refuses_a_target_outside_the_candidates():
    scores = np.array([0.0, 3.0, 2.0, 1.0])
    assert hd.stare_rank(scores, 2, {2}, np.array([False, True, True, False])) == 2
    with pytest.raises(ValueError):
        hd.stare_rank(scores, 3, {3}, np.array([False, True, True, False]))


def test_hahe_native_rank_is_filtered_and_pessimistic_on_ties():
    scores = np.array([9.0, 5.0, 5.0, 7.0, 1.0])
    assert hd.hahe_native_rank(scores, 1, {1}) == 3, "pad ignored, 7 above, tie at 5 counted against"
    assert hd.hahe_native_rank(scores, 1, {1, 3}) == 2, "known answer 3 filtered"


def test_summarise_and_mean_of_sides():
    left, right = hd.summarise([1, 2, 10]), hd.summarise([1, 1, 20])
    assert left["mrr"] == pytest.approx((1 + 0.5 + 0.1) / 3) and left["hits_at 1"] == pytest.approx(1 / 3)
    both = hd.mean_of_sides(left, right)
    assert both["count"] == 3 and both["hits_at 10"] == pytest.approx((1.0 + 2 / 3) / 2)


# ---- integration: needs both checkouts and torch-geometric -------------------------------------

HAVE_PYG = importlib.util.find_spec("torch_geometric") is not None
HAVE_CLONES = all((ups.REPO_ROOT / "third_party" / name).exists() for name in ("HAHE", "StarE"))
needs_upstream = pytest.mark.skipif(not (HAVE_PYG and HAVE_CLONES), reason="pinned HAHE/StarE checkout or torch-geometric missing")


@needs_upstream
def test_hahe_reader_labels_map_back_to_stare_ids(tmp_path):
    from src.ontology.hahe_bridge.upstream import load_hahe

    hahe = load_hahe()
    assert hahe.commit == ups.read_lock(ups.REPO_ROOT / "third_party" / "hahe.lock.json")["commit"]
    v = hd.build_vocab(E2ID, R2ID)
    (tmp_path / "vocab.txt").write_text("\n".join(v.lines()) + "\n", encoding="utf-8")
    hv = hahe.reader.Vocabulary(str(tmp_path / "vocab.txt"), v.num_relations, v.num_entities)
    sts = [hd.row_to_statement(r, ID2E, ID2R) for r in ROWS]
    examples = [hahe.reader.NaryExample(st.arity, st.head, st.relation, st.tail,
                                        collections.OrderedDict((a, list(x)) for a, x in st.auxiliary) or None) for st in sts]
    ds = hahe.reader.MultiDataset(hv, examples, max_arity=5, max_seq_length=9)
    for f in ds.features:
        if f.mask_position in (hd.SUBJECT_POSITION, hd.OBJECT_POSITION):
            row = ROWS[f.example_id]
            stare_id = row[0] if f.mask_position == hd.SUBJECT_POSITION else row[2]
            assert f.mask_label == v.entity_offset + stare_id - 1
    assert len(ds) == 9 + 9 + 3, "every non-pad position of every statement is one query"


@needs_upstream
def test_runner_smoke_crash_and_resume_on_cpu(tmp_path):
    """Two tiny epochs with a simulated crash after the first: metrics append, the checkpoint restores,
    and resuming with another --epochs is refused because OneCycleLR spans the whole run."""
    from src.ontology.hahe_bridge import runner

    out = tmp_path / "hahe_seed0"
    common = ["--seed", "0", "--out", str(out), "--device", "cpu", "--eval-every", "1", "--chunk", "1",
              "--eval-limit", "4", "--max-steps-per-epoch", "2", "--no-lock", "--step-sleep-ms", "1",
              "--local-layers", "1", "--global-layers", "1", "--hidden-dim", "32", "--batch-size", "64"]
    assert runner.main(["--epochs", "2", "--stop-after-epochs", "1", *common]) == 0
    rows = rt.read_jsonl(out / "metrics.jsonl")
    assert [r["epoch"] for r in rows] == [1] and rows[0]["steps"] == 2 and "eval" in rows[0]
    assert rows[0]["sleep_s"] == pytest.approx(0.001)
    ev = rows[0]["eval"]
    assert ev["queries"] == 8 and ev["object_prediction"]["count"] == 4 and ev["subject_prediction"]["count"] == 4
    assert set(ev["metrics"]) >= {"mrr", "hits_at 1", "hits_at 10"} and "mrr" in ev["hahe_native_ht"]
    assert not (out / "summary.json").exists()
    with pytest.raises(ValueError, match="same --epochs"):
        runner.main(["--epochs", "3", "--resume", *common])
    assert runner.main(["--epochs", "2", "--resume", *common]) == 0
    rows = rt.read_jsonl(out / "metrics.jsonl")
    assert [r["epoch"] for r in rows] == [1, 2]
    run_config = json.loads((out / "run_config.json").read_text())
    assert run_config["resumed"] is True and run_config["resumed_from_epoch"] == 1
    assert run_config["graph"]["statements_in_graph"] == 22738 and run_config["candidate_entities"] == 10874
    summary = json.loads((out / "summary.json").read_text())
    assert summary["epochs_done"] == 2 and summary["last_eval"]["epoch"] == 2 and summary["eval_split"] == "valid"
