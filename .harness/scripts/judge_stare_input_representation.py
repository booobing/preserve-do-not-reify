"""Judge the StarE input-representation run (charter docs/charter-stare-input-representation-v1.md §7-a).

Two-sided, pre-registered: on the FINAL epoch's test evaluation, an arm "wins a seed" on a metric
when its value is strictly higher. Verdict `reified_wins` if reified wins >= 4 of 5 seeds on BOTH
MRR and Hits@10; `hyper_wins` symmetrically; otherwise `no_consistent_order`. Ties are not wins.
Best-on-test values are reported as an optimistic reference only and never judged.

    uv run --frozen python .harness/scripts/judge_stare_input_representation.py            # verdict (all 10 runs)
    uv run --frozen python .harness/scripts/judge_stare_input_representation.py --partial  # interim table, no verdict
    # any other pair of arms (charter-stare-native-encoder-v1 §5):
    uv run --frozen python .harness/scripts/judge_stare_input_representation.py \n        --root outputs/stare_native_encoder_v1/judged --base native_hyper --other native_reified
    # the other arm in its own tree with its own budget (charter-hahe-same-harness-v1 §4):
    uv run --frozen python .harness/scripts/judge_stare_input_representation.py --root outputs/stare_native_encoder_v1/judged         --base native_hyper --other hahe --other-root outputs/hahe_same_harness_v1/judged --other-epochs 300
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "outputs" / "stare_input_representation_v1" / "judged"
ARMS = ("hyper", "reified")
SEEDS = (1, 2, 3, 4, 5)
JUDGED_METRICS = ("mrr", "hits_at 10")
REPORTED_METRICS = ("mrr", "hits_at 1", "hits_at 3", "hits_at 10", "mr")
WIN_THRESHOLD = 4
STARE_PAPER_TEST_MRR = 0.654  # Galkin et al. 2020, WD50K_100, StarE(H)+Transformer(H)
REPRODUCTION_TOLERANCE = 0.03


def load_summary(root: Path, arm: str, seed: int) -> dict[str, Any] | None:
    path = root / arm / f"seed_{seed}" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def final_metrics(summary: dict[str, Any], *, epochs: int) -> dict[str, float] | None:
    """Metrics of the final epoch's evaluation, or None when the run has not reached `epochs`."""
    if int(summary.get("epochs_done", 0)) < epochs or not summary.get("last_eval"):
        return None
    if int(summary["last_eval"]["epoch"]) != epochs:
        return None
    return {k: float(v) for k, v in summary["last_eval"]["metrics"].items()}


def seed_wins(values_a: Sequence[float], values_b: Sequence[float]) -> tuple[int, int, int]:
    """(a wins, b wins, ties) over paired seeds; strictly greater wins, equal is a tie."""
    a_wins = sum(1 for a, b in zip(values_a, values_b, strict=True) if a > b)
    b_wins = sum(1 for a, b in zip(values_a, values_b, strict=True) if b > a)
    return a_wins, b_wins, len(values_a) - a_wins - b_wins


def verdict(wins: dict[str, tuple[int, int, int]], *, threshold: int = WIN_THRESHOLD, base: str = "hyper", other: str = "reified") -> str:
    """wins[metric] = (other wins, base wins, ties) for each judged metric."""
    if all(wins[m][0] >= threshold for m in JUDGED_METRICS):
        return f"{other}_wins"
    if all(wins[m][1] >= threshold for m in JUDGED_METRICS):
        return f"{base}_wins"
    return "no_consistent_order"


def paired_stats(diffs: Sequence[float]) -> dict[str, float | None]:
    """Mean, sd, two-sided one-sample t (df n-1), 95% CI and the exact sign-flip p of the seed differences."""
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "t": None, "p_t": None, "ci95_low": None, "ci95_high": None, "p_signflip": None}
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    try:
        from scipy import stats

        p_t = float(stats.ttest_1samp(list(diffs), 0.0).pvalue)
        crit = float(stats.t.ppf(0.975, n - 1))
    except ImportError:  # pragma: no cover - scipy is in the lock
        p_t, crit = None, 2.776 if n == 5 else float("nan")
    observed = abs(mean)
    count = 0
    for signs in itertools.product((1, -1), repeat=n):
        flipped = sum(s * d for s, d in zip(signs, diffs, strict=True)) / n
        if abs(flipped) >= observed - 1e-12:
            count += 1
    return {
        "n": n, "mean": mean, "sd": sd, "t": t, "p_t": p_t,
        "ci95_low": mean - crit * se, "ci95_high": mean + crit * se,
        "p_signflip": count / 2**n,
    }


def reproduction_check(hyper_final_mrr: Sequence[float], *, reference: float = STARE_PAPER_TEST_MRR, tolerance: float = REPRODUCTION_TOLERANCE) -> dict[str, Any]:
    """Sanity check of the port, not a judgment: the hyper arm is upstream StarE at its default budget."""
    mean = sum(hyper_final_mrr) / len(hyper_final_mrr)
    return {"reference": reference, "mean_hyper_mrr": mean, "delta": mean - reference, "within_tolerance": abs(mean - reference) <= tolerance, "tolerance": tolerance}


def build_report(root: Path, *, epochs: int, seeds: Sequence[int] = SEEDS, partial: bool = False,
                 base: str = "hyper", other: str = "reified", other_root: Path | None = None,
                 other_epochs: int | None = None) -> dict[str, Any]:
    """`other_root` / `other_epochs` let the compared arm live in its own tree with its own budget;
    each arm is then judged at its own final epoch. Both default to the base arm's."""
    arms = (base, other)
    roots = {base: root, other: other_root if other_root is not None else root}
    arm_epochs = {base: epochs, other: other_epochs if other_epochs is not None else epochs}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in seeds:
        row: dict[str, Any] = {"seed": seed}
        for arm in arms:
            summary = load_summary(roots[arm], arm, seed)
            metrics = final_metrics(summary, epochs=arm_epochs[arm]) if summary else None
            if metrics is None:
                missing.append(f"{arm}/seed_{seed}" + (f" (epochs_done={summary.get('epochs_done')})" if summary else " (no summary)"))
                row[arm] = None
            else:
                row[arm] = {m: metrics[m] for m in REPORTED_METRICS}
                row[f"{arm}_best_test_mrr"] = float(summary["best_eval"]["metrics"]["mrr"]) if summary.get("best_eval") else None
        rows.append(row)
    complete = [r for r in rows if r[base] and r[other]]
    report: dict[str, Any] = {"root": str(root), "arms": {"base": base, "other": other}, "epochs": epochs, "seeds": list(seeds), "rows": rows, "missing": missing, "complete_pairs": len(complete)}
    if roots[other] != root or arm_epochs[other] != epochs:
        report["other_root"], report["other_epochs"] = str(roots[other]), arm_epochs[other]
    if not complete:
        report["verdict"] = None
        return report
    wins: dict[str, tuple[int, int, int]] = {}
    stats: dict[str, dict[str, float | None]] = {}
    for m in REPORTED_METRICS:
        other_vals = [r[other][m] for r in complete]
        base_vals = [r[base][m] for r in complete]
        wins[m] = seed_wins(other_vals, base_vals)
        stats[m] = paired_stats([a - b for a, b in zip(other_vals, base_vals, strict=True)])
    report[f"wins_{other}_{base}_ties"] = wins
    report[f"paired_{other}_minus_{base}"] = stats
    if base == "hyper":  # the upstream-StarE arm; the port check only applies to it
        report["reproduction"] = reproduction_check([r["hyper"]["mrr"] for r in complete])
    if missing and not partial:
        raise SystemExit(f"incomplete: {missing}; use --partial for an interim table")
    report["verdict"] = verdict(wins, base=base, other=other) if not missing else None
    return report


def output_names(report: dict[str, Any]) -> tuple[str, str]:
    base, other = report["arms"]["base"], report["arms"]["other"]
    if (base, other) == ("hyper", "reified"):
        return "judgment.json", "summary.csv"
    return f"judgment_{other}_vs_{base}.json", f"summary_{other}_vs_{base}.csv"


def write_outputs(report: dict[str, Any], root: Path) -> None:
    """Written next to the compared arm's runs (its own tree when `other_root` is set)."""
    root = Path(report.get("other_root", root))
    arms = (report["arms"]["base"], report["arms"]["other"])
    judgment_name, csv_name = output_names(report)
    (root / judgment_name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (root / csv_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed"] + [f"{arm}_{m}" for arm in arms for m in REPORTED_METRICS])
        for row in report["rows"]:
            writer.writerow([row["seed"]] + [(row[arm][m] if row[arm] else "") for arm in arms for m in REPORTED_METRICS])


def print_report(report: dict[str, Any]) -> None:
    print(f"judged run: {report['root']} epochs={report['epochs']} complete pairs={report['complete_pairs']}")
    if report.get("other_root"):
        print(f"compared arm from {report['other_root']} at epochs={report['other_epochs']}")
    base, other = report["arms"]["base"], report["arms"]["other"]
    print(f"seed | {base} mrr / h1 / h10 | {other} mrr / h1 / h10 | diff mrr ({other} - {base})")
    for row in report["rows"]:
        h, r = row[base], row[other]
        hs = f"{h['mrr']:.4f} / {h['hits_at 1']:.4f} / {h['hits_at 10']:.4f}" if h else "-"
        rs = f"{r['mrr']:.4f} / {r['hits_at 1']:.4f} / {r['hits_at 10']:.4f}" if r else "-"
        d = f"{r['mrr'] - h['mrr']:+.4f}" if h and r else "-"
        print(f"{row['seed']:>4} | {hs} | {rs} | {d}")
    if report.get(f"wins_{other}_{base}_ties"):
        for m in JUDGED_METRICS:
            w = report[f"wins_{other}_{base}_ties"][m]
            s = report[f"paired_{other}_minus_{base}"][m]
            ci = f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]" if s["ci95_low"] is not None else "-"
            print(f"{m}: {other} wins {w[0]}, {base} wins {w[1]}, ties {w[2]} | mean diff {s['mean']:+.4f} CI95 {ci} p_t {s['p_t']} p_signflip {s['p_signflip']}")
    if report.get("reproduction"):
        rep = report["reproduction"]
        print(f"reproduction: hyper mean test MRR {rep['mean_hyper_mrr']:.4f} vs paper {rep['reference']} (delta {rep['delta']:+.4f}, within {rep['tolerance']}: {rep['within_tolerance']})")
    if report.get("missing"):
        print("missing:", ", ".join(report["missing"]))
    print("VERDICT:", report.get("verdict"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--epochs", type=int, default=401)
    parser.add_argument("--partial", action="store_true", help="interim table without a verdict")
    parser.add_argument("--base", default="hyper", help="reference arm (second element of every win tuple)")
    parser.add_argument("--other", default="reified", help="compared arm (first element of every win tuple)")
    parser.add_argument("--other-root", default=None, help="tree of the compared arm (default: --root)")
    parser.add_argument("--other-epochs", type=int, default=None, help="final epoch of the compared arm (default: --epochs)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    report = build_report(root, epochs=args.epochs, partial=args.partial, base=args.base, other=args.other,
                          other_root=Path(args.other_root) if args.other_root else None, other_epochs=args.other_epochs)
    print_report(report)
    if report.get("verdict") is not None:
        write_outputs(report, root)
        out = Path(report.get("other_root", root))
        names = output_names(report)
        print(f"wrote {out / names[0]} and {out / names[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
