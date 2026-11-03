"""Significance tests for Paper 1's judged tables (docs/charter-hnn-ontology-paper1-finishing.md §3).

The pre-registered verdict rule is ">= 4/5 seeds ahead on both metrics". That rule stands and this
script does not touch it. What it adds is the thing a reviewer will ask for on top: per-pair paired
t-tests with a 95% CI, Holm-Bonferroni over all 14 factor x metric pairs, an exact sign-flip
permutation test, and Cohen's d_z. Methods were frozen in the charter before this script ran.

Values come straight from each factor's judgment.json `per_seed`, i.e. exactly what was judged.

    uv run --frozen python .harness/scripts/paper1_statistics.py
"""

from __future__ import annotations

import itertools
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "hnn_ontology_encoder_v2"
OUT = ROOT / "outputs" / "hnn_ontology_paper1_stats"
FACTORS = ["v1_baseline", "v2a_inductive", "v2b_head", "v2c_text", "v2d_real", "v2_combined",
           "v2e_depth_parity"]
ALPHA = 0.05


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip permutation p on the mean paired difference."""
    observed = abs(st.fmean(diffs))
    flips = list(itertools.product((1, -1), repeat=len(diffs)))
    extreme = sum(abs(st.fmean([s * d for s, d in zip(signs, diffs)])) >= observed - 1e-12
                  for signs in flips)
    return extreme / len(flips)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1), in input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired(diffs: list[float]) -> dict:
    n = len(diffs)
    mean = st.fmean(diffs)
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    tcrit = float(sps.t.ppf(0.975, n - 1))
    t = mean / se if se > 0 else math.inf
    p = float(2 * sps.t.sf(abs(t), n - 1)) if se > 0 else 0.0
    return {"n": n, "mean_diff": mean, "sd_diff": sd, "ci95": [mean - tcrit * se, mean + tcrit * se],
            "t": t, "df": n - 1, "p_t": p, "p_signflip": sign_flip_p(diffs),
            "cohen_dz": mean / sd if sd > 0 else math.inf,
            "wins": sum(d > 0 for d in diffs)}


def main() -> int:
    rows = []
    for factor in FACTORS:
        judgment = json.loads((SRC / factor / "judgment.json").read_text(encoding="utf-8"))
        for metric in judgment["metrics"]:
            per_seed = sorted(judgment["per_seed"], key=lambda r: r["seed"])
            diffs = [r[metric][0] - r[metric][1] for r in per_seed]
            rows.append({"factor": factor, "metric": metric, "proposal": judgment["proposal"],
                         "control": judgment["control"], "seeds": [r["seed"] for r in per_seed],
                         "diffs": diffs, **paired(diffs)})
    for row, adj in zip(rows, holm([r["p_t"] for r in rows])):
        row["p_t_holm"] = adj
        row["significant_holm"] = adj < ALPHA
    n_sig = sum(r["significant_holm"] for r in rows)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"charter": "docs/charter-hnn-ontology-paper1-finishing.md §3", "alpha": ALPHA,
               "n_comparisons": len(rows), "n_significant_holm": n_sig,
               "signflip_two_sided_floor": 2 / 2 ** 5, "rows": rows}
    (OUT / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["| factor | metric | control | mean diff [95% CI] | t(4) | p | p (Holm) | sign-flip p | d_z | wins |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = r["ci95"]
        lines.append(f"| `{r['factor']}` | {r['metric']} | `{r['control']}` | "
                     f"{r['mean_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {r['t']:.2f} | {r['p_t']:.4f} | "
                     f"{r['p_t_holm']:.4f}{'' if r['significant_holm'] else ' ✗'} | "
                     f"{r['p_signflip']:.4f} | {r['cohen_dz']:.2f} | {r['wins']}/5 |")
    table = "\n".join(lines)
    (OUT / "stats.md").write_text(table + f"\n\nHolm-significant at alpha={ALPHA}: {n_sig}/{len(rows)}\n",
                                  encoding="utf-8")
    print(table)
    print(f"\nHolm-significant at alpha={ALPHA}: {n_sig}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
