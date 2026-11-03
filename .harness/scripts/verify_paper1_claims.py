"""Check every Paper 1 headline number against the artifacts it came from.

The encoder v2 report carries hand-copied tables. A report can drift from its own outputs in two
ways that both look fine on the page: a number typed wrong, or a verdict that no longer matches the
judgment file after some later run touched the directory. That second one already happened once in
this program - a stray `flat_text` run overwrote a closed factor's aggregate - so the claims get a
script rather than a re-read.

Exits non-zero on the first mismatch, so it can gate the consolidation checkpoint.

    uv run --frozen python .harness/scripts/verify_paper1_claims.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "hnn_ontology_encoder_v2"

# factor -> (metric key in summary, arm -> expected mean, tolerance)
# Values are the ones docs/hnn-ontology-encoder-v2-results.md prints in its tables.
EXPECTED: dict[str, dict[str, dict[str, float]]] = {
    "v1_baseline": {
        "retrieval.recall@5": {"hyper": 0.863, "reified_binary": 0.670},
        "completion.acc@1": {"hyper": 0.502, "reified_binary": 0.424},
    },
    "v2a_inductive": {
        "retrieval.recall@5": {"hyper": 0.870, "reified_binary": 0.723},
        "completion.acc@1": {"hyper": 0.310, "reified_binary": 0.212},
    },
    "v2b_head": {
        "retrieval.recall@5": {"hyper": 0.863, "reified_binary": 0.699},
        "completion.acc@1": {"hyper": 0.502, "reified_binary": 0.403},
    },
    "v2c_text": {
        "retrieval.recall@5": {"hyper": 0.846, "reified_binary": 0.585, "flat_text": 0.897},
        "completion.acc@1": {"hyper": 0.405, "reified_binary": 0.251, "flat_text": 0.236},
    },
    "v2d_real": {
        "completion.mrr": {"hyper": 0.147, "reified_binary": 0.118},
        "completion.acc@10": {"hyper": 0.289, "reified_binary": 0.215},
    },
    # charter-hnn-ontology-encoder-v2e: the control here is the deep steelman, not reified_binary.
    "v2e_depth_parity": {
        "retrieval.recall@5": {"hyper": 0.863, "reified_binary_deep": 0.672},
        "completion.acc@1": {"hyper": 0.502, "reified_binary_deep": 0.400},
    },
    "v2_combined": {
        "retrieval.recall@5": {"hyper": 0.815, "reified_binary": 0.605},
        "completion.acc@1": {"hyper": 0.234, "reified_binary": 0.178},
    },
}

# The claim the whole paper rests on: hyper beats reified_binary on both pre-registered metrics,
# in every factor, on all five seeds.
JUDGED_FACTORS = tuple(EXPECTED)
TOLERANCE = 0.0015  # the report prints three decimals


def dig(blob: dict, dotted: str):
    node = blob
    for part in dotted.split("."):
        node = node[part]
    return node


def seed_means(factor: str, arm: str, metric: str) -> tuple[float, int]:
    values = []
    for path in sorted((OUT / factor / arm).glob("seed_*/summary.json")):
        values.append(float(dig(json.loads(path.read_text(encoding="utf-8")), metric)))
    if not values:
        raise FileNotFoundError(f"no seed summaries under {OUT / factor / arm}")
    return sum(values) / len(values), len(values)


def main() -> int:
    problems: list[str] = []
    checked = 0

    for factor, metrics in EXPECTED.items():
        judgment_path = OUT / factor / "judgment.json"
        if not judgment_path.exists():
            problems.append(f"{factor}: no judgment.json")
            continue
        judgment = json.loads(judgment_path.read_text(encoding="utf-8"))

        if judgment.get("verdict") != "success":
            problems.append(f"{factor}: verdict is {judgment.get('verdict')!r}, report says success")
        expected_control = "reified_binary_deep" if factor == "v2e_depth_parity" else "reified_binary"
        if judgment.get("proposal") != "hyper" or judgment.get("control") != expected_control:
            problems.append(f"{factor}: judged pair is "
                            f"{judgment.get('proposal')} vs {judgment.get('control')}")
        for metric in judgment.get("metrics", []):
            wins = judgment.get(f"wins_{metric}")
            if wins != 5:
                problems.append(f"{factor}: wins_{metric} = {wins}, report says 5/5")
            checked += 1

        for metric, per_arm in metrics.items():
            for arm, expected in per_arm.items():
                try:
                    mean, n = seed_means(factor, arm, metric)
                except (FileNotFoundError, KeyError) as exc:
                    problems.append(f"{factor}/{arm}/{metric}: {exc}")
                    continue
                if n != 5:
                    problems.append(f"{factor}/{arm}: {n} seeds on disk, report claims 5")
                if abs(mean - expected) > TOLERANCE:
                    problems.append(f"{factor}/{arm}/{metric}: disk {mean:.4f} vs report {expected:.3f}")
                checked += 1

    print(f"checked {checked} claims across {len(EXPECTED)} factors")
    if problems:
        print("\nMISMATCHES:")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("all Paper 1 headline numbers match their artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
