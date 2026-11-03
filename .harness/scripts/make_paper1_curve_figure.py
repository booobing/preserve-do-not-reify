"""Figure 4 of the Paper 1 manuscript: test-MRR learning curves of the two standard-protocol studies.

Reads only run artifacts:
  outputs/stare_input_representation_v1/judged/{hyper,reified}/seed_{1..5}/metrics.jsonl        (Study 2)
  outputs/stare_native_encoder_v1/judged/native_{hyper,reified,prune}/seed_{1..5}/metrics.jsonl (Study 3)

Top row: mean test MRR over 5 seeds (band = min..max over seeds); the legend names every line with
its final value. Bottom row: paired difference (control - preserve) per seed, mean line and min..max
band, with a zero line; the converged gaps (0.007-0.013) are invisible at full scale, so an inset
zooms into epochs 200-401. Colours follow paper1_fig1/fig2 (preserve blue, reified orange) plus
green for the qualifier-free control; line styles differ too, so identity never depends on colour.

    uv run --frozen python .harness/scripts/make_paper1_curve_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "hnn_ontology_figures"
SEEDS = (1, 2, 3, 4, 5)
BLUE, ORANGE, GREEN, INK, MUTED, GRID = "#0072B2", "#E69F00", "#009E73", "#0b0b0b", "#52514e", "#e6e5e1"
NEWLINE = chr(10)
MINUS = chr(0x2212)

STUDIES = [
    {
        "title": "(a) StarE encoder (unmodified code)",
        "root": ROOT / "outputs" / "stare_input_representation_v1" / "judged",
        "base": ("hyper", "StarE native graph", BLUE, "-"),
        "others": [("reified", "reified graph", ORANGE, "--")],
        "pilot_epoch": 76,
    },
    {
        "title": "(b) Role-aware hypergraph encoder",
        "root": ROOT / "outputs" / "stare_native_encoder_v1" / "judged",
        "base": ("native_hyper", "hyperedge (preserve)", BLUE, "-"),
        "others": [("native_reified", "reified graph", ORANGE, "--"), ("native_prune", "qualifiers removed", GREEN, ":")],
        "pilot_epoch": None,
    },
]


def curves(root: Path, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """epochs (E,), mrr (seeds, E) from the evaluation rows of metrics.jsonl."""
    per_seed = []
    epochs = None
    for seed in SEEDS:
        text = (root / arm / f"seed_{seed}" / "metrics.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        evals = [(r["epoch"], r["eval"]["metrics"]["mrr"]) for r in rows if "eval" in r]
        e = np.array([x[0] for x in evals])
        if epochs is None:
            epochs = e
        if not np.array_equal(e, epochs):
            raise ValueError(f"{arm} seed {seed}: evaluation epochs differ")
        per_seed.append([x[1] for x in evals])
    return epochs, np.array(per_seed)


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]})
    for col, study in enumerate(STUDIES):
        top, bottom = axes[0, col], axes[1, col]
        style(top)
        style(bottom)
        base_arm, base_label, base_color, base_ls = study["base"]
        epochs, base = curves(study["root"], base_arm)
        series = [(base_arm, base_label, base_color, base_ls, base)]
        for arm, label, color, ls in study["others"]:
            e, values = curves(study["root"], arm)
            if not np.array_equal(e, epochs):
                raise ValueError(f"{arm}: evaluation epochs differ from the preserve arm")
            series.append((arm, label, color, ls, values))
        for _, label, color, ls, values in series:
            top.fill_between(epochs, values.min(0), values.max(0), color=color, alpha=0.15, linewidth=0)
            top.plot(epochs, values.mean(0), color=color, linestyle=ls, linewidth=2.0,
                     label=f"{label} ({values.mean(0)[-1]:.4f})")
        top.legend(loc="lower right", fontsize=7.3, frameon=False, title="final test MRR", title_fontsize=7.3)
        top.set_title(study["title"], fontsize=9.5, color=INK, loc="left")
        top.set_ylim(0, 0.72)
        if col == 0:
            top.set_ylabel("test MRR (mean of 5 seeds)", color=INK, fontsize=8.5)

        inset = bottom.inset_axes([0.42, 0.50, 0.55, 0.42])
        style(inset)
        lows = []
        for _, _, color, ls, values in series[1:]:
            diff = values - base
            lows.append(diff.min())
            for ax in (bottom, inset):
                ax.fill_between(epochs, diff.min(0), diff.max(0), color=color, alpha=0.18, linewidth=0)
                ax.plot(epochs, diff.mean(0), color=color, linestyle=ls, linewidth=1.8)
        for ax in (bottom, inset):
            ax.axhline(0, color=MUTED, linewidth=1.0)
        inset.set_xlim(200, 401)
        inset.set_ylim(-0.025, 0.005)
        inset.tick_params(labelsize=6.5)
        inset.set_title("zoom: epochs 200-401", fontsize=6.8, color=MUTED, pad=2)
        bottom.set_ylim(min(lows) - 0.01, 0.13)
        bottom.set_xlabel("epoch", color=INK, fontsize=8.5)
        if col == 0:
            bottom.set_ylabel(f"control {MINUS} preserve (MRR)", color=INK, fontsize=8.5)
        if study["pilot_epoch"]:
            for ax in (top, bottom):
                ax.axvline(study["pilot_epoch"], color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
            top.text(study["pilot_epoch"] + 5, 0.24, f"pilot budget{NEWLINE}(76 epochs)", fontsize=7, color=MUTED)
        bottom.text(0.02, 0.03, "below 0: preserve ahead", transform=bottom.transAxes, fontsize=7, color=MUTED)
    fig.subplots_adjust(left=0.115, right=0.98, top=0.94, bottom=0.09, wspace=0.22, hspace=0.12)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"paper1_fig4_standard_protocol_curves.{ext}", dpi=220)
    print("wrote", OUT / "paper1_fig4_standard_protocol_curves.pdf")


if __name__ == "__main__":
    main()
