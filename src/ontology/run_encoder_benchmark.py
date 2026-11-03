"""Paper 1 benchmark CLI (contract C9, charter §A).

    uv run python -m src.ontology.run_encoder_benchmark --world-preset smoke --steps 30 \
        --arms hyper reified_binary flat_text --seeds 1 2 --out .harness/tmp/ontology_encoder_smoke

Writes DIR/{arm}/seed_{k}/summary.json, DIR/encoder_benchmark_summary.{json,csv} and DIR/judgment.json.
The judgment rule is the pre-registered §A-4 rule; it is applied, never tuned, here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .baselines import ARM_NAMES, JUDGED_ARMS, ModelConfig, build_model, parameter_report
from .contrastive import TrainConfig, WorldView, train_alignment
from .encoder import count_parameters
from .evaluate import evaluate_split
from .extract import (
    EntityCatalog,
    StructuredRecordExtractor,
    TemplateTextExtractor,
    extraction_fidelity,
)
from .synthetic import WORLD_PRESETS, World, load_preset

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUT_ROOTS = (REPO_ROOT / ".harness" / "tmp", REPO_ROOT / "outputs")

JUDGMENT_MIN_SEEDS = 5
JUDGMENT_WIN_THRESHOLD = 4


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def enable_determinism(enabled: bool) -> bool:
    """Bit-reproducible runs on CUDA: deterministic index_add_/scatter paths + cuBLAS workspace pin.

    Without this, float atomics in `index_add_` drift by ~1e-5 per step and the reified arm's 1500-step
    trajectory diverges between identical launches (observed 2026-08-20). Must run before the first
    cuBLAS call, so call it at the top of main().
    """
    if not enabled:
        torch.use_deterministic_algorithms(False)
        return False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    return True


def extraction_report(world: World) -> dict[str, Any]:
    catalog = EntityCatalog(world.entity_catalog)
    records = StructuredRecordExtractor(world.schema, catalog=catalog).extract(world.records)
    texts = TemplateTextExtractor(world.schema, catalog=catalog).extract([s.text for s in world.sentences])
    return {
        "structured_records": extraction_fidelity(records, world.hypergraph),
        "template_text": extraction_fidelity(texts, world.hypergraph),
    }


def run_one(
    *,
    arm: str,
    seed: int,
    world: World,
    view: WorldView,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    extraction: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = build_model(arm, view, model_cfg, seed=seed)
    history = train_alignment(model, view, world, TrainConfig(**{**train_cfg.__dict__, "seed": seed}), device=device)
    val = evaluate_split(model, view, world, "val")
    test = evaluate_split(model, view, world, "test")
    wall = time.perf_counter() - started
    peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    summary = {
        "arm": arm,
        "seed": seed,
        "world_preset": world.config.to_dict(),
        "params": count_parameters(model),
        "steps": train_cfg.steps,
        "batch_size": train_cfg.batch_size,
        "lr": train_cfg.lr,
        "first_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "retrieval": test["retrieval"],
        "completion": test["completion"],
        "val_retrieval": val["retrieval"],
        "val_completion": val["completion"],
        "extraction": extraction,
        "wall_time_s": wall,
        "peak_vram_mb": peak_vram_mb,
        "device": str(device),
        "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
        "model_config": model_cfg.__dict__,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return summary


def judge(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Charter §A-4: hyper beats reified_binary on recall@5 AND completion acc@1 in >=4 of 5 seeds."""
    by_arm_seed: dict[tuple[str, int], dict[str, Any]] = {(r["arm"], int(r["seed"])): r for r in rows}
    seeds = sorted({int(r["seed"]) for r in rows})
    proposal, control = JUDGED_ARMS
    per_seed = []
    wins_recall = wins_completion = 0
    for seed in seeds:
        a, b = by_arm_seed.get((proposal, seed)), by_arm_seed.get((control, seed))
        if a is None or b is None:
            continue
        r5 = (a["retrieval"]["recall@5"], b["retrieval"]["recall@5"])
        c1 = (a["completion"]["acc@1"], b["completion"]["acc@1"])
        win_r = r5[0] > r5[1]
        win_c = c1[0] > c1[1]
        wins_recall += int(win_r)
        wins_completion += int(win_c)
        per_seed.append({"seed": seed, "recall@5": r5, "completion_acc@1": c1, "win_recall5": win_r, "win_completion": win_c})
    n = len(per_seed)
    if n < JUDGMENT_MIN_SEEDS:
        verdict = "incomplete"
    elif wins_recall >= JUDGMENT_WIN_THRESHOLD and wins_completion >= JUDGMENT_WIN_THRESHOLD:
        verdict = "success"
    elif wins_recall >= JUDGMENT_WIN_THRESHOLD or wins_completion >= JUDGMENT_WIN_THRESHOLD:
        verdict = "partial"
    else:
        verdict = "fail"
    flat_note = None
    flat_rows = [r for r in rows if r["arm"] == "flat_text"]
    hyper_rows = [r for r in rows if r["arm"] == proposal]
    if flat_rows and hyper_rows:
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        flat_note = {
            "hyper_mean_recall@5": mean([r["retrieval"]["recall@5"] for r in hyper_rows]),
            "flat_mean_recall@5": mean([r["retrieval"]["recall@5"] for r in flat_rows]),
            "hyper_mean_completion@1": mean([r["completion"]["acc@1"] for r in hyper_rows]),
            "flat_mean_completion@1": mean([r["completion"]["acc@1"] for r in flat_rows]),
        }
    return {
        "rule": "charter-hnn-ontology-program-v1 §A-4",
        "proposal": proposal,
        "control": control,
        "n_seeds_compared": n,
        "min_seeds": JUDGMENT_MIN_SEEDS,
        "win_threshold": JUDGMENT_WIN_THRESHOLD,
        "wins_recall5": wins_recall,
        "wins_completion": wins_completion,
        "verdict": verdict,
        "per_seed": per_seed,
        "flat_text_reference": flat_note,
    }


def write_aggregate(out: Path, rows: Sequence[dict[str, Any]]) -> None:
    (out / "encoder_benchmark_summary.json").write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "arm", "seed", "params", "steps", "first_loss", "final_loss",
        "recall@1", "recall@5", "recall@10", "mrr", "acc@1", "acc@3",
        "val_recall@5", "val_acc@1", "wall_time_s", "peak_vram_mb", "device",
    ]
    with (out / "encoder_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "arm": r["arm"], "seed": r["seed"], "params": r["params"], "steps": r["steps"],
                    "first_loss": r["first_loss"], "final_loss": r["final_loss"],
                    "recall@1": r["retrieval"]["recall@1"], "recall@5": r["retrieval"]["recall@5"],
                    "recall@10": r["retrieval"]["recall@10"], "mrr": r["retrieval"]["mrr"],
                    "acc@1": r["completion"]["acc@1"], "acc@3": r["completion"]["acc@3"],
                    "val_recall@5": r["val_retrieval"]["recall@5"], "val_acc@1": r["val_completion"]["acc@1"],
                    "wall_time_s": r["wall_time_s"], "peak_vram_mb": r["peak_vram_mb"], "device": r["device"],
                }
            )
    (out / "judgment.json").write_text(json.dumps(judge(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-preset", default="smoke", choices=sorted(WORLD_PRESETS))
    parser.add_argument("--arms", nargs="+", default=list(JUDGED_ARMS) + ["flat_text"], choices=list(ARM_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--text-dim", type=int, default=64)
    parser.add_argument("--text-buckets", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True, help="bit-reproducible CUDA kernels (default)")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-any-out", action="store_true", help="bypass the .harness/tmp|outputs output-root guard")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    if not args.allow_any_out and not any(out == root or root in out.parents for root in ALLOWED_OUT_ROOTS):
        parser.error(f"--out must be under {[str(r) for r in ALLOWED_OUT_ROOTS]} (AGENTS.override.md); pass --allow-any-out to bypass")
    out.mkdir(parents=True, exist_ok=True)
    deterministic = enable_determinism(bool(args.deterministic))
    device = resolve_device(args.device)
    world = load_preset(args.world_preset)
    view = WorldView.from_world(world)
    model_cfg = ModelConfig(d=args.d, layers=args.layers, text_dim=args.text_dim, text_buckets=args.text_buckets)
    train_cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr)
    extraction = extraction_report(world)
    params = parameter_report(view, model_cfg, arms=tuple(args.arms))
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "world": world.config.to_dict(),
                "facts": world.hypergraph.num_edges,
                "nodes": world.hypergraph.num_nodes,
                "splits": {k: len(v) for k, v in world.splits.items()},
                "params_by_arm": params,
                "extraction": extraction,
                "device": str(device),
                "deterministic": deterministic,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for arm in args.arms:
        for seed in args.seeds:
            print(f"[run] arm={arm} seed={seed} device={device}", flush=True)
            summary = run_one(
                arm=arm,
                seed=seed,
                world=world,
                view=view,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=device,
                out_dir=out / arm / f"seed_{seed}",
                extraction=extraction,
            )
            rows.append(summary)
            print(
                f"  recall@5={summary['retrieval']['recall@5']:.3f} mrr={summary['retrieval']['mrr']:.3f} "
                f"acc@1={summary['completion']['acc@1']:.3f} loss {summary['first_loss']:.3f}->{summary['final_loss']:.3f} "
                f"({summary['wall_time_s']:.1f}s)",
                flush=True,
            )
            write_aggregate(out, rows)
    verdict = json.loads((out / "judgment.json").read_text(encoding="utf-8"))
    print(f"[judgment] {verdict['verdict']} wins_recall5={verdict['wins_recall5']} wins_completion={verdict['wins_completion']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
