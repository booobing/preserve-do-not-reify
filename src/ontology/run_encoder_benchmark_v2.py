"""Paper 1 v2 benchmark CLI (charter-hnn-ontology-encoder-v2).

Each factor is a single change from the v1 configuration, so a run names one factor rather than a
pile of flags:

    uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2a_inductive \
        --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto

    v1_baseline  : the v1 configuration, re-run here so v2 numbers have a same-code reference
    v2a_inductive: held-out facts contribute no hyperedge during training
    v2b_head     : the reified arm reads out through the same head as the hypergraph arm
    v2c_text     : a frozen multilingual sentence encoder replaces the hashing tower
    v2_combined  : A + B + C together (reported, not judged)
    v2d_real     : WD50K_100 n-ary link prediction, structure only, no text tower

The judgment rule is the pre-registered one and is applied, never tuned, here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .baselines import JUDGED_ARMS, ModelConfig, build_model, parameter_report
from .contrastive import TrainConfig, WorldView, train_alignment
from .encoder import count_parameters
from .evaluate import completion_metrics, evaluate_split
from .inductive import build_training_views, coverage_report
from .real_data import load_wd50k
from .run_encoder_benchmark import (
    ALLOWED_OUT_ROOTS,
    JUDGMENT_MIN_SEEDS,
    JUDGMENT_WIN_THRESHOLD,
    enable_determinism,
    resolve_device,
)
from .synthetic import Query, World, load_preset

FACTORS: dict[str, dict[str, Any]] = {
    "v1_baseline": {"inductive": False, "head_parity": False, "text_tower": "hashing", "real": False},
    "v2a_inductive": {"inductive": True, "head_parity": False, "text_tower": "hashing", "real": False},
    "v2b_head": {"inductive": False, "head_parity": True, "text_tower": "hashing", "real": False},
    "v2c_text": {"inductive": False, "head_parity": False, "text_tower": "sentence_transformer", "real": False},
    "v2_combined": {"inductive": True, "head_parity": True, "text_tower": "sentence_transformer", "real": False},
    "v2d_real": {"inductive": True, "head_parity": True, "text_tower": "hashing", "real": True},
    # charter-hnn-ontology-encoder-v2e: same conditions as v2b_head, but the control also gets
    # double depth. v1 measured depth without head parity and v2b parity without depth; the pair
    # had never been combined, and "the gap is really effective depth" is the obvious objection.
    "v2e_depth_parity": {"inductive": False, "head_parity": True, "text_tower": "hashing",
                         "real": False, "control_arm": "reified_binary_deep"},
}

# Metric pair each factor is judged on (charter v2 §7). Metrics are group-qualified because
# `mrr` exists in both the retrieval and completion blocks and must never be read from the wrong one.
JUDGED_METRICS: dict[str, tuple[str, str]] = {
    "v2d_real": ("completion.mrr", "completion.acc@10"),
}
DEFAULT_JUDGED_METRICS = ("retrieval.recall@5", "completion.acc@1")


def _synthetic_view(world: World, factor: dict[str, Any]) -> WorldView:
    view = WorldView.from_world(world)
    if factor["inductive"]:
        build_training_views(view, world.splits["train"])
    return view


def _real_view(dataset, factor: dict[str, Any]) -> WorldView:
    """A WorldView over WD50K: same containers, no text pairing."""
    graph = dataset.hypergraph
    vocab = dataset.schema.vocab()
    reified_graph = graph.reify()
    native = graph.to_incidence(vocab)
    reified = reified_graph.to_incidence(vocab)
    fact_ids = graph.edge_ids()
    view = WorldView(
        schema=dataset.schema,
        vocab=vocab,
        graph=graph,
        fact_ids=fact_ids,
        native=native,
        reified_graph=reified_graph,
        reified=reified,
        candidates_by_type={t: [n.id for n in graph.nodes_of_type(t)] for t in dataset.schema.entity_types},
        node_texts=[graph.nodes[n].text or graph.nodes[n].name for n in native.node_ids],
        reified_node_texts=[reified_graph.nodes[n].text or reified_graph.nodes[n].name for n in reified.node_ids],
        fact_index={f: i for i, f in enumerate(fact_ids)},
    )
    if factor["inductive"]:
        build_training_views(view, dataset.splits["train"])
    return view


def _real_queries(dataset, split: str, *, limit: int | None, seed: int) -> list[Query]:
    """One masked member per held-out statement, chosen deterministically per seed."""
    generator = torch.Generator().manual_seed(seed)
    graph = dataset.hypergraph
    ids = dataset.splits[split]
    if limit is not None and len(ids) > limit:
        picks = torch.randperm(len(ids), generator=generator)[:limit].tolist()
        ids = [ids[i] for i in sorted(picks)]
    queries: list[Query] = []
    for fact_id in ids:
        edge = graph.edges[fact_id]
        roles = edge.roles()
        role = roles[int(torch.randint(len(roles), (1,), generator=generator))]
        queries.append(
            Query(
                id=f"q-{fact_id}",
                edge_id=fact_id,
                kind="completion",
                text="",
                lang="none",
                masked_role=role,
                answer_node_id=edge.member(role),
            )
        )
    return queries


def run_one(
    *,
    factor_name: str,
    arm: str,
    seed: int,
    view: WorldView,
    world: World | None,
    real_queries: dict[str, list[Query]] | None,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    mask_chunk: int | None,
) -> dict[str, Any]:
    factor = FACTORS[factor_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = build_model(arm, view, model_cfg, seed=seed)

    if factor["real"]:
        history = train_structural(model, view, train_cfg, device=device, seed=seed)
        test = {
            "completion": completion_metrics(
                model, view, real_queries["test"], ks=(1, 3, 10), max_masks_per_forward=mask_chunk, with_mrr=True
            )
        }
        val = {
            "completion": completion_metrics(
                model, view, real_queries["val"], ks=(1, 3, 10), max_masks_per_forward=mask_chunk, with_mrr=True
            )
        }
        test["retrieval"] = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "n": 0.0}
        val["retrieval"] = dict(test["retrieval"])
    else:
        history = train_alignment(model, view, world, TrainConfig(**{**train_cfg.__dict__, "seed": seed}), device=device)
        val = evaluate_split(model, view, world, "val")
        test = evaluate_split(model, view, world, "test")

    summary = {
        "factor": factor_name,
        "factor_config": factor,
        "arm": arm,
        "seed": seed,
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
        "inductive": coverage_report(view),
        "mask_chunk": mask_chunk,
        "wall_time_s": time.perf_counter() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
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


def train_structural(model, view: WorldView, config: TrainConfig, *, device, seed: int) -> list[dict[str, Any]]:
    """Completion-only training for data without text (charter v2 §V2-D)."""
    import numpy as np

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    view.to(device)
    train_facts = view.train_fact_ids or view.fact_ids
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        batch_size = min(config.batch_size, len(train_facts))
        picks = rng.choice(len(train_facts), size=batch_size, replace=False)
        pairs = []
        for index in picks:
            fact_id = train_facts[int(index)]
            roles = view.graph.edges[fact_id].roles()
            pairs.append((fact_id, roles[int(rng.integers(0, len(roles)))]))
        loss = model.completion_loss(view, pairs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "loss_align": 0.0,
                    "loss_completion": float(loss.item()),
                    "temperature": float(1.0 / model.logit_scale.exp().item()),
                    "elapsed_s": time.perf_counter() - started,
                }
            )
    model.eval()
    return history


def judge(rows: Sequence[dict[str, Any]], factor_name: str) -> dict[str, Any]:
    """Charter v2 §7: the proposal must win both of the factor's metrics in >=4 of 5 seeds."""
    metric_a, metric_b = JUDGED_METRICS.get(factor_name, DEFAULT_JUDGED_METRICS)
    proposal, control = JUDGED_ARMS
    # A factor may name a stronger control (the depth steelman). Everything else about the rule
    # stays identical, so the strengthened comparison cannot quietly become an easier one.
    control = FACTORS.get(factor_name, {}).get("control_arm", control)

    def value(row: dict[str, Any], metric: str) -> float:
        group, _, key = metric.partition(".")
        return float(row[group][key])

    by_seed = {(r["arm"], int(r["seed"])): r for r in rows}
    seeds = sorted({int(r["seed"]) for r in rows})
    per_seed, wins_a, wins_b = [], 0, 0
    for seed in seeds:
        a, b = by_seed.get((proposal, seed)), by_seed.get((control, seed))
        if a is None or b is None:
            continue
        pair_a = (value(a, metric_a), value(b, metric_a))
        pair_b = (value(a, metric_b), value(b, metric_b))
        win_a, win_b = pair_a[0] > pair_a[1], pair_b[0] > pair_b[1]
        wins_a += int(win_a)
        wins_b += int(win_b)
        per_seed.append({"seed": seed, metric_a: pair_a, metric_b: pair_b, f"win_{metric_a}": win_a, f"win_{metric_b}": win_b})
    n = len(per_seed)
    if n < JUDGMENT_MIN_SEEDS:
        verdict = "incomplete"
    elif wins_a >= JUDGMENT_WIN_THRESHOLD and wins_b >= JUDGMENT_WIN_THRESHOLD:
        verdict = "success"
    elif wins_a >= JUDGMENT_WIN_THRESHOLD or wins_b >= JUDGMENT_WIN_THRESHOLD:
        verdict = "partial"
    else:
        verdict = "fail"
    return {
        "rule": f"charter-hnn-ontology-encoder-v2 §7 ({factor_name})",
        "factor": factor_name,
        "proposal": proposal,
        "control": control,
        "metrics": [metric_a, metric_b],
        "n_seeds_compared": n,
        "min_seeds": JUDGMENT_MIN_SEEDS,
        "win_threshold": JUDGMENT_WIN_THRESHOLD,
        f"wins_{metric_a}": wins_a,
        f"wins_{metric_b}": wins_b,
        "verdict": verdict,
        "per_seed": per_seed,
    }


def collect_rows(out: Path) -> list[dict[str, Any]]:
    """Every per-seed summary already on disk for this factor.

    Reading the aggregate back from the seed directories means a later run that covers only some
    arms - a reference line added after the judged pair, say - extends the aggregate instead of
    replacing it, and the judgment keeps seeing the arms it is defined over.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(out.glob("*/seed_*/summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(rows, key=lambda r: (r["arm"], r["seed"]))


def write_aggregate(out: Path, rows: Sequence[dict[str, Any]], factor_name: str) -> None:
    del rows  # the on-disk seed summaries are the source of truth
    rows = collect_rows(out)
    (out / "summary.json").write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["factor", "arm", "seed", "params", "steps", "first_loss", "final_loss",
              "recall@5", "mrr_retrieval", "acc@1", "acc@3", "acc@10", "mrr_completion",
              "wall_time_s", "peak_vram_mb", "device"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "factor": r["factor"], "arm": r["arm"], "seed": r["seed"], "params": r["params"],
                "steps": r["steps"], "first_loss": r["first_loss"], "final_loss": r["final_loss"],
                "recall@5": r["retrieval"]["recall@5"], "mrr_retrieval": r["retrieval"]["mrr"],
                "acc@1": r["completion"]["acc@1"], "acc@3": r["completion"].get("acc@3"),
                "acc@10": r["completion"].get("acc@10"), "mrr_completion": r["completion"].get("mrr"),
                "wall_time_s": r["wall_time_s"], "peak_vram_mb": r["peak_vram_mb"], "device": r["device"],
            })
    (out / "judgment.json").write_text(json.dumps(judge(rows, factor_name), ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--factor", required=True, choices=sorted(FACTORS))
    parser.add_argument("--arms", nargs="+", default=list(JUDGED_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--text-dim", type=int, default=64)
    parser.add_argument("--text-buckets", type=int, default=4096)
    parser.add_argument("--world-preset", default="v1")
    parser.add_argument("--real-variant", default="wd50k_100")
    parser.add_argument("--real-eval-limit", type=int, default=2000, help="held-out statements scored per split")
    parser.add_argument("--mask-chunk", type=int, default=128, help="facts masked per completion forward")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-any-out", action="store_true")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false", default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = Path(args.out).resolve() / args.factor
    if not args.allow_any_out and not any(root in out.parents or out == root for root in ALLOWED_OUT_ROOTS):
        parser.error(f"--out must be under {[str(r) for r in ALLOWED_OUT_ROOTS]}")
    out.mkdir(parents=True, exist_ok=True)
    deterministic = enable_determinism(bool(args.deterministic))
    device = resolve_device(args.device)
    factor = FACTORS[args.factor]

    model_cfg = ModelConfig(
        d=args.d, layers=args.layers, text_dim=args.text_dim, text_buckets=args.text_buckets,
        head_parity=bool(factor["head_parity"]), text_tower=str(factor["text_tower"]),
    )
    train_cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr)

    world: World | None = None
    real_queries: dict[str, list[Query]] | None = None
    if factor["real"]:
        dataset = load_wd50k(args.real_variant)
        view = _real_view(dataset, factor)
        real_queries = {
            split: _real_queries(dataset, split, limit=args.real_eval_limit, seed=0) for split in ("val", "test")
        }
        data_stats = dataset.stats()
    else:
        world = load_preset(args.world_preset)
        view = _synthetic_view(world, factor)
        data_stats = {"facts": float(world.hypergraph.num_edges), "entities": float(world.hypergraph.num_nodes)}

    (out / "run_config.json").write_text(json.dumps({
        "args": vars(args), "factor": factor, "data": data_stats, "device": str(device),
        "deterministic": deterministic, "inductive": coverage_report(view),
        "params_by_arm": parameter_report(view, model_cfg, arms=tuple(args.arms)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for arm in args.arms:
        for seed in args.seeds:
            print(f"[run] factor={args.factor} arm={arm} seed={seed} device={device}", flush=True)
            summary = run_one(
                factor_name=args.factor, arm=arm, seed=seed, view=view, world=world,
                real_queries=real_queries, model_cfg=model_cfg, train_cfg=train_cfg,
                device=device, out_dir=out / arm / f"seed_{seed}", mask_chunk=args.mask_chunk,
            )
            rows.append(summary)
            comp = summary["completion"]
            print(f"  recall@5={summary['retrieval']['recall@5']:.3f} acc@1={comp['acc@1']:.3f} "
                  f"mrr={comp.get('mrr', float('nan')):.3f} loss {summary['first_loss']:.3f}->"
                  f"{summary['final_loss']:.3f} ({summary['wall_time_s']:.1f}s)", flush=True)
            write_aggregate(out, rows, args.factor)
    verdict = json.loads((out / "judgment.json").read_text(encoding="utf-8"))
    print(f"[judgment] {args.factor}: {verdict['verdict']} {verdict['metrics']} "
          f"{[v for k, v in verdict.items() if k.startswith('wins_')]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
