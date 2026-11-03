"""Train and evaluate one arm on WD50K_100 with the upstream StarE pipeline (charter §5).

This reproduces the orchestration of upstream ``run.py`` (data -> graph -> reciprocals -> sampler
-> model -> Adam + StepLR -> epoch loop -> filtered evaluation) and adds what the charter needs
and upstream lacks: a seed argument, chunked checkpoints with ``--resume``, append-only
``metrics.jsonl``, the long-run lock, and a ``summary.json``.

The per-epoch body is a line-for-line port of ``loops/loops.py::training_loop_gcn`` (sampler
reset, BCE over all entities, grad clipping at 1.0, scheduler step at epoch end). Evaluation
runs after every ``--eval-every`` completed epochs and after the last one, on the split that
decision 0-2 selects (valid by default; test with ``--use-test``).

    uv run --frozen python -m src.ontology.stare_bridge.runner --arm hyper --seed 0 --epochs 3 \
        --out outputs/stare_input_representation_v1/hyper/seed_0 --device cuda
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .model import build_hyper_model, build_reified_model, count_parameters
from .native_encoder import NATIVE_ARMS, build_native_incidence, build_native_model
from .reify_stare import graph_summary, make_data_config, make_model_config, reify_statements
from .runtime import (
    LockHeld,
    acquire_long_run_lock,
    append_jsonl,
    atomic_torch_save,
    load_checkpoint,
    prune_jsonl,
    read_jsonl,
    release_long_run_lock,
    trim_eval,
    write_json,
)
from .upstream import load_upstream

ARMS: tuple[str, ...] = ("hyper", "reified", *NATIVE_ARMS)
DATASET = "wd50k_100"
CORRUPTION_POSITIONS = [0, 2]  # upstream default: subject and object only


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, required=True, help="total epochs for this run (resume continues to it)")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=10, help="epochs between checkpoints")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-test", action="store_true", help="StarE default split: graph train+valid, evaluate test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None, help="override upstream BATCH_SIZE (default 128)")
    parser.add_argument("--lr", type=float, default=None, help="override upstream LEARNING_RATE (default 1e-4)")
    parser.add_argument("--eval-limit", type=int, default=0, help="smoke/tests only: statements per direction")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="smoke/tests only")
    parser.add_argument(
        "--step-sleep-ms", type=float, default=0.0,
        help="idle the GPU for this long after every optimiser step (duty cycle for a machine that hard-resets "
             "under sustained load; the computation and results are unchanged, only wall time grows)",
    )
    parser.add_argument("--no-lock", action="store_true", help="tests only: skip the long-run lock")
    parser.add_argument("--steal-lock", action="store_true", help="replace a stale LONG_RUN.lock (after a reset)")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    return device


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Everything up to (not including) training: data, graph, model, sampler, bench, optimiser."""
    up = load_upstream()
    seed_everything(args.seed)
    config = up.default_config()
    config["DATASET"] = DATASET
    config["USE_TEST"] = bool(args.use_test)
    config["EPOCHS"] = int(args.epochs)
    config["EVAL_EVERY"] = int(args.eval_every)
    config["SEED"] = int(args.seed)
    if args.batch_size is not None:
        config["BATCH_SIZE"] = int(args.batch_size)
    if args.lr is not None:
        config["LEARNING_RATE"] = float(args.lr)
    config["WANDB"] = False
    config["SAVE"] = False

    data = up.load_data(config)
    train, valid, test = list(data["train"]), list(data["valid"]), list(data["test"])
    n_entities, n_relations = int(data["n_entities"]), int(data["n_relations"])
    config["NUM_ENTITIES"], config["NUM_RELATIONS"] = n_entities, n_relations
    device = select_device(args.device)
    config["DEVICE"] = device

    excluded = up.DataManager.gather_missing_entities(
        data=train + valid + test, positions=CORRUPTION_POSITIONS, n_ents=n_entities
    )
    graph_statements = train + valid if config["USE_TEST"] else train

    if args.arm == "hyper":
        data_config = model_config = config
        graph_repr = up.DataManager.get_alternative_graph_repr(graph_statements, config)
        model = build_hyper_model(up, graph_repr, model_config)
        graph_info = {
            "statements_in_graph": len(graph_statements),
            "edges_with_inverses": int(graph_repr["edge_index"].shape[1]),
            "qualifier_pairs_with_inverses": int(graph_repr["quals"].shape[1]),
        }
    elif args.arm in NATIVE_ARMS:
        # charter-stare-native-encoder-v1: chapter-5 encoder, upstream decoder and relation table
        data_config = model_config = config
        inc = build_native_incidence(graph_statements, num_entities=n_entities, num_relations=n_relations,
                                     mode=NATIVE_ARMS[args.arm])
        graph_repr = up.DataManager.get_alternative_graph_repr(graph_statements, config)
        model = build_native_model(up, graph_repr, model_config, inc)
        graph_info = {"statements_in_graph": len(graph_statements), **inc.summary()}
    else:
        reified = reify_statements(graph_statements, num_entities=n_entities, num_relations=n_relations)
        data_config = make_data_config(config, reified)
        model_config = make_model_config(config, reified)
        graph_repr = up.DataManager.get_graph_repr(reified.triples.tolist(), model_config)
        model = build_reified_model(up, graph_repr, model_config, reified)
        graph_info = {"statements_in_graph": len(graph_statements), **graph_summary(reified),
                      "edges_with_inverses": int(graph_repr["edge_index"].shape[1])}
    model.to(device)

    # Reciprocal statements, exactly as run.py (extend in place).
    train.extend(up.DataManager.add_reciprocals(train, data_config))
    valid.extend(up.DataManager.add_reciprocals(valid, data_config))
    test.extend(up.DataManager.add_reciprocals(test, data_config))
    combine = up.combine
    if config["USE_TEST"]:
        ev = {"index": combine(train, valid), "eval": combine(test)}
        tr = {"train": combine(train, valid)}
        eval_split = "test"
    else:
        ev = {"index": combine(train, test), "eval": combine(valid)}
        tr = {"train": combine(train)}
        eval_split = "valid"
    ev["eval"] = trim_eval(ev["eval"], args.eval_limit or None)

    sampler = up.MultiClassSampler(
        data=tr["train"], n_entities=n_entities, lbl_smooth=config["LABEL_SMOOTHING"],
        bs=config["BATCH_SIZE"], with_q=config["SAMPLER_W_QUALIFIERS"],
    )
    acc, mrr, mr, hits_at = up.extra["metrics"]
    bench = up.EvaluationBench(
        ev, model, bs=config["BATCH_SIZE"],
        metrics=[acc, mrr, mr, partial(hits_at, k=3), partial(hits_at, k=5), partial(hits_at, k=10)],
        filtered=True, n_ents=n_entities, excluding_entities=excluded, positions=CORRUPTION_POSITIONS, config=data_config,
    )
    if config["OPTIMIZER"] != "adam":
        raise NotImplementedError("upstream default optimiser is adam; others are out of scope")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95) if config["LR_SCHEDULER"] else None

    return {
        "up": up, "config": config, "data_config": data_config, "model_config": model_config, "device": device,
        "model": model, "sampler": sampler, "bench": bench, "optimizer": optimizer, "scheduler": scheduler,
        "eval_split": eval_split, "eval_queries": int(ev["eval"].shape[0]), "sampler_keys": len(sampler.keys),
        "n_entities": n_entities, "n_relations": n_relations, "excluded_entities": int(len(excluded)),
        "graph_info": graph_info, "params": count_parameters(model),
    }


def train_one_epoch(state: dict[str, Any], *, max_steps: int = 0, step_sleep_s: float = 0.0) -> tuple[float, int]:
    """One pass over the sampler: upstream training_loop_gcn's inner loop, verbatim semantics.

    `step_sleep_s` > 0 synchronises and idles the GPU after each step; it changes nothing about
    the optimisation, only the duty cycle (this machine resets under sustained GPU load).
    """
    model, sampler, optimizer, device = state["model"], state["sampler"], state["optimizer"], state["device"]
    config = state["config"]
    model.train()
    losses: list[float] = []
    for step, (triples, labels) in enumerate(sampler.reset()):
        if max_steps and step >= max_steps:
            break
        if step_sleep_s > 0 and step > 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            time.sleep(step_sleep_s)
        optimizer.zero_grad()
        sub = torch.tensor(triples[:, 0], dtype=torch.long, device=device)
        rel = torch.tensor(triples[:, 1], dtype=torch.long, device=device)
        quals = torch.tensor(triples[:, 2:], dtype=torch.long, device=device)
        target = torch.tensor(labels, dtype=torch.float, device=device)
        pred = model(sub, rel, quals)
        loss = model.loss(pred, target)
        loss.backward()
        if config["GRAD_CLIPPING"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    if state["scheduler"] is not None:
        state["scheduler"].step()
    return float(np.mean(losses)) if losses else float("nan"), len(losses)


def evaluate(state: dict[str, Any]) -> dict[str, Any]:
    summary = state["bench"].run()
    keep = {k: float(v) for k, v in summary["metrics"].items()}
    return {
        "metrics": keep,
        "object_prediction": {k: float(v) for k, v in summary["left"].items()},
        "subject_prediction": {k: float(v) for k, v in summary["right"].items()},
        "queries": int(summary["data_length"]),
        "time_s": float(summary["time_taken"]),
    }


def checkpoint_payload(state: dict[str, Any], epochs_done: int) -> dict[str, Any]:
    return {
        "epochs_done": epochs_done,
        "model": state["model"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "scheduler": state["scheduler"].state_dict() if state["scheduler"] is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def check_finite(state_dict: dict[str, Any]) -> list[str]:
    """Names of floating tensors holding NaN/inf (a checkpoint damaged by a reset must not be resumed)."""
    return [
        name for name, tensor in state_dict.items()
        if torch.is_tensor(tensor) and tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
    ]


def restore(state: dict[str, Any], payload: dict[str, Any]) -> int:
    damaged = check_finite(payload["model"])
    if damaged:
        raise RuntimeError(f"checkpoint has non-finite tensors, refusing to resume: {damaged[:5]}")
    state["model"].load_state_dict(payload["model"])
    state["optimizer"].load_state_dict(payload["optimizer"])
    if state["scheduler"] is not None and payload.get("scheduler") is not None:
        state["scheduler"].load_state_dict(payload["scheduler"])
    rng = payload.get("rng") or {}
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    # RNG states must be CPU ByteTensors; a checkpoint mapped onto cuda would otherwise fail here
    # (the first GPU smoke resume did exactly that).
    if "torch" in rng:
        torch.set_rng_state(rng["torch"].cpu())
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    return int(payload["epochs_done"])


def jsonable_config(config: dict[str, Any]) -> dict[str, Any]:
    out = {k: (str(v) if isinstance(v, torch.device) else v) for k, v in config.items() if k != "STAREARGS"}
    out["STAREARGS"] = dict(config["STAREARGS"])
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "state.pt"
    metrics_path = out / "metrics.jsonl"
    started = time.time()
    if metrics_path.exists() and not args.resume:
        raise FileExistsError(f"{metrics_path} exists; pass --resume to continue it or choose another --out")

    state = prepare(args)
    device = state["device"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run_config = {
        "arm": args.arm, "seed": args.seed, "epochs_target": args.epochs, "eval_every": args.eval_every,
        "chunk": args.chunk, "eval_split": state["eval_split"], "eval_queries": state["eval_queries"],
        "eval_limit": args.eval_limit, "max_steps_per_epoch": args.max_steps_per_epoch, "step_sleep_ms": args.step_sleep_ms,
        "device": str(device), "params": state["params"], "n_entities": state["n_entities"],
        "n_relations_upstream": state["n_relations"], "candidate_entities": state["n_entities"] - state["excluded_entities"],
        "excluded_entities": state["excluded_entities"], "sampler_keys": state["sampler_keys"],
        "graph": state["graph_info"], "upstream_commit": state["up"].commit, "shims": state["up"].shims,
        "torch": torch.__version__, "data_config": jsonable_config(state["data_config"]),
        "model_config": jsonable_config(state["model_config"]), "resumed": False,
    }

    epochs_done = 0
    if args.resume and checkpoint_path.exists():
        # Always map onto cpu: load_state_dict copies into the live (possibly cuda) parameters and
        # the optimiser casts its state to them, while RNG states have to stay on cpu.
        epochs_done = restore(state, load_checkpoint(checkpoint_path, map_location="cpu"))
        kept, dropped = prune_jsonl(metrics_path, max_epoch=epochs_done)
        run_config["resumed"] = True
        run_config["resumed_from_epoch"] = epochs_done
        run_config["metrics_rows_dropped_on_resume"] = dropped
        print(f"[resume] continuing from {epochs_done} completed epochs; metrics.jsonl kept {kept} rows, dropped {dropped}", flush=True)
    elif args.resume:
        # A reset before the first checkpoint leaves metrics rows with no state to continue from.
        kept, dropped = prune_jsonl(metrics_path, max_epoch=0)
        print(f"[resume] no checkpoint found; starting from scratch (dropped {dropped} orphan metrics rows)", flush=True)
    write_json(out / "run_config.json", run_config)
    print(f"[start] arm={args.arm} seed={args.seed} params={state['params']} device={device} "
          f"graph={state['graph_info']} eval={state['eval_split']}:{state['eval_queries']}", flush=True)

    for epoch in range(epochs_done, args.epochs):
        t0 = time.time()
        loss, steps = train_one_epoch(state, max_steps=args.max_steps_per_epoch, step_sleep_s=args.step_sleep_ms / 1000.0)
        train_time = time.time() - t0
        completed = epoch + 1
        row: dict[str, Any] = {
            "epoch": completed, "train_loss": loss, "steps": steps, "epoch_time_s": train_time,
            "sleep_s": max(steps - 1, 0) * args.step_sleep_ms / 1000.0,
            "lr": float(state["optimizer"].param_groups[0]["lr"]),
        }
        if completed % args.eval_every == 0 or completed == args.epochs:
            row["eval"] = evaluate(state)
        if device.type == "cuda":
            row["peak_vram_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024**2)
        append_jsonl(metrics_path, row)
        line = f"[epoch {completed}/{args.epochs}] loss {loss:.5f} steps {steps} {train_time:.1f}s"
        if "eval" in row:
            m = row["eval"]["metrics"]
            line += f" | {state['eval_split']} mrr {m['mrr']:.4f} h1 {m['hits_at 1']:.4f} h10 {m['hits_at 10']:.4f}"
        print(line, flush=True)
        if completed % args.chunk == 0 or completed == args.epochs:
            atomic_torch_save(checkpoint_payload(state, completed), checkpoint_path)

    rows = read_jsonl(metrics_path)
    evals = [(r["epoch"], r["eval"]) for r in rows if "eval" in r]
    best = max(evals, key=lambda item: item[1]["metrics"]["mrr"]) if evals else None
    summary = {
        **{k: run_config[k] for k in ("arm", "seed", "eval_split", "eval_queries", "params", "device", "upstream_commit", "graph", "shims", "candidate_entities", "excluded_entities")},
        "epochs_done": rows[-1]["epoch"] if rows else epochs_done,
        "epochs_target": args.epochs,
        "last_eval": {"epoch": evals[-1][0], **evals[-1][1]} if evals else None,
        "best_eval": {"epoch": best[0], **best[1]} if best else None,
        "train_time_s": float(sum(r["epoch_time_s"] for r in rows)),
        "eval_time_s": float(sum(r["eval"]["time_s"] for r in rows if "eval" in r)),
        "mean_epoch_time_s": float(np.mean([r["epoch_time_s"] for r in rows])) if rows else None,
        "peak_vram_mb": max((r.get("peak_vram_mb", 0.0) for r in rows), default=None),
        "wall_this_process_s": time.time() - started,
        "resumed": run_config["resumed"],
    }
    write_json(out / "summary.json", summary)
    print(f"[done] {summary['last_eval']['metrics'] if summary['last_eval'] else 'no eval'}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = None
    if not args.no_lock:
        try:
            token = acquire_long_run_lock(
                {"task": "stare-input-representation-v1", "arm": args.arm, "seed": args.seed, "epochs": args.epochs, "out": args.out},
                steal=args.steal_lock,
            )
        except LockHeld as exc:
            print(f"[lock] refusing to start: {exc}", file=sys.stderr, flush=True)
            return 3
    try:
        run(args)
    finally:
        if token is not None:
            release_long_run_lock(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
