"""Train HAHE on the StarE harness rows and evaluate it with the StarE protocol (charter §3-§4).

Training reproduces upstream ``src/run.py`` (``main``): HAHE's MultiDataset over every maskable
position, global hypergraph from the first (1 - hyperedge_dropout) share of training statements,
AdamW + OneCycleLR (linear, warm-up share), soft-label cross-entropy for entities and relations.
Hyperparameters default to HAHE's published WD50K command (README). What this runner adds: the
StarE harness data (train+valid graph, six-pair cap), seeds, chunked checkpoints with ``--resume``,
the GPU duty-cycle sleep, the long-run lock, and an evaluation identical to the StarE bench
(see ``data.py``) next to HAHE's own rule.

    uv run --frozen python -m src.ontology.hahe_bridge.runner --arm hahe --seed 1 --epochs 300 \
        --out outputs/hahe_same_harness_v1/judged/hahe/seed_1 --device cuda --use-test --resume
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..stare_bridge.runtime import (
    LockHeld,
    acquire_long_run_lock,
    append_jsonl,
    atomic_torch_save,
    load_checkpoint,
    prune_jsonl,
    read_jsonl,
    release_long_run_lock,
    write_json,
)
from ..stare_bridge.upstream import load_upstream
from . import data as hd
from .upstream import load_hahe

DATASET = "wd50k_100"
# HAHE README, WD50K command (the closest published configuration; WD50K(100) has none).
HAHE_WD50K = {
    "hidden_dim": 256, "global_layers": 2, "global_dropout": 0.1, "global_activation": "elu", "global_heads": 4,
    "local_layers": 12, "local_dropout": 0.1, "local_heads": 4, "decoder_activation": "gelu",
    "batch_size": 512, "lr": 5e-4, "weight_decay": 0.01, "entity_soft": 0.2, "relation_soft": 0.1,
    "hyperedge_dropout": 0.8, "warmup_proportion": 0.1,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("hahe",), default="hahe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--chunk", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-test", action="store_true", help="graph train+valid, evaluate test (StarE default)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--eval-limit", type=int, default=0, help="smoke/tests only: test statements evaluated")
    p.add_argument("--max-steps-per-epoch", type=int, default=0, help="smoke/tests only")
    p.add_argument("--stop-after-epochs", type=int, default=0, help="tests only: leave the process after N epochs (simulated crash)")
    p.add_argument("--step-sleep-ms", type=float, default=0.0)
    p.add_argument("--no-lock", action="store_true")
    p.add_argument("--steal-lock", action="store_true")
    for key, value in HAHE_WD50K.items():
        p.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    return p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    stare = load_upstream()
    hahe = load_hahe()
    seed_everything(args.seed)
    config = stare.default_config()
    config["DATASET"] = DATASET
    config["USE_TEST"] = bool(args.use_test)
    raw = stare.load_data(config)
    train, valid, test = list(raw["train"]), list(raw["valid"]), list(raw["test"])
    e2id, r2id = raw["e2id"], raw["r2id"]
    n_entities = int(raw["n_entities"])
    id2e = [None] * len(e2id)
    for t, i in e2id.items():
        id2e[i] = t
    id2r = [None] * len(r2id)
    for t, i in r2id.items():
        id2r[i] = t
    graph_rows = train + valid if args.use_test else train
    eval_rows = test if args.use_test else valid
    if args.eval_limit:
        eval_rows = eval_rows[: args.eval_limit]

    vocab = hd.build_vocab(e2id, r2id)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab_file = out / "vocab.txt"
    vocab_file.write_text("\n".join(vocab.lines()) + "\n", encoding="utf-8")
    hv = hahe.reader.Vocabulary(vocab_file=str(vocab_file), num_relations=vocab.num_relations, num_entities=vocab.num_entities)

    def to_examples(rows):
        sts = [hd.row_to_statement(r, id2e, id2r) for r in rows]
        return sts, [hahe.reader.NaryExample(arity=st.arity, head=st.head, relation=st.relation, tail=st.tail,
                                             auxiliary_info=collections.OrderedDict((a, list(v)) for a, v in st.auxiliary) or None)
                     for st in sts]

    train_sts, train_examples = to_examples(graph_rows)
    eval_sts, eval_examples = to_examples(eval_rows)
    max_arity = max(hd.max_arity(train_sts), hd.max_arity(eval_sts))
    max_seq_len = 2 * (max_arity - 2) + 3
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset = hahe.reader.MultiDataset(hv, train_examples, max_arity, max_seq_len)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=generator)
    graph = hahe.graph.build_graph(hv, train_examples, args.hyperedge_dropout, device)
    vocab_size = len(vocab.tokens)
    model = hahe.transformer.Transformer(graph, vocab_size, args.local_layers, args.global_layers, args.hidden_dim,
                                         args.local_heads, args.global_heads, True, args.local_dropout, args.global_dropout,
                                         args.decoder_activation, args.global_activation, True, False, True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
    steps_per_epoch = min(len(loader), args.max_steps_per_epoch) if args.max_steps_per_epoch else len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                    pct_start=args.warmup_proportion, anneal_strategy="linear", cycle_momentum=False)

    # Evaluation: HAHE features for the subject and object of every eval statement, in row order.
    eval_dataset = hahe.reader.MultiDataset(hv, eval_examples, max_arity, max_seq_len)
    keep = [i for i, f in enumerate(eval_dataset.features) if f.mask_position in (hd.SUBJECT_POSITION, hd.OBJECT_POSITION)]
    all_rows = train + valid + test
    return {
        "hahe": hahe, "stare": stare, "model": model, "optimizer": optimizer, "scheduler": scheduler, "loader": loader,
        "generator": generator, "device": device, "train_dataset": train_dataset, "eval_dataset": eval_dataset,
        "eval_keep": keep, "eval_rows": eval_rows, "vocab": vocab, "hv": hv, "n_entities": n_entities,
        "candidates": hd.candidate_entities(all_rows, n_entities), "index": hd.filter_index(all_rows),
        "max_arity": max_arity, "max_seq_len": max_seq_len, "steps_per_epoch": steps_per_epoch,
        "graph_info": {"statements_in_graph": len(graph_rows), "global_hyperedges": int(graph[2]),
                       "train_features": len(train_dataset), "max_arity": max_arity},
        "params": sum(p.numel() for p in model.parameters()),
        "eval_split": "test" if args.use_test else "valid",
    }


def train_one_epoch(state: dict[str, Any], args: argparse.Namespace) -> tuple[float, int]:
    model, optimizer, scheduler, device = state["model"], state["optimizer"], state["scheduler"], state["device"]
    hv = state["hv"]
    model.train()
    losses = []
    for step, item in enumerate(state["loader"]):
        if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
            break
        if args.step_sleep_ms > 0 and step > 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            time.sleep(args.step_sleep_ms / 1000.0)
        input_ids, input_mask, mask_position, mask_label, mask_output, edge_labels, query_type = (t.to(device) for t in item)
        result = model(input_ids, input_mask, mask_position, mask_output, edge_labels)
        entities, relations = (query_type == 1), (query_type == -1)
        label_entity = mask_output[entities] * (args.entity_soft / (hv.num_entities - 1))
        label_entity[torch.arange(label_entity.shape[0]), mask_label[entities]] = 1 - args.entity_soft
        label_relation = mask_output[relations] * (args.relation_soft / (hv.num_relations - 1))
        label_relation[torch.arange(label_relation.shape[0]), mask_label[relations]] = 1 - args.relation_soft
        loss1 = torch.nn.functional.cross_entropy(result[entities], label_entity, reduction="none")
        loss2 = torch.nn.functional.cross_entropy(result[relations], label_relation, reduction="none")
        loss = torch.cat((loss1, loss2)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
    return (float(np.mean(losses)) if losses else float("nan")), len(losses)


@torch.no_grad()
def evaluate(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    model, device, vocab = state["model"], state["device"], state["vocab"]
    ds, keep, rows = state["eval_dataset"], state["eval_keep"], state["eval_rows"]
    off, E = vocab.entity_offset, state["n_entities"]
    model.eval()
    left, right, left_native, right_native = [], [], [], []
    for start in range(0, len(keep), args.eval_batch):
        idx = keep[start : start + args.eval_batch]
        batch = [ds[i] for i in idx]
        tensors = [torch.as_tensor(np.stack([b[j] for b in batch])).to(device) for j in range(7)]
        input_ids, input_mask, mask_position, _, mask_output, edge_labels, _ = tensors
        logits = model(input_ids, input_mask, mask_position, mask_output, edge_labels).float().cpu().numpy()
        for i, scores_vocab in zip(idx, logits, strict=True):
            f = ds.features[i]
            row = rows[f.example_id]
            s, r, o = int(row[0]), int(row[1]), int(row[2])
            quals = tuple(int(x) for x in row[3:])
            scores = np.full(E, hd.EXCLUDED_SCORE)
            scores[1:] = scores_vocab[off : off + E - 1]  # HAHE entity token off+(eid-1) -> StarE eid
            if f.mask_position == hd.OBJECT_POSITION:
                target, known = o, state["index"][("o", s, r, quals)]
                left.append(hd.stare_rank(scores, target, known, state["candidates"]))
                left_native.append(hd.hahe_native_rank(scores, target, known))
            else:
                target, known = s, state["index"][("s", o, r, quals)]
                right.append(hd.stare_rank(scores, target, known, state["candidates"]))
                right_native.append(hd.hahe_native_rank(scores, target, known))
    lm, rm = hd.summarise(left), hd.summarise(right)
    return {
        "metrics": hd.mean_of_sides(lm, rm), "object_prediction": lm, "subject_prediction": rm,
        "hahe_native_ht": hd.summarise(left_native + right_native),
        "queries": len(left) + len(right), "time_s": time.time() - t0,
    }


def checkpoint_payload(state: dict[str, Any], epochs_done: int) -> dict[str, Any]:
    return {
        "epochs_done": epochs_done, "model": state["model"].state_dict(), "optimizer": state["optimizer"].state_dict(),
        "scheduler": state["scheduler"].state_dict(), "loader_generator": state["generator"].get_state(),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }


def restore(state: dict[str, Any], payload: dict[str, Any]) -> int:
    saved, expected = payload["scheduler"].get("total_steps"), state["scheduler"].total_steps
    if saved != expected:  # OneCycleLR spans the whole run: resuming with another --epochs would bend the schedule
        raise ValueError(f"checkpoint schedule has {saved} total steps, this run expects {expected}; resume with the same --epochs")
    state["model"].load_state_dict(payload["model"])
    state["optimizer"].load_state_dict(payload["optimizer"])
    state["scheduler"].load_state_dict(payload["scheduler"])
    state["generator"].set_state(payload["loader_generator"].cpu())
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    return int(payload["epochs_done"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt, metrics_path = out / "state.pt", out / "metrics.jsonl"
    if metrics_path.exists() and not args.resume:
        raise FileExistsError(f"{metrics_path} exists; pass --resume or choose another --out")
    started = time.time()
    state = prepare(args)
    device = state["device"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_config = {
        "arm": args.arm, "seed": args.seed, "epochs_target": args.epochs, "eval_every": args.eval_every, "chunk": args.chunk,
        "eval_split": state["eval_split"], "eval_queries": 2 * len(state["eval_rows"]), "device": str(device),
        "params": state["params"], "graph": state["graph_info"], "max_seq_len": state["max_seq_len"],
        "steps_per_epoch": state["steps_per_epoch"], "candidate_entities": int(state["candidates"].sum()),
        "hahe_commit": state["hahe"].commit, "stare_commit": state["stare"].commit,
        "hyperparameters": {k: getattr(args, k) for k in HAHE_WD50K}, "step_sleep_ms": args.step_sleep_ms,
        "eval_limit": args.eval_limit, "max_steps_per_epoch": args.max_steps_per_epoch, "torch": torch.__version__, "resumed": False,
    }
    epochs_done = 0
    if args.resume and ckpt.exists():
        epochs_done = restore(state, load_checkpoint(ckpt, map_location="cpu"))
        kept, dropped = prune_jsonl(metrics_path, max_epoch=epochs_done)
        run_config.update(resumed=True, resumed_from_epoch=epochs_done, metrics_rows_dropped_on_resume=dropped)
        print(f"[resume] continuing from {epochs_done} completed epochs; metrics.jsonl kept {kept} rows, dropped {dropped}", flush=True)
    elif args.resume:
        prune_jsonl(metrics_path, max_epoch=0)
        print("[resume] no checkpoint found; starting from scratch", flush=True)
    write_json(out / "run_config.json", run_config)
    print(f"[start] arm=hahe seed={args.seed} params={state['params']} device={device} graph={state['graph_info']} "
          f"eval={state['eval_split']}:{run_config['eval_queries']} steps/epoch={state['steps_per_epoch']}", flush=True)
    for epoch in range(epochs_done, args.epochs):
        t0 = time.time()
        loss, steps = train_one_epoch(state, args)
        row: dict[str, Any] = {"epoch": epoch + 1, "train_loss": loss, "steps": steps, "epoch_time_s": time.time() - t0,
                               "sleep_s": max(steps - 1, 0) * args.step_sleep_ms / 1000.0,
                               "lr": float(state["scheduler"].get_last_lr()[0])}
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            row["eval"] = evaluate(state, args)
        if device.type == "cuda":
            row["peak_vram_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024**2)
        append_jsonl(metrics_path, row)
        line = f"[epoch {epoch + 1}/{args.epochs}] loss {loss:.5f} steps {steps} {row['epoch_time_s']:.1f}s"
        if "eval" in row:
            m = row["eval"]["metrics"]
            line += f" | {state['eval_split']} mrr {m['mrr']:.4f} h1 {m['hits_at 1']:.4f} h10 {m['hits_at 10']:.4f}"
        print(line, flush=True)
        if (epoch + 1) % args.chunk == 0 or epoch + 1 == args.epochs:
            atomic_torch_save(checkpoint_payload(state, epoch + 1), ckpt)
        if args.stop_after_epochs and epoch + 1 - epochs_done >= args.stop_after_epochs:
            print(f"[stop] leaving after {args.stop_after_epochs} epochs (--stop-after-epochs)", flush=True)
            return {"stopped_at": epoch + 1}
    rows = read_jsonl(metrics_path)
    evals = [(r["epoch"], r["eval"]) for r in rows if "eval" in r]
    best = max(evals, key=lambda it: it[1]["metrics"]["mrr"]) if evals else None
    summary = {
        **{k: run_config[k] for k in ("arm", "seed", "eval_split", "eval_queries", "params", "device", "graph", "candidate_entities", "hahe_commit")},
        "epochs_done": rows[-1]["epoch"] if rows else epochs_done, "epochs_target": args.epochs,
        "last_eval": {"epoch": evals[-1][0], **evals[-1][1]} if evals else None,
        "best_eval": {"epoch": best[0], **best[1]} if best else None,
        "train_time_s": float(sum(r["epoch_time_s"] for r in rows)),
        "mean_epoch_time_s": float(np.mean([r["epoch_time_s"] for r in rows])) if rows else None,
        "peak_vram_mb": max((r.get("peak_vram_mb", 0.0) for r in rows), default=None),
        "wall_this_process_s": time.time() - started, "resumed": run_config["resumed"],
    }
    write_json(out / "summary.json", summary)
    print(f"[done] {summary['last_eval']['metrics'] if summary['last_eval'] else 'no eval'}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = None
    if not args.no_lock:
        try:
            token = acquire_long_run_lock({"task": "hahe-same-harness-v1", "arm": args.arm, "seed": args.seed,
                                           "epochs": args.epochs, "out": args.out}, steal=args.steal_lock)
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
