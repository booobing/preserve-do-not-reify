"""Run-time plumbing shared by the runner and its tests: the long-run lock, atomic checkpoints,
append-only metrics, evaluation trimming. Pure and torch-free except for the checkpoint helpers.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .upstream import REPO_ROOT

LONG_RUN_LOCK = REPO_ROOT / ".harness" / "tmp" / "LONG_RUN.lock"


class LockHeld(RuntimeError):
    """Another long run (or a stale lock from a hard reset) owns `.harness/tmp/LONG_RUN.lock`."""


def acquire_long_run_lock(info: Mapping[str, Any], *, path: Path = LONG_RUN_LOCK, steal: bool = False) -> str:
    """Write the lock with a unique token and return it. Refuses if a lock exists unless `steal`.

    Liveness is not probed: on Windows `os.kill(pid, 0)` terminates the process, so a stale lock
    after a machine reset is cleared by a human (or `--steal-lock`) rather than by a heuristic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not steal:
        raise LockHeld(f"{path} exists: {path.read_text(encoding='utf-8').strip()[:300]}")
    token = f"{os.getpid()}-{time.time_ns()}"
    payload = {"token": token, "pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **dict(info)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return token


def release_long_run_lock(token: str, *, path: Path = LONG_RUN_LOCK) -> bool:
    """Remove the lock only if it still carries our token (never delete someone else's)."""
    if not path.exists():
        return False
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if current.get("token") != token:
        return False
    path.unlink()
    return True


def append_jsonl(path: Path | str, row: Mapping[str, Any]) -> None:
    """Append one row and fsync: a hard reset otherwise leaves a zero-filled tail (seen 2026-10-08)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Valid rows only. Zero-filled or half-written lines left by a power loss are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().decode("utf-8", errors="replace").split("\n"):
        text = line.strip("\x00").strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return rows


def prune_jsonl(path: Path | str, *, max_epoch: int) -> tuple[int, int]:
    """Rewrite a metrics file so it holds each epoch <= `max_epoch` exactly once, in order.

    After a reset the file can contain rows past the last checkpoint (those epochs will be trained
    again) and corrupted lines; resuming must not leave duplicates. Returns (kept, dropped).
    """
    path = Path(path)
    if not path.exists():
        return 0, 0
    raw_lines = [ln for ln in path.read_bytes().decode("utf-8", errors="replace").split("\n") if ln.strip("\x00").strip()]
    kept: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in read_jsonl(path):
        epoch = row.get("epoch")
        if not isinstance(epoch, int) or epoch > max_epoch or epoch in seen:
            continue
        seen.add(epoch)
        kept.append(row)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return len(kept), len(raw_lines) - len(kept)


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def trim_eval(eval_array: np.ndarray, limit: int | None) -> np.ndarray:
    """Keep the first `limit` direct statements and their `limit` reciprocal twins.

    Upstream lays the evaluation array out as [direct rows | reciprocal rows] and splits it in
    half for object- and subject-prediction, so a trim has to take the same prefix of each half.
    `limit` None or <= 0 returns the array unchanged. Smoke runs and tests use this; pilots do not.
    """
    if limit is None or limit <= 0:
        return eval_array
    half = eval_array.shape[0] // 2
    if eval_array.shape[0] != 2 * half:
        raise ValueError("evaluation array must have an even number of rows (direct + reciprocal)")
    keep = min(int(limit), half)
    return np.concatenate([eval_array[:keep], eval_array[half : half + keep]], axis=0)


def atomic_torch_save(payload: Mapping[str, Any], path: Path | str) -> None:
    """Write to a sibling temp file and rename, so a reset mid-write leaves the previous checkpoint."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())  # the rename alone does not make the bytes durable across a power loss
    os.replace(tmp, path)


def load_checkpoint(path: Path | str, *, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    # Our own file, written by atomic_torch_save; it holds optimizer/scheduler/RNG state, which
    # `weights_only=True` rejects.
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def epoch_budget(seconds_per_epoch: float, *, minutes: float = 45.0, floor: int = 50, ceiling: int = 401) -> int:
    """Charter §5-b: E_pilot = min(401, floor(45 min / s_epoch)), with a floor of 50."""
    if seconds_per_epoch <= 0:
        raise ValueError("seconds_per_epoch must be positive")
    return max(floor, min(ceiling, int((minutes * 60.0) // seconds_per_epoch)))
