"""Locate, verify and import the pinned upstream StarE checkout.

The checkout lives at ``third_party/StarE`` (git-ignored) and is pinned by
``third_party/stare.lock.json`` (tracked). ``load_upstream`` refuses to run on a checkout whose
commit or data files differ from the lock, so every result can be tied to one upstream revision.

Importing upstream is deliberately explicit: StarE is a collection of top-level modules
(``data_loaders``, ``models``, ``loops``, ``utils``, ``run``) that import each other by those
names, so its root goes on ``sys.path``; its data loader reads ``./data/clean/...`` relative to the
working directory, so ``load_data`` enters the checkout only for the duration of that call.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "third_party" / "stare.lock.json"
SHIMS_DIR = Path(__file__).resolve().parent / "shims"
SHIMMED_MODULES: tuple[str, ...] = ("torch_scatter", "wandb")


class UpstreamError(RuntimeError):
    """The pinned checkout is missing, on the wrong commit, or has altered data files."""


def read_lock(path: Path | str = LOCK_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stare_root(lock: Mapping[str, Any] | None = None) -> Path:
    lock = lock or read_lock()
    return REPO_ROOT / lock["root"]


def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_mismatches(base: Path | str, expected: Mapping[str, str]) -> list[str]:
    """Human-readable list of files under `base` whose sha256 differs from `expected` (or are missing)."""
    problems: list[str] = []
    for relative, digest in expected.items():
        target = Path(base) / relative
        if not target.exists():
            problems.append(f"missing: {relative}")
        elif sha256_of(target) != digest:
            problems.append(f"sha256 mismatch: {relative}")
    return problems


def current_commit(root: Path | str) -> str:
    # The checkout is a local directory this repo created from a pinned URL; the only input is a path.
    completed = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607 - git resolved from PATH on purpose
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_checkout(root: Path | str | None = None, lock: Mapping[str, Any] | None = None) -> dict[str, Any]:
    lock = lock or read_lock()
    root = Path(root) if root is not None else stare_root(lock)
    if not root.exists():
        raise UpstreamError(f"{root} is missing; run `uv run --frozen python .harness/scripts/fetch_stare.py`")
    commit = current_commit(root)
    if commit != lock["commit"]:
        raise UpstreamError(f"{root} is at {commit}, lock pins {lock['commit']}")
    problems = file_mismatches(root, lock["files_sha256"]) + file_mismatches(REPO_ROOT, lock["repo_mirror_sha256"])
    if problems:
        raise UpstreamError("; ".join(problems))
    return {"root": str(root), "commit": commit, "files": sorted(lock["files_sha256"])}


def install_shims(names: tuple[str, ...] = SHIMMED_MODULES) -> dict[str, str]:
    """Make `torch_scatter` and `wandb` importable; report per module whether the real package or
    the shim answered. The shim directory is only added to sys.path when a real import fails."""
    report: dict[str, str] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError:
            if str(SHIMS_DIR) not in sys.path:
                sys.path.insert(0, str(SHIMS_DIR))
            module = importlib.import_module(name)
        version = str(getattr(module, "__version__", ""))
        report[name] = "shim" if "hypergraph-eggroll-shim" in version else f"installed {version}"
    return report


@contextlib.contextmanager
def working_directory(path: Path | str) -> Iterator[None]:
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclass
class Upstream:
    """Handles on the upstream modules this bridge uses, plus how they were obtained."""

    root: Path
    commit: str
    shims: dict[str, str]
    run: Any
    DataManager: Any
    StarE_Transformer: Any
    MultiClassSampler: Any
    EvaluationBench: Any
    combine: Any
    get_param: Any
    extra: dict[str, Any] = field(default_factory=dict)

    def default_config(self) -> dict[str, Any]:
        config = dict(self.run.DEFAULT_CONFIG)
        config["STAREARGS"] = dict(self.run.STAREARGS)
        return config

    def load_data(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """`DataManager.load(config)()` with the working directory inside the checkout, because
        the upstream loader opens `./data/clean/<dataset>/statements/*.txt`."""
        with working_directory(self.root):
            return self.DataManager.load(config=config)()


def load_upstream(root: Path | str | None = None, *, verify: bool = True) -> Upstream:
    lock = read_lock()
    root = Path(root) if root is not None else stare_root(lock)
    commit = verify_checkout(root, lock)["commit"] if verify else current_commit(root)
    shims = install_shims()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    run = importlib.import_module("run")  # defines DEFAULT_CONFIG / STAREARGS; its main is guarded
    data_manager = importlib.import_module("data_loaders.data_manager")
    models_statements = importlib.import_module("models.models_statements")
    sampler = importlib.import_module("loops.sampler")
    evaluation = importlib.import_module("loops.evaluation")
    utils = importlib.import_module("utils.utils")
    utils_gcn = importlib.import_module("utils.utils_gcn")
    return Upstream(
        root=root,
        commit=commit,
        shims=shims,
        run=run,
        DataManager=data_manager.DataManager,
        StarE_Transformer=models_statements.StarE_Transformer,
        MultiClassSampler=sampler.MultiClassSampler,
        EvaluationBench=evaluation.EvaluationBenchGNNMultiClass,
        combine=utils.combine,
        get_param=utils_gcn.get_param,
        extra={"metrics": (evaluation.acc, evaluation.mrr, evaluation.mr, evaluation.hits_at)},
    )
