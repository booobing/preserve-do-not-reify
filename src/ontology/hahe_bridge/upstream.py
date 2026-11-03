"""Locate, verify and import the pinned HAHE checkout (third_party/HAHE, lock third_party/hahe.lock.json).

Only `reader`, `graph` and `transformer` are imported; upstream `run.py` parses arguments at import
time and is never imported. Verification reuses the StarE bridge helpers (commit + file sha256).
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..stare_bridge.upstream import REPO_ROOT, read_lock, verify_checkout

LOCK_PATH = REPO_ROOT / "third_party" / "hahe.lock.json"


@dataclass
class HaheUpstream:
    root: Path
    commit: str
    reader: Any
    graph: Any
    transformer: Any


def load_hahe(*, verify: bool = True) -> HaheUpstream:
    lock = read_lock(LOCK_PATH)
    root = REPO_ROOT / lock["root"]
    commit = verify_checkout(root, lock)["commit"] if verify else lock["commit"]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return HaheUpstream(
        root=root,
        commit=commit,
        reader=importlib.import_module("reader"),
        graph=importlib.import_module("graph"),
        transformer=importlib.import_module("transformer"),
    )
