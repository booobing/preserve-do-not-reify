"""Fetch (or verify) the pinned upstream HAHE checkout (docs/charter-hahe-same-harness-v1.md §9).

    uv run --frozen python .harness/scripts/fetch_hahe.py            # clone/checkout + verify
    uv run --frozen python .harness/scripts/fetch_hahe.py --verify-only

Clones the URL from `third_party/hahe.lock.json`, checks out the pinned commit, then verifies
(offline) that HEAD equals the pin and that the five `src/*.py` files match their sha256 in the
lock. Exit 0 only when every check passes. HAHE's data files are not used and not checked.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.ontology.hahe_bridge.upstream import LOCK_PATH  # noqa: E402
from src.ontology.stare_bridge.upstream import (  # noqa: E402
    UpstreamError,
    current_commit,
    read_lock,
    verify_checkout,
)


def git(*args: str) -> None:
    # Arguments are literals plus the URL/commit/path read from the tracked lock file.
    subprocess.run(["git", *args], check=True)  # noqa: S603, S607 - git from PATH by design


def ensure_checkout(lock: dict, root: Path) -> None:
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] cloning {lock['url']} -> {root}", flush=True)
        git("clone", "--quiet", lock["url"], str(root))
    if current_commit(root) != lock["commit"]:
        print(f"[fetch] checking out {lock['commit']}", flush=True)
        try:
            git("-C", str(root), "checkout", "--quiet", lock["commit"])
        except subprocess.CalledProcessError:
            git("-C", str(root), "fetch", "--quiet", "origin")
            git("-C", str(root), "checkout", "--quiet", lock["commit"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-only", action="store_true", help="no network; fail if the checkout is absent or off-pin")
    args = parser.parse_args(argv)
    lock = read_lock(LOCK_PATH)
    root = REPO_ROOT / lock["root"]
    if not args.verify_only:
        ensure_checkout(lock, root)
    try:
        report = verify_checkout(root, lock)
    except UpstreamError as exc:
        print(f"[verify] FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"[verify] OK commit={report['commit']} files={len(report['files'])} root={report['root']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
