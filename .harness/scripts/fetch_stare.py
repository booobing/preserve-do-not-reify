"""Fetch (or verify) the pinned upstream StarE checkout (charter §9, checkpoint `stare_fetched`).

    uv run --frozen python .harness/scripts/fetch_stare.py            # clone/checkout + verify
    uv run --frozen python .harness/scripts/fetch_stare.py --verify-only

This is the only step of the bridge that touches the network. It clones the URL from
`third_party/stare.lock.json`, checks out the pinned commit, and then verifies (offline) that
HEAD equals the pin and that the three WD50K_100 statement files are byte-identical to the
repo's own `data/wd50k_100/` copies. Exit 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.ontology.stare_bridge.upstream import (  # noqa: E402
    UpstreamError,
    current_commit,
    read_lock,
    stare_root,
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
    lock = read_lock()
    root = stare_root(lock)
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
