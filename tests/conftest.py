from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from hypothesis import settings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


settings.register_profile("ci", deadline=None, derandomize=True, max_examples=20)
settings.load_profile("ci")


@pytest.fixture
def workspace_tmp_dir():
    created: list[Path] = []
    root = ROOT / ".harness" / "tmp"
    root.mkdir(parents=True, exist_ok=True)

    def factory(prefix: str) -> Path:
        path = root / prefix
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
        return path

    yield factory

    for path in created:
        shutil.rmtree(path, ignore_errors=True)
