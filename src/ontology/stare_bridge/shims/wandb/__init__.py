"""Offline stand-in for ``wandb``.

StarE imports wandb at module level in ``run.py`` and ``loops/loops.py`` even though every call is
behind ``config['WANDB']`` (False by default, and this bridge never turns it on). Installing the
real package would add a network-capable dependency for nothing; this stub makes the imports
succeed and records that it was used (``run_config.json`` lists which shims were active).
"""

from __future__ import annotations

__version__ = "0.0.0+hypergraph-eggroll-shim"

config: dict = {}


def init(*args, **kwargs):  # noqa: ARG001 - signature compatibility only
    return None


def log(*args, **kwargs):  # noqa: ARG001
    return None


def finish(*args, **kwargs):  # noqa: ARG001
    return None
