"""Per-machine kernel launch tuning.

Threadgroup shapes are measured, not derived: the defaults below are the
best shapes from sweeps on Apple M1 (see harness/tune.py). On a different
GPU (e.g. M5 Max) run

    python harness/tune.py

which re-sweeps every kernel and writes ``tuning_local.json`` next to this
module; that file (or one pointed to by ``MLX_TOMO_TUNING``) overrides the
defaults. Delete it to fall back.
"""

from __future__ import annotations

import json
import os

# Measured on Apple M1 (8-core GPU), 512^3 volume, 512^2 detector, with the
# 0.2.0 kernels (harness/tune.py, cross-checked at 100 views with bench_*.py).
# ax_interp (8,8,2) is 7% faster than (8,8,1) at 100 views; ax_siddon's
# z-depth-4 shape is a 1.66x win over (32,4,1); atb is within 3% across every
# shape swept. Validate sweep winners against the full benches.
DEFAULT_TG = {
    "ax_interp": (8, 8, 2),
    "ax_siddon": (16, 4, 4),
    "atb": (8, 8, 1),
    "texture": (8, 4),
}

_LOCAL = os.path.join(os.path.dirname(__file__), "tuning_local.json")
_cache = None


def _load():
    global _cache
    if _cache is not None:
        return _cache
    _cache = dict(DEFAULT_TG)
    path = os.environ.get("MLX_TOMO_TUNING", _LOCAL)
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            for k, v in data.get("threadgroups", {}).items():
                if k in _cache:
                    _cache[k] = tuple(int(x) for x in v)
        except (OSError, ValueError, TypeError):
            pass  # bad tuning file: keep defaults
    return _cache


def get(kind):
    """Threadgroup tuple for a kernel kind (see DEFAULT_TG keys)."""
    return _load()[kind]


def reload():
    global _cache
    _cache = None
    return _load()
