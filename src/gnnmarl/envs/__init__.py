"""Environment registry and re-exports for Phase 0."""

from __future__ import annotations

from typing import Any

from gnnmarl.envs.base import MultiAgentEnv, StepResult
from gnnmarl.envs.coord_grid import CoordGrid

__all__ = ["MultiAgentEnv", "StepResult", "CoordGrid", "make_env"]


def make_env(name: str, **kwargs: Any) -> MultiAgentEnv:
    """Construct a registered environment by name.

    Currently supports:
    - ``"coord_grid"`` — :class:`CoordGrid`.
    """
    if name == "coord_grid":
        return CoordGrid(**kwargs)
    raise ValueError(f"Unknown env name: {name!r}")
