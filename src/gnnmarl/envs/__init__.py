"""Environment registry and re-exports for Phase 0."""

from __future__ import annotations

from typing import Any

from gnnmarl.envs.base import MultiAgentEnv, StepResult
from gnnmarl.envs.coord_grid import CoordGrid
from gnnmarl.envs.token_match import TokenMatch

__all__ = ["MultiAgentEnv", "StepResult", "CoordGrid", "TokenMatch", "make_env"]


def make_env(name: str, **kwargs: Any) -> MultiAgentEnv:
    """Construct a registered environment by name.

    Supports:
    - ``"coord_grid"`` — :class:`CoordGrid`.
    - ``"mpe:simple_spread"``, ``"mpe:simple_tag"`` — Phase-3 adapters
      around PettingZoo MPE (lazy import; requires the ``[mpe]`` extra).
    - ``"mpe_reference_pairs"`` — Phase-D external-validity vehicle:
      :class:`~gnnmarl.envs.mpe_reference_pairs.MPEReferencePairs`, K
      composed ``simple_reference`` pairs with comm disabled (lazy import;
      requires the ``[mpe]`` extra).
    """
    if name == "coord_grid":
        return CoordGrid(**kwargs)
    if name == "token_match":
        return TokenMatch(**kwargs)
    if name == "mpe_reference_pairs":
        from gnnmarl.envs.mpe_reference_pairs import MPEReferencePairs

        return MPEReferencePairs(**kwargs)
    if name.startswith("mpe:"):
        from gnnmarl.envs.mpe_adapter import MPEEnvAdapter

        return MPEEnvAdapter(env_name=name.split(":", 1)[1], **kwargs)
    if name.startswith("lbf:") or name.startswith("rware:"):
        from gnnmarl.envs.epymarl_adapter import GymMARLAdapter

        return GymMARLAdapter(env_id=name.split(":", 1)[1], **kwargs)
    raise ValueError(f"Unknown env name: {name!r}")
