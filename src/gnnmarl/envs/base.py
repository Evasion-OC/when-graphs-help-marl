"""Binding env interface for Phase 0.

See ``docs/INTERFACES.md`` for the contract — this module is the single source
of truth that algos, trainers, and loggers depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class StepResult:
    """The return type of ``reset`` and ``step``.

    Fields
    ------
    obs:    ``[N, obs_dim]`` float32 — per-agent local observation.
    state:  ``[state_dim]`` float32 — privileged global state for the mixer.
    reward: scalar cooperative reward (``0.0`` on reset).
    done:   episode termination flag (``False`` on reset).
    info:   free-form dict; must include ``episode_step: int``.
    """

    obs: np.ndarray
    state: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MultiAgentEnv(Protocol):
    """Cooperative multi-agent env protocol.

    All Phase-0 environments must conform to this. The adjacency is static
    within an episode but may change across episodes (e.g. Erdős–Rényi
    resampled on every ``reset``).
    """

    n_agents: int
    obs_dim: int
    state_dim: int
    n_actions: int

    def reset(self, *, seed: int | None = None) -> StepResult: ...

    def step(self, actions: np.ndarray) -> StepResult: ...

    def adjacency(self) -> np.ndarray:
        """Return ``[N, N]`` float32 adjacency: symmetric, no self-loops, 0/1."""
        ...

    def close(self) -> None: ...
