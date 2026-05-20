"""CoordGrid: a cooperative toroidal gridworld with a tunable coordination graph.

The whole point of this env is that the coordination *graph* is the
experimental variable for hypotheses H1–H3. Reward is the number of graph
edges whose endpoints are co-located on the same toroidal cell, so the
optimal joint policy is literally "edge-connected agents must rendezvous".
Agents not connected in the graph have no incentive to coordinate — which
is exactly the load-bearing property that makes the graph matter.

See ``docs/INTERFACES.md`` for the env protocol, and the project README /
Phase-0 plan for the experimental setup.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from gnnmarl.envs.base import StepResult


# Action layout. Order matters — tests and downstream code reason about it.
# 0 = stay, 1 = up (-y), 2 = down (+y), 3 = left (-x), 4 = right (+x).
_ACTION_DELTAS: np.ndarray = np.array(
    [
        (0, 0),
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
    ],
    dtype=np.int64,
)

_VALID_GRAPHS = ("ring", "line", "complete", "erdos_renyi", "grid2d")


class CoordGrid:
    """Toroidal gridworld where coordination structure is given by a graph.

    Parameters
    ----------
    n_agents
        Number of agents ``N``.
    grid_size
        Side length of the square toroidal grid.
    episode_steps
        Fixed episode length. ``done`` only fires on the final step.
    graph
        One of ``{"ring", "line", "complete", "erdos_renyi", "grid2d"}``.
    er_prob
        Edge probability for ``erdos_renyi``; ignored otherwise.
    seed
        Seed for the env-owned ``numpy`` RNG.
    """

    n_actions: int = 5

    def __init__(
        self,
        n_agents: int = 4,
        grid_size: int = 5,
        episode_steps: int = 25,
        graph: str = "ring",
        er_prob: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}")
        if episode_steps < 1:
            raise ValueError(f"episode_steps must be >= 1, got {episode_steps}")
        if graph not in _VALID_GRAPHS:
            raise ValueError(
                f"graph must be one of {_VALID_GRAPHS}, got {graph!r}"
            )
        if graph == "grid2d":
            side = int(round(math.sqrt(n_agents)))
            if side * side != n_agents:
                raise ValueError(
                    "grid2d requires n_agents to be a perfect square; "
                    f"got n_agents={n_agents}"
                )
            self._grid2d_side = side
        else:
            self._grid2d_side = 0

        if graph == "erdos_renyi" and not (0.0 <= er_prob <= 1.0):
            raise ValueError(
                f"er_prob must be in [0, 1], got {er_prob}"
            )

        self.n_agents = int(n_agents)
        self.grid_size = int(grid_size)
        self.episode_steps = int(episode_steps)
        self.graph_kind = graph
        self.er_prob = float(er_prob)

        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Build an initial adjacency so obs_dim / max_neighbors are defined
        # before the first reset. Deterministic graph kinds yield the final
        # adjacency immediately; erdos_renyi will be resampled on each reset.
        self._adj: np.ndarray = self._build_adjacency()

        # max_neighbors caps obs slots so obs_dim is constant across resamples.
        # For erdos_renyi we use the worst case (N - 1) rather than the
        # initial draw's max degree, otherwise resampling can change shape.
        if graph == "erdos_renyi":
            self.max_neighbors = max(1, self.n_agents - 1)
        else:
            self.max_neighbors = int(max(1, self._adj.sum(axis=1).max()))

        self.obs_dim = 2 + 2 * self.max_neighbors + self.n_agents
        self.state_dim = 2 * self.n_agents

        # Per-episode state.
        self._positions: np.ndarray = np.zeros((self.n_agents, 2), dtype=np.int64)
        self._episode_step: int = 0
        self._closed: bool = False

    # ------------------------------------------------------------------ env API

    def reset(self, *, seed: int | None = None) -> StepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Resample adjacency for graph kinds that are stochastic per episode.
        if self.graph_kind == "erdos_renyi":
            self._adj = self._build_adjacency()

        self._positions = self._rng.integers(
            low=0, high=self.grid_size, size=(self.n_agents, 2), dtype=np.int64
        )
        self._episode_step = 0

        return StepResult(
            obs=self._compute_obs(),
            state=self._compute_state(),
            reward=0.0,
            done=False,
            info={"episode_step": self._episode_step},
        )

    def step(self, actions: np.ndarray) -> StepResult:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n_agents,):
            raise ValueError(
                f"actions must have shape ({self.n_agents},), got {actions.shape}"
            )
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise ValueError(
                f"actions must be in [0, {self.n_actions}); got {actions.tolist()}"
            )

        deltas = _ACTION_DELTAS[actions]  # [N, 2]
        self._positions = (self._positions + deltas) % self.grid_size

        reward = self._compute_reward()
        self._episode_step += 1
        done = self._episode_step >= self.episode_steps

        return StepResult(
            obs=self._compute_obs(),
            state=self._compute_state(),
            reward=float(reward),
            done=bool(done),
            info={"episode_step": self._episode_step},
        )

    def adjacency(self) -> np.ndarray:
        return self._adj.copy()

    def close(self) -> None:
        self._closed = True

    # --------------------------------------------------------------- internals

    def _build_adjacency(self) -> np.ndarray:
        n = self.n_agents
        a = np.zeros((n, n), dtype=np.float32)

        if self.graph_kind == "ring":
            if n >= 2:
                for i in range(n):
                    j = (i + 1) % n
                    a[i, j] = 1.0
                    a[j, i] = 1.0
                if n == 2:
                    # Special case: with N=2 the ring is really a single edge.
                    # The loop above already set a[0,1]=a[1,0]=1; no double.
                    pass
        elif self.graph_kind == "line":
            for i in range(n - 1):
                a[i, i + 1] = 1.0
                a[i + 1, i] = 1.0
        elif self.graph_kind == "complete":
            a[:] = 1.0
            np.fill_diagonal(a, 0.0)
        elif self.graph_kind == "erdos_renyi":
            # Upper triangle Bernoulli, mirror to lower.
            upper = self._rng.random((n, n)) < self.er_prob
            upper = np.triu(upper, k=1)
            a = (upper | upper.T).astype(np.float32)
        elif self.graph_kind == "grid2d":
            side = self._grid2d_side
            for r in range(side):
                for c in range(side):
                    i = r * side + c
                    # Right neighbour.
                    if c + 1 < side:
                        j = r * side + (c + 1)
                        a[i, j] = 1.0
                        a[j, i] = 1.0
                    # Down neighbour.
                    if r + 1 < side:
                        j = (r + 1) * side + c
                        a[i, j] = 1.0
                        a[j, i] = 1.0
        else:  # pragma: no cover — guarded in __init__.
            raise ValueError(self.graph_kind)

        np.fill_diagonal(a, 0.0)
        return a

    def _compute_obs(self) -> np.ndarray:
        n = self.n_agents
        gs = float(self.grid_size)
        obs = np.zeros((n, self.obs_dim), dtype=np.float32)

        # Pre-extract neighbour lists once per call.
        for i in range(n):
            xi, yi = self._positions[i]
            # Own normalized position.
            obs[i, 0] = xi / gs
            obs[i, 1] = yi / gs

            # Relative positions of graph neighbours, zero-padded.
            neighbours = np.where(self._adj[i] > 0.0)[0]
            for slot, j in enumerate(neighbours[: self.max_neighbors]):
                xj, yj = self._positions[j]
                # Toroidal shortest signed delta. Positive points from i to j.
                dx = self._toroidal_delta(xj - xi)
                dy = self._toroidal_delta(yj - yi)
                base = 2 + 2 * slot
                obs[i, base] = dx / gs
                obs[i, base + 1] = dy / gs

            # Agent-id one-hot.
            obs[i, 2 + 2 * self.max_neighbors + i] = 1.0

        return obs

    def _toroidal_delta(self, d: int) -> int:
        """Wrap an integer delta to the shortest signed offset on the torus."""
        gs = self.grid_size
        d = int(d) % gs
        if d > gs // 2:
            d -= gs
        return d

    def _compute_state(self) -> np.ndarray:
        gs = float(self.grid_size)
        return (self._positions.astype(np.float32) / gs).reshape(-1)

    def _compute_reward(self) -> float:
        # Sum +1 per graph edge (i<j) whose endpoints share a cell.
        same = np.all(
            self._positions[:, None, :] == self._positions[None, :, :], axis=-1
        )
        upper_edges = np.triu(self._adj > 0.0, k=1)
        return float(np.sum(same & upper_edges))

    # --------------------------------------------------------- test helpers

    def _place_agents(self, positions: np.ndarray) -> None:
        """Test helper: deterministically place agents, bypassing random init.

        ``positions`` must be ``[N, 2]`` integer coordinates in ``[0, grid_size)``.
        """
        positions = np.asarray(positions, dtype=np.int64)
        if positions.shape != (self.n_agents, 2):
            raise ValueError(
                f"positions must have shape ({self.n_agents}, 2), "
                f"got {positions.shape}"
            )
        if np.any(positions < 0) or np.any(positions >= self.grid_size):
            raise ValueError(
                f"positions must be in [0, {self.grid_size}); got {positions.tolist()}"
            )
        self._positions = positions.copy()
