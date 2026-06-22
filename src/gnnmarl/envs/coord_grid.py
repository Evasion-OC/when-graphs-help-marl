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

_VALID_GRAPHS = (
    "ring",
    "line",
    "complete",
    "erdos_renyi",
    "grid2d",
    "matching",
    "matching_wrong",
)
_VALID_OBS_MODES = ("full", "ego", "radius")


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
    obs_mode
        Local observability of neighbours. ``"full"`` (default) reveals every
        graph neighbour's relative position; ``"ego"`` withholds the neighbour
        block entirely (the agent sees only its own cell + id, so the graph is
        the sole channel to neighbour information); ``"radius"`` reveals only
        neighbours within ``obs_radius`` toroidal Manhattan steps. ``obs_dim``
        is identical across modes (hidden slots are zero-filled), so every
        algorithm receives the same input shape.
    obs_radius
        Required when ``obs_mode="radius"``; the visibility radius in cells.
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
        max_neighbors_override: int | None = None,
        obs_mode: str = "full",
        obs_radius: int | None = None,
        goal_routing: bool = False,
        goal_dense: bool = False,
        pair_routing: bool = False,
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

        if graph in ("matching", "matching_wrong"):
            if n_agents % 2 != 0:
                raise ValueError(
                    f"graph={graph!r} requires an even n_agents (a perfect "
                    f"matching); got n_agents={n_agents}"
                )
            if graph == "matching_wrong" and n_agents < 4:
                raise ValueError(
                    "graph='matching_wrong' needs n_agents >= 4 (with N=2 the "
                    "only matching is the canonical one)"
                )

        if graph == "erdos_renyi" and not (0.0 <= er_prob <= 1.0):
            raise ValueError(
                f"er_prob must be in [0, 1], got {er_prob}"
            )

        # Observability mode controls how much of a neighbour an agent can see
        # *locally* in its own observation vector. The graph adjacency (used by
        # GNN-QMIX for message passing) is unaffected — so under "ego" the graph
        # is the only channel through which neighbour information can reach an
        # agent at decentralised execution. obs_dim is held constant across
        # modes (hidden slots are zero-filled) so every algorithm sees the same
        # input shape and the contrast stays clean.
        if obs_mode not in _VALID_OBS_MODES:
            raise ValueError(
                f"obs_mode must be one of {_VALID_OBS_MODES}, got {obs_mode!r}"
            )
        if obs_mode == "radius":
            if obs_radius is None or obs_radius < 0:
                raise ValueError(
                    "obs_mode='radius' requires obs_radius >= 0, "
                    f"got {obs_radius!r}"
                )
        self.obs_mode = str(obs_mode)
        self.obs_radius = None if obs_radius is None else int(obs_radius)

        # Goal-routing task: each episode samples a secret goal cell that only
        # the source agent (agent 0) observes. Reward is the number of agents
        # co-located at that goal. The task is convention-proof (the goal is
        # random per episode, so no fixed rendezvous policy works) and the
        # non-source agents can only learn the goal if it is *routed* to them
        # over the coordination graph -- so GNN-QMIX has a channel the no-graph
        # controls lack, despite all algorithms receiving identical observations.
        self.goal_routing = bool(goal_routing)
        # Dense shaping makes the goal *learnable*: instead of a sparse +1 only
        # when exactly on the goal, each agent scores by how close it is. This
        # lets the goal-seeing source learn to approach, turning the task into a
        # valid test of whether the graph routes the goal to the other agents.
        self.goal_dense = bool(goal_dense)

        # Pair-routing task: every agent holds a *private* goal it alone observes,
        # and is rewarded for reaching its graph *partner's* goal (a fixed perfect
        # matching defines the partners). Partners must therefore swap goals over
        # their edge. Unlike goal_routing's single broadcast goal, the information
        # an agent needs is held by one specific other agent -- so the
        # *communication graph must match the task structure*: a wrong matching
        # (matching_wrong) or all-to-all (complete) delivers the wrong / diluted
        # partner, and only the true matching routes the right goal. This makes
        # graph STRUCTURE (not merely having a channel) the load-bearing variable.
        self.pair_routing = bool(pair_routing)
        if self.pair_routing and self.goal_routing:
            raise ValueError("pair_routing and goal_routing are mutually exclusive")
        if self.pair_routing and n_agents % 2 != 0:
            raise ValueError(
                f"pair_routing requires an even n_agents (a perfect matching "
                f"defines the partners); got n_agents={n_agents}"
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
        # ``max_neighbors_override`` lets Phase 4 force the same obs_dim
        # across different graph kinds so the diameter sweep is not
        # confounded with input dimension.
        if max_neighbors_override is not None:
            if max_neighbors_override < 1 or max_neighbors_override > self.n_agents - 1:
                raise ValueError(
                    f"max_neighbors_override must be in [1, n_agents-1]; "
                    f"got {max_neighbors_override}"
                )
            self.max_neighbors = int(max_neighbors_override)
        elif graph == "erdos_renyi":
            self.max_neighbors = max(1, self.n_agents - 1)
        else:
            self.max_neighbors = int(max(1, self._adj.sum(axis=1).max()))

        # Goal block (3 slots: has-goal flag + normalized goal x/y) is appended
        # in goal-routing mode (filled only for the source agent) and in
        # pair-routing mode (filled for *every* agent with its own private goal).
        self._goal_slots = 3 if (self.goal_routing or self.pair_routing) else 0
        self.obs_dim = 2 + 2 * self.max_neighbors + self.n_agents + self._goal_slots
        self.state_dim = 2 * self.n_agents

        # Canonical perfect matching used by pair-routing to define each agent's
        # partner: (0,1), (2,3), ... This is the *task* structure and is fixed
        # regardless of the communication ``graph`` -- so feeding the GNN the
        # matching graph means comm == task (true), while matching_wrong /
        # complete deliberately mismatch it (the structure controls).
        if self.pair_routing:
            self._partner = np.empty(self.n_agents, dtype=np.int64)
            for k in range(self.n_agents // 2):
                self._partner[2 * k] = 2 * k + 1
                self._partner[2 * k + 1] = 2 * k

        # Per-episode state.
        self._positions: np.ndarray = np.zeros((self.n_agents, 2), dtype=np.int64)
        self._goal: np.ndarray = np.zeros(2, dtype=np.int64)
        self._goals: np.ndarray = np.zeros((self.n_agents, 2), dtype=np.int64)
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
        if self.goal_routing:
            self._goal = self._rng.integers(
                low=0, high=self.grid_size, size=2, dtype=np.int64
            )
        if self.pair_routing:
            self._goals = self._rng.integers(
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
        elif self.graph_kind == "matching":
            # Canonical perfect matching: edges (0,1), (2,3), ... Each agent has
            # exactly one neighbour -- its task partner -- so message passing
            # routes precisely the partner's state and nothing else.
            for k in range(n // 2):
                a[2 * k, 2 * k + 1] = 1.0
                a[2 * k + 1, 2 * k] = 1.0
        elif self.graph_kind == "matching_wrong":
            # A perfect matching that shares NO edge with the canonical one:
            # (1,2), (3,4), ..., (n-1,0). Same density (1-regular) as `matching`
            # but pairs every agent with a *non-partner* -- the structure control
            # that isolates "right graph" from "a graph of the right size".
            for k in range(n // 2):
                i = 2 * k + 1
                j = (2 * k + 2) % n
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
            # Under "ego" the whole neighbour block is withheld (the agent must
            # rely on the graph to learn about neighbours); under "radius" only
            # neighbours within ``obs_radius`` toroidal Manhattan steps are
            # revealed; under "full" all graph neighbours are revealed.
            if self.obs_mode != "ego":
                neighbours = np.where(self._adj[i] > 0.0)[0]
                for slot, j in enumerate(neighbours[: self.max_neighbors]):
                    xj, yj = self._positions[j]
                    # Toroidal shortest signed delta. Positive points from i to j.
                    dx = self._toroidal_delta(xj - xi)
                    dy = self._toroidal_delta(yj - yi)
                    if self.obs_mode == "radius" and (
                        abs(dx) + abs(dy) > self.obs_radius
                    ):
                        # Out of sight: leave this neighbour's slot zero-filled.
                        continue
                    base = 2 + 2 * slot
                    obs[i, base] = dx / gs
                    obs[i, base + 1] = dy / gs

            # Agent-id one-hot.
            obs[i, 2 + 2 * self.max_neighbors + i] = 1.0

        # Goal-routing: only the source agent (index 0) sees the goal. Other
        # agents must receive it via graph message passing. Goal block lives in
        # the final 3 slots: [has_goal flag, goal_x / gs, goal_y / gs].
        if self.goal_routing:
            base = 2 + 2 * self.max_neighbors + n
            obs[0, base] = 1.0
            obs[0, base + 1] = self._goal[0] / gs
            obs[0, base + 2] = self._goal[1] / gs

        # Pair-routing: every agent sees ONLY its own private goal (the thing its
        # partner must reach). It never sees its partner's goal directly -- under
        # any obs_mode -- so the partner's goal can reach it only via the graph.
        # The own-goal block is shown regardless of obs_mode (it is the agent's
        # own private state, not a neighbour's).
        if self.pair_routing:
            base = 2 + 2 * self.max_neighbors + n
            for i in range(n):
                obs[i, base] = 1.0
                obs[i, base + 1] = self._goals[i, 0] / gs
                obs[i, base + 2] = self._goals[i, 1] / gs

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
        if self.pair_routing:
            # Dense: each agent scores by toroidal proximity to its PARTNER's
            # private goal, summed over agents. An agent that has not received its
            # partner's goal (no graph channel, or the wrong/diluted one) cannot
            # systematically approach it, so it stays near the random floor; only
            # routing the correct partner's goal lets the team score.
            targets = self._goals[self._partner]                 # [N, 2]
            raw = np.abs(self._positions - targets)
            tor = np.minimum(raw, self.grid_size - raw)          # toroidal
            dist = tor.sum(axis=-1)                               # [N]
            max_dist = max(1, 2 * (self.grid_size // 2))
            return float(np.sum(1.0 - dist / max_dist))
        if self.goal_routing:
            if self.goal_dense:
                # Shaped: each agent scores 1 - (toroidal Manhattan distance to
                # goal)/max_dist, in [0, 1]; summed over agents. An agent with no
                # goal information cannot systematically reduce its distance, so
                # the no-graph control's non-source agents stay near the random
                # baseline while a routed graph can drive them to the goal.
                gs = self.grid_size
                raw = np.abs(self._positions - self._goal[None, :])     # [N, 2]
                tor = np.minimum(raw, gs - raw)                         # toroidal
                dist = tor.sum(axis=-1)                                 # [N]
                max_dist = 2 * (gs // 2)
                return float(np.sum(1.0 - dist / max_dist))
            # +1 per agent co-located with the secret goal cell. Only the source
            # observes the goal directly, so non-source agents must have it
            # routed to them over the graph to score.
            at_goal = np.all(self._positions == self._goal[None, :], axis=-1)
            return float(np.sum(at_goal))
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
