"""TokenMatch: an out-of-harness referential game that isolates the same
graph-structure mechanism as CoordGrid's pair-routing, with *no spatial grid,
no movement, and no navigation* -- so a positive result here shows the Stage C
structure effect is not an artifact of the gridworld.

Each agent privately holds a token ``c_i in [0, n_tokens)`` (it observes only its
own token, one-hot). A fixed perfect matching pairs the agents; agent ``i`` is
rewarded for OUTPUTTING (as its discrete action) its partner's token ``c_{p(i)}``.
The partner's token is never in agent ``i``'s observation, so it can only arrive
over the coordination graph -- and only the *correct* matching delivers the right
partner (a wrong matching delivers someone else's token; all-to-all delivers a
diluted mix). Tokens are resampled every episode, so no fixed convention works.

Comm graphs (returned by :meth:`adjacency`, fed to the GNN) are decoupled from the
fixed reward pairing exactly as in CoordGrid: ``matching`` => comm == task (true),
``matching_wrong`` / ``complete`` => mismatched controls.
"""

from __future__ import annotations

import numpy as np

from gnnmarl.envs.base import StepResult

_VALID_GRAPHS = ("matching", "matching_wrong", "complete")


class TokenMatch:
    """Cooperative referential matching game; graph structure is load-bearing."""

    def __init__(
        self,
        n_agents: int = 6,
        n_tokens: int = 5,
        episode_steps: int = 10,
        graph: str = "matching",
        seed: int | None = None,
    ) -> None:
        if n_agents < 2 or n_agents % 2 != 0:
            raise ValueError(f"n_agents must be even and >= 2, got {n_agents}")
        if n_tokens < 2:
            raise ValueError(f"n_tokens must be >= 2, got {n_tokens}")
        if episode_steps < 1:
            raise ValueError(f"episode_steps must be >= 1, got {episode_steps}")
        if graph not in _VALID_GRAPHS:
            raise ValueError(f"graph must be one of {_VALID_GRAPHS}, got {graph!r}")
        if graph == "matching_wrong" and n_agents < 4:
            raise ValueError("graph='matching_wrong' needs n_agents >= 4")

        self.n_agents = int(n_agents)
        self.n_tokens = int(n_tokens)
        self.episode_steps = int(episode_steps)
        self.graph_kind = graph
        self._rng = np.random.default_rng(seed)

        # Action = the agent's guess of its partner's token.
        self.n_actions = self.n_tokens
        # Observation = own token one-hot (the agent never sees another's token).
        self.obs_dim = self.n_tokens
        # Privileged state for the mixer = all tokens (normalized).
        self.state_dim = self.n_agents

        # Fixed canonical matching defines the reward partner (task structure),
        # independent of the communication graph.
        self._partner = np.empty(self.n_agents, dtype=np.int64)
        for k in range(self.n_agents // 2):
            self._partner[2 * k] = 2 * k + 1
            self._partner[2 * k + 1] = 2 * k

        self._adj = self._build_adjacency()
        self._tokens = np.zeros(self.n_agents, dtype=np.int64)
        self._episode_step = 0
        self._closed = False

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None) -> StepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._tokens = self._rng.integers(0, self.n_tokens, size=self.n_agents, dtype=np.int64)
        self._episode_step = 0
        return StepResult(obs=self._compute_obs(), state=self._compute_state(),
                          reward=0.0, done=False, info={"episode_step": 0})

    def step(self, actions: np.ndarray) -> StepResult:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n_agents,):
            raise ValueError(f"actions must have shape ({self.n_agents},), got {actions.shape}")
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise ValueError(f"actions must be in [0, {self.n_actions})")
        # +1 per agent that correctly outputs its partner's token.
        reward = float(np.sum(actions == self._tokens[self._partner]))
        self._episode_step += 1
        done = self._episode_step >= self.episode_steps
        return StepResult(obs=self._compute_obs(), state=self._compute_state(),
                          reward=reward, done=bool(done),
                          info={"episode_step": self._episode_step})

    def adjacency(self) -> np.ndarray:
        return self._adj.copy()

    def close(self) -> None:
        self._closed = True

    # ------------------------------------------------------------ internals
    def _build_adjacency(self) -> np.ndarray:
        n = self.n_agents
        a = np.zeros((n, n), dtype=np.float32)
        if self.graph_kind == "matching":
            for k in range(n // 2):
                a[2 * k, 2 * k + 1] = a[2 * k + 1, 2 * k] = 1.0
        elif self.graph_kind == "matching_wrong":
            for k in range(n // 2):
                i, j = 2 * k + 1, (2 * k + 2) % n
                a[i, j] = a[j, i] = 1.0
        elif self.graph_kind == "complete":
            a[:] = 1.0
            np.fill_diagonal(a, 0.0)
        return a

    def _compute_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        obs[np.arange(self.n_agents), self._tokens] = 1.0  # own token one-hot
        return obs

    def _compute_state(self) -> np.ndarray:
        return (self._tokens.astype(np.float32) / self.n_tokens)

    # --------------------------------------------------------- test helper
    def _set_tokens(self, tokens: np.ndarray) -> None:
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.shape != (self.n_agents,):
            raise ValueError(f"tokens must have shape ({self.n_agents},)")
        self._tokens = tokens.copy()
