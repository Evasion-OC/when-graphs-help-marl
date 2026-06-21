"""Adapter for EPyMARL-style cooperative gym benchmarks -> MultiAgentEnv.

Wraps the two standard cooperative MARL benchmarks from the EPyMARL suite
\\citep{papoudakis_2021_benchmarking} in our ``MultiAgentEnv`` protocol so
the harness (algos, trainer, logger) runs unchanged:

* **Level-Based Foraging** (``lbforaging``): agents on a grid must
  cooperate to collect food whose level can exceed any single agent's, so
  loading requires coordinated co-location.
* **Multi-Robot Warehouse** (``rware``): agents deliver requested shelves
  to workstations; sparse team reward, hard exploration.

Both register Gymnasium envs with a ``Tuple(Discrete, ...)`` action space
(one discrete head per agent) and a ``Tuple(Box, ...)`` observation space
(per-agent local obs). ``step`` returns ``(obs, reward_list, terminated,
truncated, info)`` with a per-agent reward list; we use the **sum** as the
cooperative team return (the EPyMARL convention).

**Coordination graph.** Neither benchmark exposes a graph. We use a
*complete* graph over the agents by default --- every agent attends to
every other, the maximally-informative case and therefore the GNN's
strongest setting; a graph-below-no-graph result here is conservative.
(``graph="ring"`` is also available.)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gnnmarl.envs.base import MultiAgentEnv, StepResult


class GymMARLAdapter(MultiAgentEnv):
    """Adapter from a Gymnasium Tuple-action multi-agent env to MultiAgentEnv.

    Args:
        env_id: registered Gymnasium id, e.g. ``"Foraging-10x10-3p-3f-v3"``
            or ``"rware-tiny-4ag-v2"``.
        graph: coordination-graph topology over agents
            (``"complete"`` default, or ``"ring"``).
        max_episode_steps: optional override of the env's TimeLimit.
    """

    def __init__(
        self,
        env_id: str,
        *,
        graph: str = "complete",
        max_episode_steps: int | None = None,
        seed: int | None = None,
    ):
        try:
            import gymnasium as gym
            import lbforaging  # noqa: F401  (registers Foraging-* ids)
            import rware  # noqa: F401       (registers rware-* ids)
        except ImportError as e:
            raise ImportError(
                "GymMARLAdapter requires 'lbforaging' and 'rware'; install with "
                "`pip install lbforaging rware`"
            ) from e

        self.env_id = env_id
        make_kwargs: dict[str, Any] = {}
        if max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = max_episode_steps
        self._env = gym.make(env_id, **make_kwargs)

        # Tuple(Discrete, ...) action space -> n_agents, n_actions.
        self.n_agents = len(self._env.action_space)
        self.n_actions = int(self._env.action_space[0].n)
        sample_obs_space = self._env.observation_space[0]
        self.obs_dim = int(np.prod(sample_obs_space.shape))
        self.state_dim = self.obs_dim * self.n_agents

        self._adj = self._build_adj(graph)
        self._episode_step = 0
        self._last_obs = None

    def _build_adj(self, graph: str) -> np.ndarray:
        n = self.n_agents
        a = np.zeros((n, n), dtype=np.float32)
        if graph == "complete":
            a[:] = 1.0
            np.fill_diagonal(a, 0.0)
        elif graph == "ring":
            for i in range(n):
                a[i, (i + 1) % n] = 1.0
                a[i, (i - 1) % n] = 1.0
            if n <= 2:  # ring degenerates; fall back to complete
                a[:] = 1.0
                np.fill_diagonal(a, 0.0)
        else:
            raise ValueError(f"unsupported graph {graph!r}")
        return a

    # ----------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None) -> StepResult:
        obs, _info = self._env.reset(seed=seed)
        self._last_obs = obs
        self._episode_step = 0
        st = self._stack(obs)
        return StepResult(obs=st, state=st.reshape(-1), reward=0.0, done=False,
                          info={"episode_step": 0})

    def step(self, actions: np.ndarray) -> StepResult:
        acts = tuple(int(a) for a in np.asarray(actions, dtype=np.int64).reshape(-1))
        obs, reward, terminated, truncated, _info = self._env.step(acts)
        self._last_obs = obs
        self._episode_step += 1
        # Cooperative team return = sum of per-agent rewards.
        reward_scalar = float(np.sum(np.asarray(reward, dtype=np.float64)))
        done = bool(np.any(terminated) or np.any(truncated))
        st = self._stack(obs)
        return StepResult(obs=st, state=st.reshape(-1), reward=reward_scalar,
                          done=done, info={"episode_step": self._episode_step})

    def adjacency(self) -> np.ndarray:
        return self._adj

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    # ------------------------------------------------------------ helpers
    def _stack(self, obs: Any) -> np.ndarray:
        return np.stack(
            [np.asarray(o, dtype=np.float32).reshape(-1) for o in obs], axis=0
        )
