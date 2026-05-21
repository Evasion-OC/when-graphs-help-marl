"""PettingZoo MPE -> MultiAgentEnv adapter (Phase 3 scaffold).

Wraps ``pettingzoo.mpe.simple_spread_v3`` (and ``simple_tag_v3``) in our
``MultiAgentEnv`` protocol so the rest of the harness (algos, trainer,
logger) works unchanged on canonical MPE benchmarks.

This module imports ``pettingzoo`` lazily — it is in the ``[mpe]`` extra,
not in the core dependencies, so Phase 0–2 unit tests do not break when
PettingZoo is absent.

**Design call on the coordination graph for MPE.** MPE does not expose an
explicit graph. We expose a *k-NN* graph over current agent positions
(``k`` configurable, default 2), re-computed at each ``reset``. This is the
same convention used by DGN \citep{jiang_2020_dgn}. For ``simple_tag``,
the predator-prey distinction is encoded only through observations; the
graph is built over all agents in the role under control.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gnnmarl.envs.base import MultiAgentEnv, StepResult


class MPEEnvAdapter(MultiAgentEnv):
    """Adapter from PettingZoo MPE parallel envs to MultiAgentEnv.

    Currently supports:

    * ``simple_spread_v3`` (fully cooperative)
    * ``simple_tag_v3``    (mixed; predators-only training is the typical
      cooperative slice).

    Constructor kwargs:
        env_name: ``"simple_spread"`` or ``"simple_tag"``.
        n_agents: number of cooperative agents (predators for tag).
        k_neighbors: k for the kNN coordination graph.
        max_cycles: episode length forwarded to the underlying env.
        local_ratio: ``simple_spread`` reward shaping; default 0.5.
        seed: forwarded to the underlying env on reset.
    """

    def __init__(
        self,
        env_name: str = "simple_spread",
        n_agents: int = 3,
        *,
        k_neighbors: int = 2,
        max_cycles: int = 25,
        local_ratio: float = 0.5,
        continuous_actions: bool = False,
        seed: int | None = None,
    ):
        try:
            from pettingzoo.mpe import simple_spread_v3, simple_tag_v3
        except ImportError as e:
            raise ImportError(
                "MPEEnvAdapter requires the 'mpe' extra; install with "
                "`pip install -e .[mpe]`"
            ) from e

        self.env_name = env_name
        self.n_agents = n_agents
        self.k_neighbors = k_neighbors

        if env_name == "simple_spread":
            self._env = simple_spread_v3.parallel_env(
                N=n_agents,
                max_cycles=max_cycles,
                local_ratio=local_ratio,
                continuous_actions=continuous_actions,
            )
        elif env_name == "simple_tag":
            # n_agents counts cooperative predators; default prey = 1.
            self._env = simple_tag_v3.parallel_env(
                num_good=1, num_adversaries=n_agents, num_obstacles=2,
                max_cycles=max_cycles,
                continuous_actions=continuous_actions,
            )
        else:
            raise ValueError(f"unsupported env_name {env_name!r}")

        # We need a peek to infer obs / action dims.
        obs_dict, _ = self._env.reset(seed=seed)
        self._agents = list(self._env.agents)
        sample_obs = next(iter(obs_dict.values()))
        self.obs_dim = int(np.prod(sample_obs.shape))
        self.state_dim = self.obs_dim * len(self._agents)
        # Discrete action space (5 for spread/tag with discrete actions).
        action_space = self._env.action_space(self._agents[0])
        self.n_actions = int(getattr(action_space, "n", 5))

        # Position-extraction helper depends on MPE's obs convention.
        # simple_spread obs = [self_vel(2), self_pos(2), landmark_relpos(...),
        # other_agent_relpos(...), comms(...)]. Self position is dims [2:4].
        # We use these for the kNN graph.
        self._adj = np.eye(len(self._agents), dtype=np.float32)
        self._last_obs = obs_dict

    # ----------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None) -> StepResult:
        obs_dict, _ = self._env.reset(seed=seed)
        self._agents = list(self._env.agents)
        self._last_obs = obs_dict
        self._rebuild_adj(obs_dict)
        return StepResult(
            obs=self._stack(obs_dict),
            state=self._stack(obs_dict).reshape(-1),
            reward=0.0, done=False, info={"episode_step": 0},
        )

    def step(self, actions: np.ndarray) -> StepResult:
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if len(actions) != len(self._agents):
            raise ValueError(
                f"expected {len(self._agents)} actions; got {len(actions)}"
            )
        action_dict = {a: int(actions[i]) for i, a in enumerate(self._agents)}
        next_obs_dict, rewards_dict, terms_dict, truncs_dict, _infos = \
            self._env.step(action_dict)

        # Cooperative reward = mean over cooperative agents (spread is fully
        # cooperative so all rewards are equal; tag predators share team
        # reward).
        if rewards_dict:
            reward = float(np.mean(list(rewards_dict.values())))
        else:
            reward = 0.0
        done = bool(any(terms_dict.values()) or any(truncs_dict.values()))

        # The env truncates the agent list when an agent terminates; for the
        # cooperative case we keep the previous obs to preserve shape.
        if next_obs_dict:
            self._last_obs = next_obs_dict
        return StepResult(
            obs=self._stack(self._last_obs),
            state=self._stack(self._last_obs).reshape(-1),
            reward=reward, done=done,
            info={"episode_step": self._env.steps if hasattr(self._env, "steps") else -1},
        )

    def adjacency(self) -> np.ndarray:
        return self._adj

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    # ------------------------------------------------------------ helpers
    def _stack(self, obs_dict: dict[str, Any]) -> np.ndarray:
        # Stack into [N, obs_dim] in the canonical agent order.
        return np.stack(
            [np.asarray(obs_dict[a], dtype=np.float32).reshape(-1)
             for a in self._agents],
            axis=0,
        )

    def _rebuild_adj(self, obs_dict: dict[str, Any]) -> None:
        # Extract self position dims [2:4] per simple_spread convention.
        # For simple_tag predators the convention is similar.
        try:
            positions = np.stack(
                [np.asarray(obs_dict[a], dtype=np.float32)[2:4]
                 for a in self._agents],
                axis=0,
            )
        except Exception:
            self._adj = np.eye(len(self._agents), dtype=np.float32)
            return

        n = len(self._agents)
        # kNN graph (symmetric: an edge if either endpoint picks the other).
        dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        nn = np.argsort(dists, axis=1)[:, : self.k_neighbors]
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in nn[i]:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        np.fill_diagonal(adj, 0.0)
        self._adj = adj
