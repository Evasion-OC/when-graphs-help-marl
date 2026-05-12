"""PettingZoo MPE adapter.

Wraps `mpe2.simple_spread_v3` (and friends) into the same interface as
`CoordGridEnv` so the existing trainer drives it without changes.

What this exposes:
    obs_dim, state_dim, n_actions          (ints, attributes)
    graph                                  (N×N float32 adjacency)
    graph_diameter                         (int, or -1 if disconnected)
    reset(seed=None) -> (obs, state)
    step(actions: np.ndarray) -> (obs, state, reward, done, info)

Notes
-----
- MPE returns per-agent observations (dict[agent_name, np.ndarray]). The
  adapter stacks them into a (N, obs_dim) array, matching CoordGridEnv.
- MPE returns per-agent rewards. simple_spread is a cooperative shared
  reward, so all agents receive the same scalar; the adapter takes the
  first agent's reward as the team reward.
- Episodes end when `max_cycles` is reached. simple_spread has no
  early-termination signal, so `done = any(truncation)` works.
- "Global state" is built by concatenating all per-agent observations.
  For simple_spread this triple-counts shared landmark info — fine for
  centralised-training value-decomposition algorithms (the Q-net only
  uses the per-agent obs; the mixer uses this global state).
- Coordination graph defaults to `full` (every agent connected to every
  other). Phase 4's ablation will let it vary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from mpe2 import simple_spread_v3
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mpe2 is required for MPEEnv. Install with `pip install mpe2`."
    ) from e


@dataclass
class MPEConfig:
    """Phase 3 supports `simple_spread` only.

    Other MPE envs (`simple_tag`, `simple_world_comm`) have heterogeneous
    observation spaces and adversarial agents that don't fit the
    homogeneous cooperative-MARL setup we use for the algorithm comparison.
    Phase 3's external validity story is "GNN-MARL findings transfer to a
    canonical cooperative MARL benchmark" — simple_spread is sufficient.
    """

    name: str = "simple_spread"
    n_agents: int = 3
    max_cycles: int = 25
    coordination_graph: str = "full"  # 'full', 'ring', 'lattice'
    seed: int = 0


class MPEEnv:
    """Trainer-facing wrapper around an MPE parallel env."""

    def __init__(self, cfg: MPEConfig):
        self.cfg = cfg
        if cfg.name != "simple_spread":
            raise ValueError(
                f"Phase 3 supports name='simple_spread' only; got {cfg.name!r}."
            )
        self._raw = simple_spread_v3.parallel_env(
            N=cfg.n_agents, max_cycles=cfg.max_cycles, continuous_actions=False
        )

        # Probe the env to get shapes
        self._raw.reset(seed=cfg.seed)
        self._agent_names = list(self._raw.agents)
        if len(self._agent_names) != cfg.n_agents and cfg.name == "simple_spread":
            raise RuntimeError(
                f"Expected {cfg.n_agents} agents, got {len(self._agent_names)}"
            )
        per_agent_obs_dim = self._raw.observation_space(self._agent_names[0]).shape[0]
        self.n_agents = len(self._agent_names)
        self.obs_dim = per_agent_obs_dim
        self.state_dim = per_agent_obs_dim * self.n_agents
        self.n_actions = int(self._raw.action_space(self._agent_names[0]).n)
        self._build_graph()
        self._terminated = False

    # ---- Graph (coordination topology, separate from physics) -------------
    def _build_graph(self) -> None:
        N = self.n_agents
        if self.cfg.coordination_graph == "full":
            adj = np.ones((N, N), dtype=np.float32) - np.eye(N, dtype=np.float32)
        elif self.cfg.coordination_graph == "ring":
            adj = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                adj[i, (i + 1) % N] = 1.0
                adj[i, (i - 1) % N] = 1.0
        elif self.cfg.coordination_graph == "lattice":
            adj = np.zeros((N, N), dtype=np.float32)
            for i in range(N - 1):
                adj[i, i + 1] = 1.0
                adj[i + 1, i] = 1.0
        else:
            raise ValueError(
                f"Unsupported coordination_graph for MPE: {self.cfg.coordination_graph}"
            )
        self.adj = adj

    @property
    def graph(self) -> np.ndarray:
        return self.adj.copy()

    @property
    def graph_diameter(self) -> int:
        N = self.n_agents
        adj_bool = self.adj > 0
        diam = 0
        for src in range(N):
            dist = np.full(N, -1, dtype=int)
            dist[src] = 0
            frontier = [src]
            while frontier:
                nxt = []
                for u in frontier:
                    for v in range(N):
                        if adj_bool[u, v] and dist[v] == -1:
                            dist[v] = dist[u] + 1
                            nxt.append(v)
                frontier = nxt
            if (dist == -1).any():
                return -1
            diam = max(diam, int(dist.max()))
        return diam

    # ---- Trainer API ------------------------------------------------------
    def _stack_obs(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([obs_dict[a] for a in self._agent_names]).astype(np.float32)

    def _make_state(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [obs_dict[a].astype(np.float32) for a in self._agent_names]
        )

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        obs_dict, _info = self._raw.reset(seed=seed if seed is not None else self.cfg.seed)
        self._terminated = False
        # On reset, env.agents may shrink later when an agent's `terminated`
        # is True; we cache the order for action dispatch on this episode.
        self._agent_names = list(self._raw.agents)
        return self._stack_obs(obs_dict), self._make_state(obs_dict)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        if actions.shape != (self.n_agents,):
            raise ValueError(f"actions shape {actions.shape} != ({self.n_agents},)")
        action_dict = {a: int(actions[i]) for i, a in enumerate(self._agent_names)}
        obs_dict, rew_dict, term_dict, trunc_dict, info = self._raw.step(action_dict)
        # cooperative shared reward → take any agent's value
        first = self._agent_names[0]
        reward = float(rew_dict.get(first, 0.0))
        any_term = any(term_dict.values()) if term_dict else False
        any_trunc = any(trunc_dict.values()) if trunc_dict else False
        done = bool(any_term or any_trunc)
        # Some envs drop agents from `agents` after termination; we pad
        # missing observations with zeros to maintain (N, obs_dim).
        if not obs_dict:
            obs_array = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
            state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            full_obs = {
                a: obs_dict.get(a, np.zeros(self.obs_dim, dtype=np.float32))
                for a in self._agent_names
            }
            obs_array = self._stack_obs(full_obs)
            state = self._make_state(full_obs)
        return obs_array, state, reward, done, {"shared_reward": reward, **info}
