"""Cooperative grid coordination env.

N agents and N targets on an LxL grid. Each agent is assigned a unique target.
Agents move simultaneously each step (one of: stay, up, down, left, right).

Reward design (shared / cooperative):
- step penalty: -0.01 each timestep
- collision penalty: -0.05 per overlapping pair (encourages avoidance)
- per-agent goal bonus: +0.5 the first time agent i lands on its target
- terminal bonus: +1.0 when all agents are simultaneously on their targets

Episode ends when all agents are on targets (success) or `max_steps` is hit.

Why this design:
- Sparse-but-shaped reward → discriminates IQL (no coord) from VDN/QMIX/GNN-QMIX
- Targets are shuffled per agent → trivial greedy doesn't work
- Tunable via `coordination_graph` which limits which agents observe each other
- Deterministic given seed → seed-controlled reproducibility
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Action constants
STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
N_ACTIONS = 5
_DIRS = np.array(
    [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32
)  # row, col deltas


@dataclass
class CoordGridConfig:
    grid_size: int = 6
    n_agents: int = 4
    max_steps: int = 40
    # 'full' (everyone sees everyone), 'ring' (cycle), 'knn' (k-nearest by L2),
    # 'random' (Erdos-Renyi at p=edge_p), or 'lattice' (grid-spatial).
    coordination_graph: str = "full"
    edge_p: float = 0.5  # only used for 'random'
    knn_k: int = 2  # only used for 'knn'
    obs_radius: Optional[int] = None  # if set, agents see only within radius
    step_penalty: float = -0.01
    collision_penalty: float = -0.05
    goal_bonus: float = 0.5
    terminal_bonus: float = 1.0


class CoordGridEnv:
    """Vectorized cooperative gridworld for MARL.

    Returns numpy arrays. Trainer is responsible for tensor conversion.

    Observation per agent (egocentric): [row/L, col/L, target_row/L, target_col/L]
        + concatenated relative positions of graph neighbors (zero-padded to N).
    Each neighbor block: [dr/L, dc/L, has_neighbor_flag].
    """

    def __init__(self, cfg: CoordGridConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._build_graph()

        # observation dim: 4 (self) + 3 * (N-1) (neighbor block, padded)
        self.obs_dim = 4 + 3 * (cfg.n_agents - 1)
        # global state for centralized training: all agent positions + all targets
        self.state_dim = 4 * cfg.n_agents
        self.n_actions = N_ACTIONS

        self.positions: np.ndarray  # (N, 2) int
        self.targets: np.ndarray  # (N, 2) int
        self.reached: np.ndarray  # (N,) bool — has agent been to its target this episode
        self.t: int = 0
        self._first_reset = True

    # ---- Graph construction -------------------------------------------------
    def _build_graph(self) -> None:
        N = self.cfg.n_agents
        if self.cfg.coordination_graph == "full":
            adj = np.ones((N, N), dtype=np.float32) - np.eye(N, dtype=np.float32)
        elif self.cfg.coordination_graph == "ring":
            adj = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                adj[i, (i + 1) % N] = 1.0
                adj[i, (i - 1) % N] = 1.0
        elif self.cfg.coordination_graph == "random":
            adj = (self.rng.random((N, N)) < self.cfg.edge_p).astype(np.float32)
            adj = np.maximum(adj, adj.T)
            np.fill_diagonal(adj, 0)
        elif self.cfg.coordination_graph == "knn":
            # graph rebuilt every reset based on positions; placeholder here
            adj = np.eye(N, dtype=np.float32) - np.eye(N, dtype=np.float32)
        elif self.cfg.coordination_graph == "lattice":
            # 1D lattice: each agent connected to its index neighbors (chain)
            adj = np.zeros((N, N), dtype=np.float32)
            for i in range(N - 1):
                adj[i, i + 1] = 1.0
                adj[i + 1, i] = 1.0
        else:
            raise ValueError(f"Unknown coordination_graph: {self.cfg.coordination_graph}")
        self.adj = adj

    def _maybe_rebuild_knn(self) -> None:
        if self.cfg.coordination_graph != "knn":
            return
        N = self.cfg.n_agents
        d = np.linalg.norm(
            self.positions[:, None, :] - self.positions[None, :, :], axis=-1
        )
        np.fill_diagonal(d, np.inf)
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            nn = np.argsort(d[i])[: self.cfg.knn_k]
            adj[i, nn] = 1.0
        adj = np.maximum(adj, adj.T)
        self.adj = adj

    # ---- Episode lifecycle --------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        N = self.cfg.n_agents
        L = self.cfg.grid_size
        all_cells = np.array([(r, c) for r in range(L) for c in range(L)])
        # need 2N distinct cells (positions + targets)
        if 2 * N > all_cells.shape[0]:
            raise ValueError("Grid too small: need at least 2N cells.")
        chosen = self.rng.choice(all_cells.shape[0], size=2 * N, replace=False)
        self.positions = all_cells[chosen[:N]].astype(np.int32)
        self.targets = all_cells[chosen[N:]].astype(np.int32)
        self.reached = np.zeros(N, dtype=bool)
        self.t = 0
        self._maybe_rebuild_knn()
        return self._make_obs(), self._make_state()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        cfg = self.cfg
        L = cfg.grid_size
        actions = np.asarray(actions, dtype=np.int32)
        if actions.shape != (cfg.n_agents,):
            raise ValueError(f"actions shape {actions.shape} != ({cfg.n_agents},)")

        # propose moves
        proposed = self.positions + _DIRS[actions]
        proposed = np.clip(proposed, 0, L - 1)
        # commit moves (overlapping is allowed; collision penalty applied below)
        self.positions = proposed.astype(np.int32)
        self._maybe_rebuild_knn()

        reward = float(cfg.step_penalty)

        # goal bonus on first arrival
        on_target = np.all(self.positions == self.targets, axis=1)
        new_reach = on_target & (~self.reached)
        reward += cfg.goal_bonus * int(new_reach.sum())
        self.reached |= on_target  # sticky once reached

        # collision penalty (count distinct overlapping pairs)
        pos_view = self.positions.view([("r", np.int32), ("c", np.int32)]).reshape(-1)
        _, counts = np.unique(pos_view, return_counts=True)
        n_overlapping_pairs = int(((counts * (counts - 1)) // 2).sum())
        reward += cfg.collision_penalty * n_overlapping_pairs

        self.t += 1
        success = bool(on_target.all())
        if success:
            reward += cfg.terminal_bonus
        done = success or (self.t >= cfg.max_steps)

        info = {
            "success": success,
            "n_at_target": int(on_target.sum()),
            "n_collisions": n_overlapping_pairs,
            "t": self.t,
        }
        return self._make_obs(), self._make_state(), reward, done, info

    # ---- Observations -------------------------------------------------------
    def _make_obs(self) -> np.ndarray:
        N = self.cfg.n_agents
        L = self.cfg.grid_size
        obs = np.zeros((N, self.obs_dim), dtype=np.float32)
        for i in range(N):
            obs[i, 0] = self.positions[i, 0] / max(L - 1, 1)
            obs[i, 1] = self.positions[i, 1] / max(L - 1, 1)
            obs[i, 2] = self.targets[i, 0] / max(L - 1, 1)
            obs[i, 3] = self.targets[i, 1] / max(L - 1, 1)
            # neighbor block (j != i, ordered by index, only graph-neighbors)
            block = 4
            for j in range(N):
                if j == i:
                    continue
                if self.adj[i, j] > 0 and (
                    self.cfg.obs_radius is None
                    or np.abs(self.positions[i] - self.positions[j]).sum()
                    <= self.cfg.obs_radius
                ):
                    obs[i, block + 0] = (
                        self.positions[j, 0] - self.positions[i, 0]
                    ) / max(L - 1, 1)
                    obs[i, block + 1] = (
                        self.positions[j, 1] - self.positions[i, 1]
                    ) / max(L - 1, 1)
                    obs[i, block + 2] = 1.0
                # else leave as zeros (no neighbor flag = 0)
                block += 3
        return obs

    def _make_state(self) -> np.ndarray:
        L = self.cfg.grid_size
        s = np.empty(self.state_dim, dtype=np.float32)
        for i in range(self.cfg.n_agents):
            s[4 * i + 0] = self.positions[i, 0] / max(L - 1, 1)
            s[4 * i + 1] = self.positions[i, 1] / max(L - 1, 1)
            s[4 * i + 2] = self.targets[i, 0] / max(L - 1, 1)
            s[4 * i + 3] = self.targets[i, 1] / max(L - 1, 1)
        return s

    @property
    def graph(self) -> np.ndarray:
        return self.adj.copy()

    @property
    def graph_diameter(self) -> int:
        """Shortest-path diameter of `self.adj` (∞ if disconnected, returns -1)."""
        N = self.cfg.n_agents
        adj_bool = self.adj > 0
        # BFS from each node
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
