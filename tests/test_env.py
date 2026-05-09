"""Env determinism + invariants."""
from __future__ import annotations

import numpy as np

from gnnmarl.envs.coord_grid import N_ACTIONS, CoordGridConfig, CoordGridEnv


def test_reset_deterministic_under_seed():
    cfg = CoordGridConfig(grid_size=5, n_agents=3, max_steps=20)
    e1 = CoordGridEnv(cfg, seed=42)
    e2 = CoordGridEnv(cfg, seed=42)
    o1, s1 = e1.reset(seed=7)
    o2, s2 = e2.reset(seed=7)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(s1, s2)


def test_obs_state_shapes():
    cfg = CoordGridConfig(grid_size=6, n_agents=4, max_steps=20)
    env = CoordGridEnv(cfg, seed=0)
    obs, state = env.reset(seed=0)
    assert obs.shape == (cfg.n_agents, env.obs_dim)
    assert state.shape == (env.state_dim,)
    assert env.obs_dim == 4 + 3 * (cfg.n_agents - 1)
    assert env.state_dim == 4 * cfg.n_agents


def test_step_returns_well_formed():
    cfg = CoordGridConfig(grid_size=6, n_agents=4, max_steps=10)
    env = CoordGridEnv(cfg, seed=0)
    env.reset(seed=0)
    actions = np.zeros(cfg.n_agents, dtype=np.int64)
    obs, state, r, done, info = env.step(actions)
    assert obs.shape == (cfg.n_agents, env.obs_dim)
    assert state.shape == (env.state_dim,)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert "success" in info
    assert "n_at_target" in info


def test_episode_terminates_at_max_steps():
    cfg = CoordGridConfig(grid_size=6, n_agents=4, max_steps=5)
    env = CoordGridEnv(cfg, seed=0)
    env.reset(seed=0)
    actions = np.zeros(cfg.n_agents, dtype=np.int64)  # all stay
    done = False
    n = 0
    while not done:
        _, _, _, done, _ = env.step(actions)
        n += 1
    assert n == 5  # exactly max_steps


def test_graph_full_has_n_minus_1_neighbors():
    cfg = CoordGridConfig(grid_size=5, n_agents=4, coordination_graph="full")
    env = CoordGridEnv(cfg, seed=0)
    adj = env.graph
    assert adj.shape == (4, 4)
    np.testing.assert_array_equal(np.diag(adj), np.zeros(4))
    # full graph: every off-diagonal entry is 1
    assert (adj + np.eye(4) > 0).all()


def test_graph_ring_diameter_scales():
    cfg = CoordGridConfig(grid_size=5, n_agents=6, coordination_graph="ring")
    env = CoordGridEnv(cfg, seed=0)
    # ring of 6 has diameter floor(6/2) = 3
    assert env.graph_diameter == 3


def test_n_actions_constant():
    assert N_ACTIONS == 5


def test_collision_penalty_applied():
    cfg = CoordGridConfig(
        grid_size=5,
        n_agents=2,
        max_steps=20,
        collision_penalty=-10.0,
        step_penalty=0.0,
        goal_bonus=0.0,
        terminal_bonus=0.0,
    )
    env = CoordGridEnv(cfg, seed=0)
    env.reset(seed=0)
    # Force both agents to (0, 0): clip + STAY can't guarantee that, so we
    # manually place them.
    env.positions[:] = np.array([[2, 2], [2, 2]], dtype=np.int32)
    _, _, r, _, info = env.step(np.array([0, 0]))  # STAY both
    assert info["n_collisions"] == 1
    assert r <= -10.0  # at least the collision penalty
