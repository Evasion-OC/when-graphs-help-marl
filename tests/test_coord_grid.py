"""Unit tests for ``gnnmarl.envs.coord_grid.CoordGrid``."""

from __future__ import annotations

import numpy as np
import pytest

from gnnmarl.envs import CoordGrid, make_env
from gnnmarl.envs.base import MultiAgentEnv, StepResult


GRAPH_CASES = [
    ("ring", {"n_agents": 6}),
    ("line", {"n_agents": 6}),
    ("complete", {"n_agents": 6}),
    ("erdos_renyi", {"n_agents": 6, "er_prob": 0.5}),
    ("grid2d", {"n_agents": 9}),
]


@pytest.mark.parametrize("graph,kwargs", GRAPH_CASES)
def test_shapes(graph: str, kwargs: dict) -> None:
    env = CoordGrid(graph=graph, seed=0, **kwargs)
    assert isinstance(env, MultiAgentEnv)
    n = env.n_agents

    out = env.reset(seed=0)
    assert isinstance(out, StepResult)
    assert out.obs.shape == (n, env.obs_dim)
    assert out.obs.dtype == np.float32
    assert out.state.shape == (env.state_dim,)
    assert out.state.dtype == np.float32
    assert out.reward == 0.0
    assert out.done is False
    assert out.info["episode_step"] == 0

    adj = env.adjacency()
    assert adj.shape == (n, n)
    assert adj.dtype == np.float32

    # A step preserves shapes.
    rng = np.random.default_rng(123)
    actions = rng.integers(0, env.n_actions, size=n)
    nxt = env.step(actions)
    assert nxt.obs.shape == (n, env.obs_dim)
    assert nxt.state.shape == (env.state_dim,)


def test_determinism() -> None:
    env_a = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=42)
    env_b = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=42)
    a0 = env_a.reset(seed=42)
    b0 = env_b.reset(seed=42)
    np.testing.assert_array_equal(a0.obs, b0.obs)
    np.testing.assert_array_equal(a0.state, b0.state)
    assert a0.reward == b0.reward
    assert a0.done == b0.done

    action_rng = np.random.default_rng(2024)
    for _ in range(10):
        actions = action_rng.integers(0, env_a.n_actions, size=env_a.n_agents)
        ra = env_a.step(actions)
        rb = env_b.step(actions)
        np.testing.assert_array_equal(ra.obs, rb.obs)
        np.testing.assert_array_equal(ra.state, rb.state)
        assert ra.reward == rb.reward
        assert ra.done == rb.done


@pytest.mark.parametrize("graph,kwargs", GRAPH_CASES)
def test_adjacency_properties(graph: str, kwargs: dict) -> None:
    env = CoordGrid(graph=graph, seed=7, **kwargs)
    env.reset(seed=7)
    adj = env.adjacency()
    # Symmetric.
    np.testing.assert_array_equal(adj, adj.T)
    # Zero diagonal.
    np.testing.assert_array_equal(np.diag(adj), np.zeros(env.n_agents, dtype=adj.dtype))
    # 0/1 valued.
    unique = np.unique(adj)
    assert set(unique.tolist()).issubset({0.0, 1.0})


def test_ring_topology() -> None:
    env = CoordGrid(n_agents=6, graph="ring", seed=0)
    env.reset(seed=0)
    adj = env.adjacency()
    # Six edges → upper-triangular sum is 6.
    assert int(np.triu(adj, k=1).sum()) == 6
    # Every node has degree 2.
    np.testing.assert_array_equal(adj.sum(axis=1), np.full(6, 2.0, dtype=np.float32))


def test_reward_when_colocated() -> None:
    env = CoordGrid(n_agents=2, grid_size=5, graph="ring", seed=0)
    env.reset(seed=0)
    # Force colocation, then take stay actions so they remain colocated.
    env._place_agents(np.array([[2, 2], [2, 2]]))
    out = env.step(np.zeros(2, dtype=np.int64))  # 0 = stay
    assert out.reward > 0.0
    assert out.reward == pytest.approx(1.0)


def test_reward_zero_when_apart() -> None:
    env = CoordGrid(n_agents=2, grid_size=5, graph="ring", seed=0)
    env.reset(seed=0)
    env._place_agents(np.array([[0, 0], [3, 3]]))
    out = env.step(np.zeros(2, dtype=np.int64))  # both stay
    assert out.reward == 0.0


def test_episode_termination() -> None:
    steps = 5
    env = CoordGrid(n_agents=3, grid_size=4, episode_steps=steps, graph="ring", seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    dones = []
    for _ in range(steps):
        actions = rng.integers(0, env.n_actions, size=env.n_agents)
        out = env.step(actions)
        dones.append(out.done)
    assert dones[:-1] == [False] * (steps - 1)
    assert dones[-1] is True


def test_grid2d_requires_square() -> None:
    with pytest.raises(ValueError):
        CoordGrid(n_agents=5, graph="grid2d")


def test_make_env_factory() -> None:
    env = make_env("coord_grid", n_agents=4, graph="ring", seed=0)
    assert isinstance(env, CoordGrid)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        make_env("unknown_env")
