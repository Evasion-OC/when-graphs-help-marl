"""MPE adapter tests. Skipped when mpe2 is not installed."""
from __future__ import annotations

import numpy as np
import pytest

mpe2 = pytest.importorskip("mpe2")  # noqa: F841

from gnnmarl.envs.mpe_env import MPEConfig, MPEEnv  # noqa: E402


def test_reset_obs_state_shapes_simple_spread():
    env = MPEEnv(MPEConfig(name="simple_spread", n_agents=3, max_cycles=10))
    obs, state = env.reset(seed=0)
    assert obs.shape == (3, env.obs_dim)
    assert state.shape == (env.state_dim,)
    assert env.state_dim == 3 * env.obs_dim
    assert env.n_actions == 5


def test_step_returns_well_formed():
    env = MPEEnv(MPEConfig(name="simple_spread", n_agents=3, max_cycles=10))
    env.reset(seed=0)
    actions = np.zeros(env.n_agents, dtype=np.int64)
    obs, state, r, done, info = env.step(actions)
    assert obs.shape == (env.n_agents, env.obs_dim)
    assert state.shape == (env.state_dim,)
    assert isinstance(r, float)
    assert isinstance(done, bool)


def test_seeded_reset_is_deterministic():
    e1 = MPEEnv(MPEConfig(name="simple_spread", n_agents=3, max_cycles=10))
    e2 = MPEEnv(MPEConfig(name="simple_spread", n_agents=3, max_cycles=10))
    o1, s1 = e1.reset(seed=42)
    o2, s2 = e2.reset(seed=42)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(s1, s2)


def test_episode_terminates_at_max_cycles():
    env = MPEEnv(MPEConfig(name="simple_spread", n_agents=3, max_cycles=5))
    env.reset(seed=0)
    actions = np.zeros(env.n_agents, dtype=np.int64)
    n = 0
    done = False
    while not done and n < 50:
        _, _, _, done, _ = env.step(actions)
        n += 1
    assert n == 5  # exactly max_cycles


def test_graph_topology_options():
    env_full = MPEEnv(MPEConfig(name="simple_spread", n_agents=4, coordination_graph="full"))
    assert env_full.graph.shape == (4, 4)
    assert (env_full.graph + np.eye(4) > 0).all()
    env_ring = MPEEnv(MPEConfig(name="simple_spread", n_agents=6, coordination_graph="ring"))
    assert env_ring.graph_diameter == 3  # ring of 6


def test_unsupported_name_raises():
    with pytest.raises(ValueError, match="simple_spread"):
        MPEEnv(MPEConfig(name="simple_tag", n_agents=3))
