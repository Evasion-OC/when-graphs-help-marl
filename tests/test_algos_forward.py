"""Forward-pass and action-selection tests for the Phase-0 algorithms.

These tests verify the structural contract from ``docs/INTERFACES.md``:
* ``act()`` returns ``int64`` actions of the right shape and range.
* ``epsilon=1.0`` action selection is RNG-determined (no network argmax).

They run on CPU and are deterministic given a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gnnmarl.algos import GNNQMIX, IQL, QMIX, VDN, make_algo

ALGO_CLASSES = [IQL, VDN, QMIX, GNNQMIX]
ALGO_NAMES = ["iql", "vdn", "qmix", "gnn_qmix"]

N_AGENTS = 4
OBS_DIM = 11
STATE_DIM = 8
N_ACTIONS = 5


def _adj() -> np.ndarray:
    # Identity + 1-step ring shift -> each agent connected to the next one.
    return (np.eye(N_AGENTS) + np.roll(np.eye(N_AGENTS), 1, axis=0)).astype(np.float32)


def _make(cls, seed: int = 0):
    return cls(
        n_agents=N_AGENTS,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        device=torch.device("cpu"),
        seed=seed,
    )


@pytest.mark.parametrize("cls", ALGO_CLASSES, ids=ALGO_NAMES)
def test_algo_construct(cls):
    algo = _make(cls)
    assert algo.n_agents == N_AGENTS
    assert algo.obs_dim == OBS_DIM
    assert algo.state_dim == STATE_DIM
    assert algo.n_actions == N_ACTIONS
    assert algo.name in {"iql", "vdn", "qmix", "gnn_qmix"}


@pytest.mark.parametrize("cls", ALGO_CLASSES, ids=ALGO_NAMES)
def test_algo_act_shape(cls):
    algo = _make(cls)
    rng = np.random.default_rng(0)
    obs = np.zeros((N_AGENTS, OBS_DIM), dtype=np.float32)
    adj = _adj()
    actions = algo.act(obs, adj, epsilon=0.0, rng=rng)
    assert isinstance(actions, np.ndarray)
    assert actions.dtype == np.int64
    assert actions.shape == (N_AGENTS,)
    assert int(actions.min()) >= 0
    assert int(actions.max()) < N_ACTIONS


@pytest.mark.parametrize("cls", ALGO_CLASSES, ids=ALGO_NAMES)
def test_algo_eps_greedy(cls):
    """With epsilon=1.0 every action must come from the supplied RNG.

    We compare against the RNG sequence we would have drawn directly: the
    contract is that with epsilon=1.0 the network is bypassed entirely.
    """
    algo = _make(cls)
    obs = np.zeros((N_AGENTS, OBS_DIM), dtype=np.float32)
    adj = _adj()

    # Two passes with epsilon=1.0 must equal what we'd draw directly from
    # an RNG of the same seed using the same (Bernoulli mask, action draw)
    # pattern as BaseAlgo.act.
    rng = np.random.default_rng(42)
    actions_1 = algo.act(obs, adj, epsilon=1.0, rng=rng)
    actions_2 = algo.act(obs, adj, epsilon=1.0, rng=rng)

    expected_rng = np.random.default_rng(42)
    # First call's draws
    mask_1 = expected_rng.random(N_AGENTS) < 1.0
    rand_1 = expected_rng.integers(low=0, high=N_ACTIONS, size=N_AGENTS)
    # Second call's draws
    mask_2 = expected_rng.random(N_AGENTS) < 1.0
    rand_2 = expected_rng.integers(low=0, high=N_ACTIONS, size=N_AGENTS)

    assert mask_1.all() and mask_2.all()
    assert np.array_equal(actions_1, rand_1.astype(np.int64))
    assert np.array_equal(actions_2, rand_2.astype(np.int64))


@pytest.mark.parametrize("name", ALGO_NAMES)
def test_make_algo_factory(name):
    algo = make_algo(
        name,
        n_agents=N_AGENTS,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        device=torch.device("cpu"),
        seed=0,
    )
    assert algo.name == name
