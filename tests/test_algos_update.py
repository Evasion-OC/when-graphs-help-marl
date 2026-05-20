"""Gradient-step tests for the Phase-0 algorithms.

These tests run the full TD pipeline through each algorithm to confirm:
* :meth:`update` returns a sane metrics dict once the buffer is large enough.
* The target net is hard-synced every ``target_update_interval`` updates.
* GNN-QMIX actually uses the adjacency (changing the graph changes the
  resulting per-agent Q-values).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from gnnmarl.algos import GNNQMIX, IQL, QMIX, VDN
from gnnmarl.utils.replay import Transition

ALGO_CLASSES = [IQL, VDN, QMIX, GNNQMIX]
ALGO_NAMES = ["iql", "vdn", "qmix", "gnn_qmix"]

N_AGENTS = 4
OBS_DIM = 11
STATE_DIM = 8
N_ACTIONS = 5


def _make(cls, *, batch_size: int = 4, target_update_interval: int = 200, seed: int = 0):
    return cls(
        n_agents=N_AGENTS,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        batch_size=batch_size,
        target_update_interval=target_update_interval,
        device=torch.device("cpu"),
        seed=seed,
    )


def _fake_transition(rng: np.random.Generator) -> Transition:
    obs = rng.normal(size=(N_AGENTS, OBS_DIM)).astype(np.float32)
    next_obs = rng.normal(size=(N_AGENTS, OBS_DIM)).astype(np.float32)
    state = rng.normal(size=(STATE_DIM,)).astype(np.float32)
    next_state = rng.normal(size=(STATE_DIM,)).astype(np.float32)
    actions = rng.integers(low=0, high=N_ACTIONS, size=N_AGENTS).astype(np.int64)
    reward = float(rng.normal())
    # Simple ring graph; symmetric and self-loop-free.
    adj = np.eye(N_AGENTS, dtype=np.float32)
    adj = ((np.roll(adj, 1, axis=0) + np.roll(adj, -1, axis=0)) > 0).astype(np.float32)
    return Transition(
        obs=obs,
        state=state,
        actions=actions,
        reward=reward,
        next_obs=next_obs,
        next_state=next_state,
        done=False,
        adj=adj,
    )


def _flat_params(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


@pytest.mark.parametrize("cls", ALGO_CLASSES, ids=ALGO_NAMES)
def test_algo_update_runs(cls):
    algo = _make(cls, batch_size=4)
    rng = np.random.default_rng(0)
    for _ in range(10):
        algo.observe(_fake_transition(rng))

    saw_metrics = None
    for _ in range(5):
        metrics = algo.update()
        if metrics is not None:
            saw_metrics = metrics
            break

    assert saw_metrics is not None, "update() never produced a metrics dict"
    assert "loss" in saw_metrics
    assert "grad_norm" in saw_metrics
    assert math.isfinite(saw_metrics["loss"])
    assert math.isfinite(saw_metrics["grad_norm"])


@pytest.mark.parametrize("cls", ALGO_CLASSES, ids=ALGO_NAMES)
def test_algo_target_update(cls):
    target_update_interval = 3
    algo = _make(cls, batch_size=4, target_update_interval=target_update_interval)

    rng = np.random.default_rng(1)
    for _ in range(10):
        algo.observe(_fake_transition(rng))

    # Run exactly target_update_interval successful updates.
    successful = 0
    safety = 0
    while successful < target_update_interval:
        if algo.update() is not None:
            successful += 1
        safety += 1
        if safety > 100:
            raise AssertionError("update() failed to produce metrics in 100 calls")

    online_flat = _flat_params(algo)
    target_flat = _flat_params(algo._target)
    assert online_flat.shape == target_flat.shape
    assert torch.allclose(online_flat, target_flat, atol=0.0, rtol=0.0)


def test_gnn_qmix_uses_adj():
    algo = _make(GNNQMIX, batch_size=4)
    b = 2
    obs = torch.randn(b, N_AGENTS, OBS_DIM, generator=torch.Generator().manual_seed(0))
    adj_eye = torch.eye(N_AGENTS).unsqueeze(0).expand(b, N_AGENTS, N_AGENTS).contiguous()
    # Fully connected, no self-loops (the GCN re-adds I internally).
    adj_full = (torch.ones(N_AGENTS, N_AGENTS) - torch.eye(N_AGENTS))
    adj_full = adj_full.unsqueeze(0).expand(b, N_AGENTS, N_AGENTS).contiguous()

    with torch.no_grad():
        q_eye = algo._q_values(obs, adj_eye)
        q_full = algo._q_values(obs, adj_full)

    assert q_eye.shape == q_full.shape == (b, N_AGENTS, N_ACTIONS)
    # The whole point of GNN-QMIX: changing the graph changes the per-agent Q's.
    assert not torch.allclose(q_eye, q_full, atol=1e-5), (
        "GNN-QMIX produced identical Q-values for two very different adjacency matrices; "
        "the GCN is not actually mixing across the graph."
    )
