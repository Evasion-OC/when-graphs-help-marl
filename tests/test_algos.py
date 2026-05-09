"""Algorithm forward-pass + learn-step shape checks.

We don't test learning (that's the smoke test's job), only that each algo
runs end-to-end on a single batch without dim errors.
"""
from __future__ import annotations

import numpy as np
import torch

from gnnmarl.algos.base import AlgoConfig
from gnnmarl.algos.gnn_qmix import GNNQMIX
from gnnmarl.algos.iql import IQL
from gnnmarl.algos.qmix import QMIX
from gnnmarl.algos.vdn import VDN
from gnnmarl.training.replay import ReplayBuffer

ALGOS = [IQL, VDN, QMIX, GNNQMIX]


def _fake_batch(N: int, obs_dim: int, state_dim: int, B: int = 8):
    rb = ReplayBuffer(capacity=64, n_agents=N, obs_dim=obs_dim, state_dim=state_dim)
    for _ in range(B + 1):
        rb.push(
            obs=np.random.randn(N, obs_dim).astype(np.float32),
            state=np.random.randn(state_dim).astype(np.float32),
            actions=np.random.randint(0, 5, size=N).astype(np.int64),
            reward=float(np.random.randn()),
            next_obs=np.random.randn(N, obs_dim).astype(np.float32),
            next_state=np.random.randn(state_dim).astype(np.float32),
            done=False,
        )
    return rb.sample(B)


def _adj(N: int) -> np.ndarray:
    a = np.ones((N, N), dtype=np.float32) - np.eye(N, dtype=np.float32)
    return a


def test_each_algo_act_and_learn_run():
    N, obs_dim, state_dim, n_actions = 4, 4 + 3 * 3, 4 * 4, 5
    cfg = AlgoConfig(
        n_agents=N,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=16,
        embed_dim=16,
        gnn_layers=2,
    )
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    obs_single = np.random.randn(N, obs_dim).astype(np.float32)
    batch = _fake_batch(N, obs_dim, state_dim)
    adj = _adj(N)
    for AlgoCls in ALGOS:
        algo = AlgoCls(cfg, device)
        a = algo.act(obs_single, epsilon=0.5, rng=rng)
        assert a.shape == (N,)
        assert a.dtype == np.int64
        assert a.min() >= 0 and a.max() < n_actions
        metrics = algo.learn(batch, adj)
        assert "loss" in metrics
        assert np.isfinite(metrics["loss"])


def test_gnn_qmix_uses_adj():
    """Sanity: GNN-QMIX produces different Q_tot under different adjacencies."""
    N, obs_dim, state_dim = 4, 4 + 3 * 3, 4 * 4
    cfg = AlgoConfig(
        n_agents=N, obs_dim=obs_dim, state_dim=state_dim, n_actions=5,
        hidden=16, embed_dim=16, gnn_layers=2,
    )
    device = torch.device("cpu")
    algo = GNNQMIX(cfg, device)
    batch = _fake_batch(N, obs_dim, state_dim, B=4)
    adj1 = np.ones((N, N), dtype=np.float32) - np.eye(N, dtype=np.float32)  # full
    adj2 = np.zeros((N, N), dtype=np.float32)  # empty
    np.fill_diagonal(adj2, 0)

    # Run mixer manually to compare outputs
    obs_t = torch.from_numpy(batch.obs)
    state_t = torch.from_numpy(batch.state)
    actions_t = torch.from_numpy(batch.actions)
    q_taken = algo.q(obs_t).gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)

    out1 = algo.mixer(q_taken, state_t, torch.from_numpy(adj1))
    out2 = algo.mixer(q_taken, state_t, torch.from_numpy(adj2))
    assert not torch.allclose(out1, out2, atol=1e-4), (
        "GNN-QMIX produced identical output under full vs empty graph — "
        "the GNN is not actually using the adjacency."
    )
