"""Tests for the MLP-QMIX parameter-matched control.

The scientific value of MLP-QMIX rests on two testable facts:
  1. It has EXACTLY the same parameter count as GNN-QMIX at matched
     (gnn_hidden, gnn_layers) — so any GNN-QMIX vs MLP-QMIX performance
     difference is attributable to the graph, not extra capacity.
  2. Its per-agent Q-values are INVARIANT to the adjacency (it carries no
     graph information), whereas GNN-QMIX's are not.
We also confirm MLP-QMIX adds capacity over plain QMIX (so the
MLP-QMIX vs QMIX comparison isolates capacity).
"""

from __future__ import annotations

import numpy as np
import torch

from gnnmarl.algos import make_algo
from gnnmarl.algos.gnn_qmix import _GNNAgentNet
from gnnmarl.algos.mlp_qmix import _MLPAgentNet

N_AGENTS = 6
OBS_DIM = 10
STATE_DIM = 24
N_ACTIONS = 5


def _make(name: str, **kw):
    base = dict(
        n_agents=N_AGENTS,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        device=torch.device("cpu"),
        seed=0,
    )
    base.update(kw)
    return make_algo(name, **base)


def _nparams(algo) -> int:
    seen, tot = set(), 0
    for m in algo.__dict__.values():
        if isinstance(m, torch.nn.Module):
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    tot += p.numel()
    return tot


def test_mlp_and_gnn_have_identical_param_count():
    for gnn_layers in (1, 2, 3):
        for gnn_hidden in (32, 64):
            g = _make("gnn_qmix", gnn_layers=gnn_layers, gnn_hidden=gnn_hidden)
            m = _make("mlp_qmix", gnn_layers=gnn_layers, gnn_hidden=gnn_hidden)
            assert _nparams(g) == _nparams(m), (
                f"param mismatch at layers={gnn_layers}, hidden={gnn_hidden}: "
                f"gnn={_nparams(g)} mlp={_nparams(m)}"
            )


def test_mlp_adds_capacity_over_qmix():
    m = _make("mlp_qmix", gnn_layers=2, gnn_hidden=64)
    q = _make("qmix", gnn_layers=2, gnn_hidden=64)
    assert _nparams(m) > _nparams(q)


def test_mlp_agent_net_is_adjacency_invariant():
    torch.manual_seed(0)
    obs = torch.randn(4, N_AGENTS, OBS_DIM)
    adj_full = torch.ones(4, N_AGENTS, N_AGENTS) - torch.eye(N_AGENTS)
    adj_ring = torch.zeros(4, N_AGENTS, N_AGENTS)
    for i in range(N_AGENTS):
        adj_ring[:, i, (i + 1) % N_AGENTS] = 1.0
        adj_ring[:, i, (i - 1) % N_AGENTS] = 1.0

    mlp = _MLPAgentNet(OBS_DIM, N_AGENTS, N_ACTIONS, gnn_hidden=64, gnn_layers=2)
    assert torch.allclose(mlp(obs, adj_full), mlp(obs, adj_ring)), (
        "MLP-QMIX agent net changed with adjacency — it must be graph-invariant"
    )

    torch.manual_seed(0)
    gnn = _GNNAgentNet(OBS_DIM, N_AGENTS, N_ACTIONS, gnn_hidden=64, gnn_layers=2)
    assert not torch.allclose(gnn(obs, adj_full), gnn(obs, adj_ring)), (
        "GNN-QMIX agent net did NOT change with adjacency — graph path is dead"
    )


def test_mlp_qmix_act_contract():
    algo = _make("mlp_qmix")
    rng = np.random.default_rng(0)
    obs = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
    adj = (np.ones((N_AGENTS, N_AGENTS)) - np.eye(N_AGENTS)).astype(np.float32)
    a = algo.act(obs, adj, epsilon=0.0, rng=rng)
    assert a.shape == (N_AGENTS,)
    assert a.dtype == np.int64
    assert a.min() >= 0 and a.max() < N_ACTIONS
