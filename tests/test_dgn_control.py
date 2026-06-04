"""Tests for DGN-QMIX, the multi-head graph-attention variant.

DGN-QMIX realises the relational mechanism of the attention-based
graph-MARL methods the paper discusses (multi-head dot-product graph
attention). Testable facts:
  1. It has MORE parameters than every other graph variant (Q/K/V/output
     projections per head), so a DGN < MLP result is the most conservative
     form of the graph-penalty claim.
  2. Its per-agent Q-values depend on the adjacency (the attention path is
     live), unlike the MLP control.
  3. hidden_dim must be divisible by n_heads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gnnmarl.algos import make_algo
from gnnmarl.algos.dgn_qmix import _DGNAgentNet
from gnnmarl.algos.networks import DGNStack

N_AGENTS = 6
OBS_DIM = 10
STATE_DIM = 24
N_ACTIONS = 5


def _make(name: str, **kw):
    base = dict(n_agents=N_AGENTS, obs_dim=OBS_DIM, state_dim=STATE_DIM,
                n_actions=N_ACTIONS, device=torch.device("cpu"), seed=0)
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


def test_dgn_has_most_parameters_of_all_graph_variants():
    d = _make("dgn_qmix", gnn_layers=2, gnn_hidden=64)
    g = _make("gnn_qmix", gnn_layers=2, gnn_hidden=64)
    a = _make("gat_qmix", gnn_layers=2, gnn_hidden=64)
    m = _make("mlp_qmix", gnn_layers=2, gnn_hidden=64)
    q = _make("qmix", gnn_layers=2, gnn_hidden=64)
    assert _nparams(d) > _nparams(a) > _nparams(m) == _nparams(g) > _nparams(q)


def test_dgn_agent_net_uses_the_graph():
    torch.manual_seed(0)
    obs = torch.randn(4, N_AGENTS, OBS_DIM)
    adj_full = torch.ones(4, N_AGENTS, N_AGENTS) - torch.eye(N_AGENTS)
    adj_ring = torch.zeros(4, N_AGENTS, N_AGENTS)
    for i in range(N_AGENTS):
        adj_ring[:, i, (i + 1) % N_AGENTS] = 1.0
        adj_ring[:, i, (i - 1) % N_AGENTS] = 1.0
    torch.manual_seed(0)
    dgn = _DGNAgentNet(OBS_DIM, N_AGENTS, N_ACTIONS, gnn_hidden=64, gnn_layers=2, n_heads=4)
    assert not torch.allclose(dgn(obs, adj_full), dgn(obs, adj_ring)), (
        "DGN-QMIX agent net did NOT change with adjacency --- attention path is dead"
    )


def test_dgn_attention_is_graph_masked_and_finite():
    torch.manual_seed(0)
    stack = DGNStack(in_dim=8, hidden_dim=8, n_layers=1, n_heads=2)
    b, n = 2, 4
    h = torch.randn(b, n, 8)
    adj = torch.zeros(b, n, n)
    for i in (0, 1, 2):
        adj[:, i, i + 1] = 1.0
        adj[:, i + 1, i] = 1.0
    out_line = stack(h, adj)
    out_empty = stack(h, torch.zeros_like(adj))
    assert torch.isfinite(out_line).all()
    assert not torch.allclose(out_line, out_empty)


def test_dgn_requires_divisible_heads():
    with pytest.raises(ValueError):
        DGNStack(in_dim=8, hidden_dim=10, n_layers=1, n_heads=4)


def test_dgn_qmix_act_contract():
    algo = _make("dgn_qmix")
    rng = np.random.default_rng(0)
    obs = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
    adj = (np.ones((N_AGENTS, N_AGENTS)) - np.eye(N_AGENTS)).astype(np.float32)
    a = algo.act(obs, adj, epsilon=0.0, rng=rng)
    assert a.shape == (N_AGENTS,)
    assert a.dtype == np.int64
    assert a.min() >= 0 and a.max() < N_ACTIONS
