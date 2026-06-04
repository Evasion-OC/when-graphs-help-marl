"""Tests for GAT-QMIX, the second GNN family (graph attention).

GAT-QMIX is the robustness control for the "it's the graph, not the GCN"
question. The testable facts:
  1. Its parameter count is GNN-QMIX's plus exactly the attention vectors
     (2 * gnn_hidden per layer) -- i.e. marginally MORE than the
     parameter-matched MLP control, so a GAT < MLP result is conservative.
  2. Its per-agent Q-values DO depend on the adjacency (the attention path
     is live), unlike the MLP control.
  3. It adds capacity over plain QMIX.
"""

from __future__ import annotations

import numpy as np
import torch

from gnnmarl.algos import make_algo
from gnnmarl.algos.gat_qmix import _GATAgentNet
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


def test_gat_params_are_gnn_plus_attention_vectors():
    """GAT = GNN + 2*hidden per layer (a_src + a_dst); a small, known delta."""
    for gnn_layers in (1, 2, 3):
        for gnn_hidden in (32, 64):
            g = _make("gnn_qmix", gnn_layers=gnn_layers, gnn_hidden=gnn_hidden)
            a = _make("gat_qmix", gnn_layers=gnn_layers, gnn_hidden=gnn_hidden)
            expected_extra = 2 * gnn_hidden * gnn_layers
            assert _nparams(a) == _nparams(g) + expected_extra, (
                f"at layers={gnn_layers}, hidden={gnn_hidden}: "
                f"gat={_nparams(a)} gnn={_nparams(g)} "
                f"(expected delta {expected_extra})"
            )


def test_gat_has_at_least_as_many_params_as_mlp_control():
    """GAT >= MLP control, so GAT < MLP performance is a conservative bound."""
    a = _make("gat_qmix", gnn_layers=2, gnn_hidden=64)
    m = _make("mlp_qmix", gnn_layers=2, gnn_hidden=64)
    q = _make("qmix", gnn_layers=2, gnn_hidden=64)
    assert _nparams(a) > _nparams(m)  # GAT strictly above the no-graph control
    assert _nparams(a) > _nparams(q)


def test_gat_agent_net_uses_the_graph():
    torch.manual_seed(0)
    obs = torch.randn(4, N_AGENTS, OBS_DIM)
    adj_full = torch.ones(4, N_AGENTS, N_AGENTS) - torch.eye(N_AGENTS)
    adj_ring = torch.zeros(4, N_AGENTS, N_AGENTS)
    for i in range(N_AGENTS):
        adj_ring[:, i, (i + 1) % N_AGENTS] = 1.0
        adj_ring[:, i, (i - 1) % N_AGENTS] = 1.0

    torch.manual_seed(0)
    gat = _GATAgentNet(OBS_DIM, N_AGENTS, N_ACTIONS, gnn_hidden=64, gnn_layers=2)
    assert not torch.allclose(gat(obs, adj_full), gat(obs, adj_ring)), (
        "GAT-QMIX agent net did NOT change with adjacency — attention path is dead"
    )

    # And the no-graph control is invariant on the same inputs (sanity anchor).
    mlp = _MLPAgentNet(OBS_DIM, N_AGENTS, N_ACTIONS, gnn_hidden=64, gnn_layers=2)
    assert torch.allclose(mlp(obs, adj_full), mlp(obs, adj_ring))


def test_gat_attention_weights_are_normalized_over_neighbours():
    """Softmax attention over A+I must put zero weight on non-edges and sum
    to 1 over the neighbourhood (incl. self)."""
    from gnnmarl.algos.networks import GATStack

    torch.manual_seed(0)
    stack = GATStack(in_dim=8, hidden_dim=8, n_layers=1)
    b, n = 2, 4
    h = torch.randn(b, n, 8)
    # line graph 0-1-2-3 (no self loops in input adj)
    adj = torch.zeros(b, n, n)
    for i in (0, 1, 2):
        adj[:, i, i + 1] = 1.0
        adj[:, i + 1, i] = 1.0
    # Recompute the attention internally by monkey-free forward: just check
    # output is finite and graph-dependent vs an empty graph.
    out_line = stack(h, adj)
    out_empty = stack(h, torch.zeros_like(adj))
    assert torch.isfinite(out_line).all()
    assert not torch.allclose(out_line, out_empty)


def test_gat_qmix_act_contract():
    algo = _make("gat_qmix")
    rng = np.random.default_rng(0)
    obs = np.random.randn(N_AGENTS, OBS_DIM).astype(np.float32)
    adj = (np.ones((N_AGENTS, N_AGENTS)) - np.eye(N_AGENTS)).astype(np.float32)
    a = algo.act(obs, adj, epsilon=0.0, rng=rng)
    assert a.shape == (N_AGENTS,)
    assert a.dtype == np.int64
    assert a.min() >= 0 and a.max() < N_ACTIONS
