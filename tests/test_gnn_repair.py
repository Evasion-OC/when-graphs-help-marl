"""Tests for the anti-over-smoothing GCN knobs (residual + layernorm)."""

from __future__ import annotations

import torch

from gnnmarl.algos import make_algo
from gnnmarl.algos.networks import GCNStack


def _count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def test_gcnstack_defaults_are_plain() -> None:
    stack = GCNStack(in_dim=16, hidden_dim=16, n_layers=2)
    assert stack.residual is False
    assert stack.norms is None


def test_gcnstack_residual_layernorm_shapes_and_params() -> None:
    plain = GCNStack(in_dim=16, hidden_dim=16, n_layers=2)
    repaired = GCNStack(in_dim=16, hidden_dim=16, n_layers=2, residual=True, layernorm=True)
    # LayerNorm adds parameters; residual adds none.
    assert _count_params(repaired) > _count_params(plain)

    h = torch.randn(3, 4, 16)
    adj = (torch.rand(3, 4, 4) > 0.5).float()
    out = repaired(h, adj)
    assert out.shape == (3, 4, 16)
    assert torch.isfinite(out).all()


def test_repaired_gnn_qmix_builds_and_forwards() -> None:
    algo = make_algo(
        "gnn_qmix", n_agents=4, obs_dim=10, state_dim=8, n_actions=5,
        gnn_layers=2, gnn_hidden=64, gnn_residual=True, gnn_layernorm=True,
    )
    assert algo.gnn_agent_q.gnn.residual is True
    assert algo.gnn_agent_q.gnn.norms is not None

    obs = torch.randn(2, 4, 10)
    adj = (torch.rand(2, 4, 4) > 0.5).float()
    q = algo.gnn_agent_q(obs, adj)
    assert q.shape == (2, 4, 5)
    assert torch.isfinite(q).all()


def test_default_gnn_qmix_unchanged() -> None:
    algo = make_algo(
        "gnn_qmix", n_agents=4, obs_dim=10, state_dim=8, n_actions=5,
        gnn_layers=2, gnn_hidden=64,
    )
    assert algo.gnn_agent_q.gnn.residual is False
    assert algo.gnn_agent_q.gnn.norms is None
