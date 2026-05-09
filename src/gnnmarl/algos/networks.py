"""Reusable network blocks: Q-net, mixers, GNN block.

We deliberately keep these small (≤ 2 hidden layers) — the experiments are about
*algorithmic* differences, not capacity. All four algorithms share the same
per-agent Q-net so any difference between them is attributable to the mixing
mechanism, not to representation power.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    """Per-agent Q-network: obs -> Q(a)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (..., obs_dim) -> q: (..., n_actions)
        return self.net(obs)


class VDNMixer(nn.Module):
    """Q_tot = sum_i Q_i. No state input, no parameters."""

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor  # noqa: ARG002
    ) -> torch.Tensor:
        # agent_qs: (B, N) -> (B, 1)
        return agent_qs.sum(dim=-1, keepdim=True)


class QMixer(nn.Module):
    """Monotonic mixer (Rashid et al. 2018), simplified.

    Takes per-agent Q values (B, N) and global state (B, state_dim);
    outputs Q_tot (B, 1). Hypernetworks generate non-negative mixer weights
    so that ∂Q_tot / ∂Q_i ≥ 0 (monotonicity).
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # agent_qs: (B, N), state: (B, state_dim)
        B = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        x = agent_qs.unsqueeze(1)  # (B, 1, N)
        hidden = F.elu(torch.bmm(x, w1) + b1)  # (B, 1, embed)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        return q_tot.view(B, 1)


class GNNBlock(nn.Module):
    """One layer of message passing on a fixed adjacency.

    Equivalent to a graph convolution with mean aggregation:
        h_i' = ReLU( W (h_i + mean_{j in N(i)} h_j) )
    Identity (self-loop) is added so isolated nodes still update.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (B, N, in_dim), adj: (N, N) — broadcast over batch
        # add self-loops + degree-normalize
        N = h.size(1)
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype)
        a_hat = adj + eye
        deg = a_hat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        a_norm = a_hat / deg
        # message: aggregate
        m = torch.einsum("ij,bjd->bid", a_norm, h)
        return F.relu(self.lin(m))


class GNNQMixer(nn.Module):
    """GNN-conditioned monotonic mixer.

    The novelty over QMIX: per-agent contribution embeddings pass through a
    small GNN over the *coordination graph* before being combined into the
    monotonic mixer's hidden representation. The mixer remains monotonic
    in the original per-agent Q values via a separate non-negative weight path.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        gnn_layers: int = 2,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # GNN tower: each agent gets a small embedding (Q_i + per-agent state slice)
        self.gnn = nn.ModuleList(
            [GNNBlock(1 + state_dim // n_agents, embed_dim)]
            + [GNNBlock(embed_dim, embed_dim) for _ in range(gnn_layers - 1)]
        )

        # Monotonic mixer over the gnn-pooled representation + raw agent_qs
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        # b2 also conditioned on the GNN-pooled summary (this is where graph info enters)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        # agent_qs: (B, N), state: (B, state_dim), adj: (N, N)
        B = agent_qs.size(0)
        # Build per-agent input: their Q + their slice of state
        slice_dim = self.state_dim // self.n_agents
        per_agent_state = state.view(B, self.n_agents, slice_dim)
        h = torch.cat([agent_qs.unsqueeze(-1), per_agent_state], dim=-1)
        for layer in self.gnn:
            h = layer(h, adj)
        gnn_summary = h.mean(dim=1)  # (B, embed)

        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        x = agent_qs.unsqueeze(1)  # (B, 1, N)
        hidden = F.elu(torch.bmm(x, w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(torch.cat([state, gnn_summary], dim=-1)).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(B, 1)
