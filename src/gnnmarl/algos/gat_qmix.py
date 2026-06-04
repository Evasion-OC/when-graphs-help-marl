"""GAT-QMIX — QMIX with a graph-*attention* encoder (second GNN family).

Identical to :class:`gnnmarl.algos.gnn_qmix.GNNQMIX` except the per-agent
encoder's :class:`GCNStack` is replaced by a :class:`GATStack` (single-head
graph attention, Veličković et al. 2018). Everything else --- the encoder,
the monotonic QMIX mixer, the optimiser, the exploration schedule --- is
unchanged.

Why this exists. The paper's negative result is established with a vanilla
GCN. The most natural objection is that the *specific* GNN is to blame:
the attention-based methods the paper critiques (DGN, G2ANet) use learned
edge weights, not fixed degree-normalized ones, so perhaps attention
rescues the graph. GAT-QMIX tests exactly that. The decisive comparisons
mirror GNN-QMIX:

    GAT-QMIX ~ MLP-QMIX   -> learned attention also contributes nothing
    GAT-QMIX < MLP-QMIX   -> the graph hurts even with attention
    GAT-QMIX > MLP-QMIX   -> attention is what was missing (the positive case)

and GAT-QMIX vs GNN-QMIX isolates attention vs fixed aggregation. Note
GATStack has marginally *more* parameters than the MLP control (the two
attention vectors per layer), so a GAT-vs-MLP deficit is conservative.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseAlgo
from .networks import GATStack, QMixerHypernet


class _GATAgentNet(nn.Module):
    """Per-agent encoder + GAT + Q head.

    Mirrors :class:`gnnmarl.algos.gnn_qmix._GNNAgentNet` exactly, with
    :class:`GATStack` in place of :class:`GCNStack`. Architecture:

        ``Linear(obs+id, 64) -> ReLU -> Linear(64, gnn_hidden) -> ReLU
        -> GAT(gnn_hidden, gnn_hidden, gnn_layers) -> Linear(gnn_hidden, n_actions)``
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int,
        gnn_hidden: int,
        gnn_layers: int,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gnn_hidden = gnn_hidden
        self.gnn_layers = gnn_layers

        in_dim = obs_dim + n_agents
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, gnn_hidden),
            nn.ReLU(),
        )
        self.gnn = GATStack(in_dim=gnn_hidden, hidden_dim=gnn_hidden, n_layers=gnn_layers)
        self.head = nn.Linear(gnn_hidden, n_actions)

        self.register_buffer(
            "_agent_eye",
            torch.eye(n_agents, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 3:
            raise ValueError(f"_GATAgentNet expects obs[B,N,D]; got {obs.shape}")
        b, n, d = obs.shape
        if n != self.n_agents or d != self.obs_dim:
            raise ValueError(
                f"_GATAgentNet got (N,D)=({n},{d}); expected ({self.n_agents},{self.obs_dim})"
            )

        ids = self._agent_eye.unsqueeze(0).expand(b, n, n)  # [B, N, n_agents]
        x = torch.cat([obs, ids], dim=-1)                   # [B, N, obs+id]
        h = self.encoder(x)                                  # [B, N, gnn_hidden]
        h = self.gnn(h, adj)                                 # [B, N, gnn_hidden]
        return self.head(h)                                  # [B, N, n_actions]


class GATQMIX(BaseAlgo):
    """QMIX with a single-head graph-attention encoder over the adjacency."""

    name = "gat_qmix"

    def _register_extra_modules(self) -> None:
        if hasattr(self, "agent_q"):
            del self.agent_q
        self.gat_agent_q = _GATAgentNet(
            obs_dim=self.obs_dim,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            gnn_hidden=self.gnn_hidden,
            gnn_layers=self.gnn_layers,
        )
        self.mixer = QMixerHypernet(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            embed_dim=32,
            hyper_hidden=32,
        )

    def _q_values(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.gat_agent_q(obs, adj)

    def _total_q(
        self,
        q_per_agent: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        del adj
        chosen = q_per_agent.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # [B, N]
        return self.mixer(chosen, state)  # [B]
