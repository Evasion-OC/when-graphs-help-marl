"""GNN-QMIX — QMIX with a GCN message-passing block before the per-agent Q head.

Per-agent local observations are projected by the shared agent encoder (the
first two layers of :class:`AgentQNet`'s MLP would be the natural place, but
to keep things simple and to expose the GNN as a clean module we instead
*replace* :class:`AgentQNet`'s third (output) linear with: ``GCN -> Linear``).

Implementation: we run obs through a small linear projection to ``gnn_hidden``,
pass it through the GCN stack over the per-episode adjacency, then through a
Q head. The mixer is identical to QMIX, by design — the GNN is the only
structural change relative to QMIX so the Phase-4 depth ablation is clean.

The "agent identity one-hot" trick is still applied at the input so the
shared per-agent encoder can specialize.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseAlgo
from .networks import GCNStack, QMixerHypernet


class _GNNAgentNet(nn.Module):
    """Per-agent encoder + GCN + Q head.

    The encoder mirrors the first ReLU-MLP layers of :class:`AgentQNet`,
    then GCN message-passing happens over the static adjacency, then the
    final Q head produces ``n_actions`` Q-values per agent.

    Architecture:
        ``Linear(obs+id, 64) -> ReLU -> Linear(64, gnn_hidden) -> ReLU
        -> GCN(gnn_hidden, gnn_hidden, gnn_layers) -> Linear(gnn_hidden, n_actions)``
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
        self.gnn = GCNStack(in_dim=gnn_hidden, hidden_dim=gnn_hidden, n_layers=gnn_layers)
        self.head = nn.Linear(gnn_hidden, n_actions)

        self.register_buffer(
            "_agent_eye",
            torch.eye(n_agents, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 3:
            raise ValueError(f"_GNNAgentNet expects obs[B,N,D]; got {obs.shape}")
        b, n, d = obs.shape
        if n != self.n_agents or d != self.obs_dim:
            raise ValueError(
                f"_GNNAgentNet got (N,D)=({n},{d}); expected ({self.n_agents},{self.obs_dim})"
            )

        ids = self._agent_eye.unsqueeze(0).expand(b, n, n)  # [B, N, n_agents]
        x = torch.cat([obs, ids], dim=-1)                   # [B, N, obs+id]
        h = self.encoder(x)                                  # [B, N, gnn_hidden]
        h = self.gnn(h, adj)                                 # [B, N, gnn_hidden]
        return self.head(h)                                  # [B, N, n_actions]


class GNNQMIX(BaseAlgo):
    """QMIX with a GCN encoder over the per-episode adjacency."""

    name = "gnn_qmix"

    def _register_extra_modules(self) -> None:
        # Replace the BaseAlgo-installed AgentQNet with the GNN-aware variant.
        # Keeping the attribute name distinct (``gnn_agent_q``) makes it
        # obvious in checkpoints which architecture was trained; we keep the
        # base ``agent_q`` around but unused so the shared init code stays
        # uniform across algos.
        self.gnn_agent_q = _GNNAgentNet(
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
        return self.gnn_agent_q(obs, adj)

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
