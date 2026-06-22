"""DGN-QMIX --- QMIX with a multi-head graph-attention encoder.

The relational-attention mechanism of the attention-based graph-MARL
methods this paper discusses (DGN, Jiang et al. 2020; G2ANet, Liu et al.
2020) is multi-head scaled-dot-product attention over the coordination
graph. This algorithm puts that mechanism (:class:`DGNStack`) in the same
slot that GNN-QMIX uses for a GCN and GAT-QMIX uses for single-head
attention --- everything else (encoder, monotonic mixer, optimiser,
exploration) is unchanged.

Why this exists. The negative result is established with a vanilla GCN
(GNN-QMIX) and single-head attention (GAT-QMIX). The strongest remaining
objection is that the *sophisticated multi-head relational attention*
these methods actually use would behave differently. DGN-QMIX tests
exactly that, and it is the most expressive (and most over-parameterised)
graph variant in the study --- so DGN-QMIX < MLP-QMIX is the most
conservative possible form of the graph-penalty claim. The decisive
comparison mirrors the others:

    DGN-QMIX ~ MLP-QMIX   -> multi-head attention also contributes nothing
    DGN-QMIX < MLP-QMIX   -> the graph hurts even with the strongest bias
    DGN-QMIX > MLP-QMIX   -> the relational attention is what was missing
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseAlgo
from .networks import DGNStack, QMixerHypernet


class _DGNAgentNet(nn.Module):
    """Per-agent encoder + multi-head graph attention + Q head.

    Mirrors :class:`gnnmarl.algos.gnn_qmix._GNNAgentNet` exactly, with
    :class:`DGNStack` in place of :class:`GCNStack`. Architecture:

        ``Linear(obs+id, 64) -> ReLU -> Linear(64, gnn_hidden) -> ReLU
        -> DGN(gnn_hidden, gnn_hidden, gnn_layers) -> Linear(gnn_hidden, n_actions)``
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int,
        gnn_hidden: int,
        gnn_layers: int,
        n_heads: int = 4,
        gnn_residual: bool = False,
        gnn_layernorm: bool = False,
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
        self.gnn = DGNStack(in_dim=gnn_hidden, hidden_dim=gnn_hidden,
                            n_layers=gnn_layers, n_heads=n_heads,
                            residual=gnn_residual, layernorm=gnn_layernorm)
        self.head = nn.Linear(gnn_hidden, n_actions)

        self.register_buffer(
            "_agent_eye",
            torch.eye(n_agents, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 3:
            raise ValueError(f"_DGNAgentNet expects obs[B,N,D]; got {obs.shape}")
        b, n, d = obs.shape
        if n != self.n_agents or d != self.obs_dim:
            raise ValueError(
                f"_DGNAgentNet got (N,D)=({n},{d}); expected ({self.n_agents},{self.obs_dim})"
            )

        ids = self._agent_eye.unsqueeze(0).expand(b, n, n)  # [B, N, n_agents]
        x = torch.cat([obs, ids], dim=-1)                   # [B, N, obs+id]
        h = self.encoder(x)                                  # [B, N, gnn_hidden]
        h = self.gnn(h, adj)                                 # [B, N, gnn_hidden]
        return self.head(h)                                  # [B, N, n_actions]


class DGNQMIX(BaseAlgo):
    """QMIX with a multi-head graph-attention encoder over the adjacency."""

    name = "dgn_qmix"

    def _register_extra_modules(self) -> None:
        if hasattr(self, "agent_q"):
            del self.agent_q
        n_heads = int(getattr(self, "gnn_heads", 4))
        self.dgn_agent_q = _DGNAgentNet(
            obs_dim=self.obs_dim,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            gnn_hidden=self.gnn_hidden,
            gnn_layers=self.gnn_layers,
            n_heads=n_heads,
            gnn_residual=self.gnn_residual,
            gnn_layernorm=self.gnn_layernorm,
        )
        self.mixer = QMixerHypernet(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            embed_dim=32,
            hyper_hidden=32,
        )

    def _q_values(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.dgn_agent_q(obs, adj)

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
