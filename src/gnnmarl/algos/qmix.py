"""QMIX (Rashid et al., 2018).

Mixes per-agent Q-values with a monotonic hypernetwork conditioned on the
privileged global state. The mixer's nonnegative weights ensure
:math:`\\partial Q_{tot} / \\partial Q_i \\ge 0`, which keeps the per-agent
greedy argmax consistent with the team greedy argmax (the IGM principle).
"""

from __future__ import annotations

import torch

from .base import BaseAlgo
from .networks import QMixerHypernet


class QMIX(BaseAlgo):
    """Per-agent Q's mixed monotonically by a state-conditioned hypernet."""

    name = "qmix"

    def _register_extra_modules(self) -> None:
        # Constructed after super().__init__() has stashed n_agents/state_dim.
        self.mixer = QMixerHypernet(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            embed_dim=32,
            hyper_hidden=32,
        )

    def _q_values(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        del adj
        return self.agent_q(obs)

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
