"""Value Decomposition Networks (VDN, Sunehag et al., 2017).

VDN is structurally identical to IQL in the per-agent net, but the team Q
that gets fed to the TD loss is explicitly defined as the *sum* of per-agent
Q-values. The single TD loss backpropagates through every agent's net,
which is the only behavioural difference vs IQL — and yet it matters a lot
in practice for cooperative tasks.

For IQL we used the same sum-of-Q reduction internally, so the implementation
of ``_total_q`` is literally identical. We keep VDN as its own class anyway
so the algorithm name shows up in logs and the file layout matches the
phase plan.
"""

from __future__ import annotations

import torch

from .base import BaseAlgo


class VDN(BaseAlgo):
    """Sum-of-Q value decomposition."""

    name = "vdn"

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
        del state, adj
        chosen = q_per_agent.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # [B, N]
        return chosen.sum(dim=-1)  # [B]
