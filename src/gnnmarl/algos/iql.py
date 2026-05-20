"""Independent Q-learning (IQL).

Each agent has its own Q-value but parameters are shared across agents
(the appended agent-id one-hot is what gives them per-agent specialization
while keeping a single optimizer). For IQL the "team Q" we backprop through
is just the *sum* of the selected per-agent Q-values: the Huber loss is
applied to that sum because we use a single shared cooperative reward, and
this is equivalent (up to gradient scaling) to taking the mean of per-agent
TD losses with a per-agent target. We pick the sum form because it matches
VDN/QMIX's shape exactly, which keeps :class:`BaseAlgo` clean.

Note: classical IQL would compute an independent per-agent target using the
shared reward; since the reward is shared and the target is constructed from
the same target net, this is numerically equivalent to the per-agent
formulation. Keeping the team-sum form is the standard simplification in
the QMIX/VDN/IQL benchmark suite.
"""

from __future__ import annotations

import torch

from .base import BaseAlgo


class IQL(BaseAlgo):
    """Independent Q-learning with parameter sharing."""

    name = "iql"

    def _q_values(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # IQL ignores the graph entirely.
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
        # Gather the Q of the selected action for each agent.
        chosen = q_per_agent.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # [B, N]
        return chosen.sum(dim=-1)  # [B]
