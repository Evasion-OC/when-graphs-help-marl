"""Independent Q-learning (IQL).

True classical IQL: each agent has its own Q-function (parameter-shared
through the agent-id one-hot trick) and is trained against its own per-agent
Bellman target on the shared cooperative reward. The team utility is *not*
explicitly mixed; the loss is the sum of per-agent Huber TD losses, not the
Huber of the summed TD.

This is what differentiates IQL from VDN. VDN minimises
``Huber( sum_i Q_i - (r + gamma sum_i max_a' Q_i^-(o', a)) )``,
which couples the per-agent gradients through the *outer* nonlinearity. IQL
minimises
``sum_i Huber( Q_i - (r + gamma max_a' Q_i^-(o', a)) )``,
where each agent's TD error is clipped independently. The two are
numerically equivalent **only** when the Huber operates in its linear
regime; they differ once any TD error exceeds the Huber threshold (which is
typical early in training).

We implement IQL by overriding :meth:`_td_step` so the loss is computed
per-agent. The other plumbing (replay, target net, Double-DQN action
selection, hard target updates) is inherited from :class:`BaseAlgo`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gnnmarl.utils.replay import Batch

from .base import BaseAlgo


class IQL(BaseAlgo):
    """Independent Q-learning with parameter sharing."""

    name = "iql"

    # --- forward / mixing --------------------------------------------------
    def _q_values(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        del adj  # IQL ignores the graph entirely.
        return self.agent_q(obs)

    def _total_q(
        self,
        q_per_agent: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        # Only used for compatibility with BaseAlgo's API; IQL does not mix.
        del state, adj
        chosen = q_per_agent.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        return chosen.sum(dim=-1)  # [B] — unused by our _td_step override

    # --- training step (override) -----------------------------------------
    def _td_step(self, batch: Batch) -> tuple[float, float]:
        """Per-agent Huber TD step on the shared reward.

        For each agent i:
            y_i = r + gamma (1 - done) max_{a'} Q_i^-(o_i', a')   (Double-DQN)
            L_i = Huber( Q_i(o_i, a_i) - y_i )
        Total loss = sum_i L_i (mean over the batch).
        """
        # Online Q for the chosen actions.
        q = self._q_values(batch.obs, batch.adj)  # [B, N, A]
        chosen = q.gather(dim=-1, index=batch.actions.unsqueeze(-1)).squeeze(-1)  # [B, N]

        # Double-DQN per-agent target.
        with torch.no_grad():
            next_q_online = self._q_values(batch.next_obs, batch.adj)             # [B, N, A]
            next_actions = next_q_online.argmax(dim=-1, keepdim=True)             # [B, N, 1]
            next_q_target = self._q_values_target(batch.next_obs, batch.adj)      # [B, N, A]
            next_q_chosen = next_q_target.gather(dim=-1, index=next_actions).squeeze(-1)  # [B, N]
            y = batch.reward.unsqueeze(-1) + self.gamma * (1.0 - batch.done.unsqueeze(-1)) * next_q_chosen

        # Per-agent Huber loss, summed across agents, meaned across batch.
        loss = F.smooth_l1_loss(chosen, y, reduction="none")  # [B, N]
        loss = loss.sum(dim=-1).mean()                        # scalar

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online_parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.detach().item(), float(grad_norm)
