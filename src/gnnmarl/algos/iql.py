"""Independent Q-Learning (cooperative variant).

Each agent has the same shared Q-network (parameter sharing across agents),
trained independently on the *shared cooperative reward*. No mixing.

This is the canonical "no coordination signal" baseline — any improvement
from VDN / QMIX / GNN-QMIX over IQL on the same env is attributable to the
mixing mechanism.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .base import AlgoConfig, BaseAgent
from .networks import QNet


class IQL(BaseAgent):
    name = "iql"

    def __init__(self, cfg: AlgoConfig, device: torch.device):
        super().__init__(cfg, device)
        self.q = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(device)
        self.q_target = copy.deepcopy(self.q).eval().to(device)
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        # obs: (N, obs_dim)
        obs_t = torch.from_numpy(obs).to(self.device)
        q = self.q(obs_t)  # (N, A)
        greedy = q.argmax(dim=-1).cpu().numpy()
        explore = rng.random(self.cfg.n_agents) < epsilon
        rand = rng.integers(0, self.cfg.n_actions, size=self.cfg.n_agents)
        out = np.where(explore, rand, greedy).astype(np.int64)
        return out

    def learn(self, batch, adj: np.ndarray) -> dict:  # noqa: ARG002 — IQL ignores adj
        cfg = self.cfg
        d = self.device
        obs = torch.from_numpy(batch.obs).to(d)              # (B, N, O)
        actions = torch.from_numpy(batch.actions).to(d)      # (B, N)
        reward = torch.from_numpy(batch.reward).to(d)        # (B,)
        next_obs = torch.from_numpy(batch.next_obs).to(d)    # (B, N, O)
        done = torch.from_numpy(batch.done).to(d)            # (B,)

        q_all = self.q(obs)                                  # (B, N, A)
        q_taken = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (B, N)

        with torch.no_grad():
            q_next = self.q_target(next_obs).max(dim=-1).values  # (B, N)
            # shared reward broadcast to all agents (cooperative IQL)
            target = reward.unsqueeze(-1) + cfg.gamma * (1.0 - done.unsqueeze(-1)) * q_next

        loss = F.mse_loss(q_taken, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.opt.step()

        self._step += 1
        if self._step % cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return {"loss": float(loss.item()), "q_mean": float(q_taken.mean().item())}
