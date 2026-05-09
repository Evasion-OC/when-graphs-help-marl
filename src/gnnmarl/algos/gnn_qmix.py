"""GNN-QMIX — graph-conditioned monotonic mixer.

The novelty over QMIX: the mixer's state-conditioned bias term is computed
using a small GNN over the *coordination graph* applied to per-agent embeddings.
This is a faithful instantiation of Algorithm 1 from the source paper
(Appendix C, "GNN-QMIX") with deliberately conservative architectural choices
(2 GNN layers, mean aggregation, identity self-loops).

Decentralised execution is preserved: each agent acts greedily on its own Q_i.
"""
from __future__ import annotations

import copy
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .base import AlgoConfig, BaseAgent
from .networks import GNNQMixer, QNet, apply_orthogonal_init


class GNNQMIX(BaseAgent):
    name = "gnn_qmix"

    def __init__(self, cfg: AlgoConfig, device: torch.device):
        super().__init__(cfg, device)
        self.q = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(device)
        self.q_target = copy.deepcopy(self.q).eval().to(device)
        # Build + initialize mixer on CPU, then move to device. (See QMIX for
        # why: torch.linalg.qr — used by orthogonal init — is not implemented
        # on PyTorch's MPS backend.)
        mixer_cpu = GNNQMixer(
            n_agents=cfg.n_agents,
            state_dim=cfg.state_dim,
            embed_dim=cfg.embed_dim,
            gnn_layers=cfg.gnn_layers,
        )
        if cfg.mixer_init == "orthogonal":
            apply_orthogonal_init(mixer_cpu, gain=cfg.init_scale)
        elif cfg.mixer_init != "default":
            raise ValueError(f"Unknown mixer_init: {cfg.mixer_init}")
        self.mixer = mixer_cpu.to(device)
        self.mixer_target = copy.deepcopy(self.mixer).eval().to(device)
        for p in chain(self.q_target.parameters(), self.mixer_target.parameters()):
            p.requires_grad_(False)
        mixer_lr = cfg.mixer_lr if cfg.mixer_lr is not None else cfg.lr
        self.opt = optim.Adam(
            [
                {"params": self.q.parameters(), "lr": cfg.lr},
                {"params": self.mixer.parameters(), "lr": mixer_lr},
            ]
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        obs_t = torch.from_numpy(obs).to(self.device)
        q = self.q(obs_t)
        greedy = q.argmax(dim=-1).cpu().numpy()
        explore = rng.random(self.cfg.n_agents) < epsilon
        rand = rng.integers(0, self.cfg.n_actions, size=self.cfg.n_agents)
        return np.where(explore, rand, greedy).astype(np.int64)

    def learn(self, batch, adj: np.ndarray) -> dict:
        cfg = self.cfg
        d = self.device
        obs = torch.from_numpy(batch.obs).to(d)
        state = torch.from_numpy(batch.state).to(d)
        actions = torch.from_numpy(batch.actions).to(d)
        reward = torch.from_numpy(batch.reward).to(d)
        next_obs = torch.from_numpy(batch.next_obs).to(d)
        next_state = torch.from_numpy(batch.next_state).to(d)
        done = torch.from_numpy(batch.done).to(d)
        adj_t = torch.from_numpy(adj.astype(np.float32)).to(d)

        q_all = self.q(obs)
        q_taken = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(q_taken, state, adj_t).squeeze(-1)

        with torch.no_grad():
            q_next_max = self.q_target(next_obs).max(dim=-1).values
            q_tot_next = self.mixer_target(q_next_max, next_state, adj_t).squeeze(-1)
            target = reward + cfg.gamma * (1.0 - done) * q_tot_next

        loss = F.mse_loss(q_tot, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q.parameters()) + list(self.mixer.parameters()), cfg.grad_clip
        )
        self.opt.step()

        self._step += 1
        if self._step % cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
            self.mixer_target.load_state_dict(self.mixer.state_dict())

        return {"loss": float(loss.item()), "q_tot_mean": float(q_tot.mean().item())}
