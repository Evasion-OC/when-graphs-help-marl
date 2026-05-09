"""Base agent interface — every algorithm exposes these methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AlgoConfig:
    n_agents: int
    obs_dim: int
    state_dim: int
    n_actions: int
    hidden: int = 64
    embed_dim: int = 32
    gnn_layers: int = 2  # only used by gnn_qmix
    lr: float = 5e-4
    # Optional separate mixer learning rate. None → use lr for both Q-net and mixer.
    # Used by Phase 2's tuning grid to address Phase 1b's finding that QMIX
    # didn't recover from embed_dim shrinkage alone — a smaller mixer LR may
    # prevent the state-conditioned bias path from absorbing TD error.
    mixer_lr: float | None = None
    gamma: float = 0.99
    target_update_every: int = 200
    grad_clip: float = 10.0
    # Mixer-only initialization. 'default' → PyTorch defaults (kaiming_uniform).
    # 'orthogonal' → orthogonal init with `init_scale` gain. Phase 1b suggested
    # init might matter for QMIX in particular.
    mixer_init: str = "default"
    init_scale: float = 0.1  # only used when mixer_init='orthogonal'


class BaseAgent(ABC):
    """Common API: act() during rollout, learn() on a batch."""

    name: str = "base"

    def __init__(self, cfg: AlgoConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._step = 0

    @abstractmethod
    def act(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        """Greedy action with epsilon-greedy exploration. Returns (n_agents,) ints."""

    @abstractmethod
    def learn(self, batch, adj: np.ndarray) -> dict:
        """One gradient step. Returns metrics dict."""

    def _maybe_target_sync(self) -> None:
        pass
