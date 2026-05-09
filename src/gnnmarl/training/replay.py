"""Tiny episodic replay buffer for off-policy MARL training.

We store transitions in a flat ring buffer. For QMIX-family losses we need
the global state at both s and s', so we track those alongside per-agent
observations.

Shapes (per stored sample):
    obs:        (N, obs_dim)
    state:      (state_dim,)
    actions:    (N,)
    reward:     ()
    next_obs:   (N, obs_dim)
    next_state: (state_dim,)
    done:       ()
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    next_state: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        seed: int = 0,
    ):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

        self._idx = 0
        self._size = 0

    def push(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self.obs[i] = obs
        self.state[i] = state
        self.actions[i] = actions
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.next_state[i] = next_state
        self.done[i] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = self.rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=self.obs[idx],
            state=self.state[idx],
            actions=self.actions[idx],
            reward=self.reward[idx],
            next_obs=self.next_obs[idx],
            next_state=self.next_state[idx],
            done=self.done[idx],
        )

    def __len__(self) -> int:
        return self._size
