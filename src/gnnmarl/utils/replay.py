"""Replay buffer + transition/batch dataclasses.

Storage is NumPy ring-buffers (one preallocated array per field, indexed by a
write cursor). Sampling converts the requested slice into torch tensors on the
buffer's configured device. Shapes and dtypes are pinned to the contract in
``docs/INTERFACES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    """A single environment transition (one team step).

    Shapes / dtypes are exactly as specified in ``docs/INTERFACES.md``.
    """

    obs: np.ndarray         # [N, obs_dim] f32
    state: np.ndarray       # [state_dim] f32
    actions: np.ndarray     # [N] int64
    reward: float
    next_obs: np.ndarray    # [N, obs_dim] f32
    next_state: np.ndarray  # [state_dim] f32
    done: bool
    adj: np.ndarray         # [N, N] f32


@dataclass
class Batch:
    """A minibatch of transitions as torch tensors with leading batch dim."""

    obs: torch.Tensor         # [B, N, obs_dim] f32
    state: torch.Tensor       # [B, state_dim] f32
    actions: torch.Tensor     # [B, N] int64
    reward: torch.Tensor      # [B] f32
    next_obs: torch.Tensor    # [B, N, obs_dim] f32
    next_state: torch.Tensor  # [B, state_dim] f32
    done: torch.Tensor        # [B] f32 in {0., 1.}
    adj: torch.Tensor         # [B, N, N] f32


class ReplayBuffer:
    """FIFO ring buffer with uniform-random batched sampling.

    Storage is lazy: arrays are allocated on the first ``add`` call once we
    know the per-transition shapes. This keeps the buffer agnostic to env
    dimensions while still being a single contiguous numpy block per field
    (which keeps sampling fast and copy-free until the torch conversion).
    """

    def __init__(self, capacity: int, device: torch.device):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.device = device

        # Lazily allocated on first add — we don't know N/obs_dim/state_dim yet.
        self._obs: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._reward: np.ndarray | None = None
        self._next_obs: np.ndarray | None = None
        self._next_state: np.ndarray | None = None
        self._done: np.ndarray | None = None
        self._adj: np.ndarray | None = None

        self._cursor: int = 0
        self._size: int = 0

    # ------------------------------------------------------------------ utils
    def _allocate(self, t: Transition) -> None:
        n, obs_dim = t.obs.shape
        (state_dim,) = t.state.shape
        cap = self.capacity
        self._obs = np.zeros((cap, n, obs_dim), dtype=np.float32)
        self._state = np.zeros((cap, state_dim), dtype=np.float32)
        self._actions = np.zeros((cap, n), dtype=np.int64)
        self._reward = np.zeros((cap,), dtype=np.float32)
        self._next_obs = np.zeros((cap, n, obs_dim), dtype=np.float32)
        self._next_state = np.zeros((cap, state_dim), dtype=np.float32)
        self._done = np.zeros((cap,), dtype=np.float32)
        self._adj = np.zeros((cap, n, n), dtype=np.float32)

    # ----------------------------------------------------------------- public
    def add(self, t: Transition) -> None:
        if self._obs is None:
            self._allocate(t)
        i = self._cursor
        # Mypy-friendly local aliases; these are guaranteed non-None after _allocate.
        assert self._obs is not None
        assert self._state is not None
        assert self._actions is not None
        assert self._reward is not None
        assert self._next_obs is not None
        assert self._next_state is not None
        assert self._done is not None
        assert self._adj is not None

        self._obs[i] = t.obs
        self._state[i] = t.state
        self._actions[i] = t.actions
        self._reward[i] = float(t.reward)
        self._next_obs[i] = t.next_obs
        self._next_state[i] = t.next_state
        self._done[i] = 1.0 if t.done else 0.0
        self._adj[i] = t.adj

        self._cursor = (self._cursor + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        if self._obs is None:
            # Defensive — should be unreachable given the size check.
            raise RuntimeError("buffer not initialised")

        # Uniform sampling without replacement when we have at least batch_size
        # distinct transitions; otherwise fall back to with-replacement so the
        # caller still gets the requested shape (training loops occasionally
        # warm up sampling before the buffer is large enough).
        replace = self._size < batch_size
        idx = rng.choice(self._size, size=batch_size, replace=replace)

        dev = self.device
        return Batch(
            obs=torch.as_tensor(self._obs[idx], dtype=torch.float32, device=dev),
            state=torch.as_tensor(self._state[idx], dtype=torch.float32, device=dev),
            actions=torch.as_tensor(self._actions[idx], dtype=torch.int64, device=dev),
            reward=torch.as_tensor(self._reward[idx], dtype=torch.float32, device=dev),
            next_obs=torch.as_tensor(self._next_obs[idx], dtype=torch.float32, device=dev),
            next_state=torch.as_tensor(self._next_state[idx], dtype=torch.float32, device=dev),
            done=torch.as_tensor(self._done[idx], dtype=torch.float32, device=dev),
            adj=torch.as_tensor(self._adj[idx], dtype=torch.float32, device=dev),
        )
