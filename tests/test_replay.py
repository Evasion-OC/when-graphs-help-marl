"""Tests for `gnnmarl.utils.replay.ReplayBuffer`."""

from __future__ import annotations

import numpy as np
import torch

from gnnmarl.utils.replay import Batch, ReplayBuffer, Transition


# ---------------------------------------------------------------------- helpers
def _make_transition(
    *,
    n: int = 4,
    obs_dim: int = 11,
    state_dim: int = 8,
    n_actions: int = 5,
    reward: float = 0.0,
    done: bool = False,
    rng: np.random.Generator | None = None,
) -> Transition:
    rng = rng if rng is not None else np.random.default_rng(0)
    return Transition(
        obs=rng.standard_normal((n, obs_dim)).astype(np.float32),
        state=rng.standard_normal(state_dim).astype(np.float32),
        actions=rng.integers(0, n_actions, size=n).astype(np.int64),
        reward=float(reward),
        next_obs=rng.standard_normal((n, obs_dim)).astype(np.float32),
        next_state=rng.standard_normal(state_dim).astype(np.float32),
        done=bool(done),
        adj=(rng.integers(0, 2, size=(n, n))).astype(np.float32),
    )


# ------------------------------------------------------------------------ tests
def test_add_and_len() -> None:
    buf = ReplayBuffer(capacity=100, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    for _ in range(10):
        buf.add(_make_transition(rng=rng))
    assert len(buf) == 10


def test_capacity_ringbuffer() -> None:
    """Ring buffer must overwrite the oldest entries first."""
    buf = ReplayBuffer(capacity=100, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    for i in range(150):
        buf.add(_make_transition(rng=rng, reward=float(i)))
    assert len(buf) == 100

    # Drain the full buffer with a 1:1 sample (no replacement) and check the
    # smallest surviving reward. After 150 inserts into a cap-100 ring, the
    # remaining rewards are 50..149, so the minimum should be 50.
    sample = buf.sample(100, np.random.default_rng(123))
    rewards = sample.reward.cpu().numpy()
    assert rewards.min() == 50.0
    assert rewards.max() == 149.0


def test_sample_shapes() -> None:
    n, obs_dim, state_dim, n_actions = 4, 11, 8, 5
    buf = ReplayBuffer(capacity=200, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    for _ in range(20):
        buf.add(
            _make_transition(
                n=n, obs_dim=obs_dim, state_dim=state_dim, n_actions=n_actions, rng=rng
            )
        )

    batch = buf.sample(8, np.random.default_rng(1))
    assert isinstance(batch, Batch)
    assert batch.obs.shape == (8, n, obs_dim)
    assert batch.obs.dtype == torch.float32
    assert batch.state.shape == (8, state_dim)
    assert batch.state.dtype == torch.float32
    assert batch.actions.shape == (8, n)
    assert batch.actions.dtype == torch.int64
    assert batch.reward.shape == (8,)
    assert batch.reward.dtype == torch.float32
    assert batch.next_obs.shape == (8, n, obs_dim)
    assert batch.next_obs.dtype == torch.float32
    assert batch.next_state.shape == (8, state_dim)
    assert batch.next_state.dtype == torch.float32
    assert batch.done.shape == (8,)
    assert batch.done.dtype == torch.float32
    assert batch.adj.shape == (8, n, n)
    assert batch.adj.dtype == torch.float32


def test_sample_devices() -> None:
    buf = ReplayBuffer(capacity=50, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    for _ in range(20):
        buf.add(_make_transition(rng=rng))
    batch = buf.sample(4, np.random.default_rng(0))
    for name in ("obs", "state", "actions", "reward", "next_obs", "next_state", "done", "adj"):
        t = getattr(batch, name)
        assert t.device.type == "cpu", f"{name} on {t.device}"


def test_sample_deterministic() -> None:
    """Two identically-seeded generators must yield identical samples."""
    buf = ReplayBuffer(capacity=64, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    for i in range(40):
        buf.add(_make_transition(rng=rng, reward=float(i)))

    b1 = buf.sample(8, np.random.default_rng(2024))
    b2 = buf.sample(8, np.random.default_rng(2024))

    assert torch.equal(b1.reward, b2.reward)
    assert torch.equal(b1.actions, b2.actions)
    assert torch.equal(b1.obs, b2.obs)


def test_done_is_float() -> None:
    buf = ReplayBuffer(capacity=32, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    for i in range(16):
        buf.add(_make_transition(rng=rng, done=bool(i % 2)))
    batch = buf.sample(8, np.random.default_rng(5))
    assert batch.done.dtype == torch.float32
    uniq = set(batch.done.cpu().numpy().tolist())
    assert uniq.issubset({0.0, 1.0})
