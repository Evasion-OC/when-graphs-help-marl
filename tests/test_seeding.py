"""Tests for `gnnmarl.utils.seeding.set_seed`."""

from __future__ import annotations

import numpy as np
import torch

from gnnmarl.utils.seeding import set_seed


def test_set_seed_reproducible() -> None:
    """Two `set_seed(42)` calls must produce identical numpy + torch streams."""
    np_rng_a, torch_rng_a = set_seed(42)
    np_rng_b, torch_rng_b = set_seed(42)

    np_a = np_rng_a.random(100)
    np_b = np_rng_b.random(100)
    assert np.array_equal(np_a, np_b)

    torch_a = torch.randn(100, generator=torch_rng_a)
    torch_b = torch.randn(100, generator=torch_rng_b)
    assert torch.equal(torch_a, torch_b)


def test_set_seed_global_numpy() -> None:
    """The global numpy RNG should also be deterministic after `set_seed`."""
    set_seed(7)
    a = np.random.rand(5)
    set_seed(7)
    b = np.random.rand(5)
    assert np.array_equal(a, b)
