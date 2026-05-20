"""Seeding utilities.

Provides a single entry point `set_seed` that seeds Python's `random` module,
NumPy's global RNG, and PyTorch's global RNG, while also returning a fresh
`np.random.Generator` and a CPU `torch.Generator` that the rest of the codebase
should use for *per-component* randomness. Per-`INTERFACES.md`, no module in
this project should pull from a global RNG at runtime — but the global seeds
are still set as belt-and-suspenders against third-party libraries that do.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> tuple[np.random.Generator, torch.Generator]:
    """Seed every global RNG and hand back fresh per-component generators.

    Args:
        seed: integer seed reused for Python, NumPy global, PyTorch global,
            and both returned generators.

    Returns:
        (numpy_generator, torch_generator) — both freshly seeded from ``seed``.
        The torch generator is a CPU generator; algorithms that need a CUDA
        generator should derive one explicitly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # cuda manual_seed is a no-op if CUDA is not available, but seeding it is
    # cheap and keeps determinism when a GPU happens to be present.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np_rng = np.random.default_rng(seed)
    torch_rng = torch.Generator(device="cpu")
    torch_rng.manual_seed(seed)
    return np_rng, torch_rng
