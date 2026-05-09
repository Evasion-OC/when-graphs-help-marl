"""Deterministic seeding utilities."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed everything we can. MPS gets the same treatment as CPU."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str = "auto") -> torch.device:
    """Choose torch device.

    `auto` → MPS if available, else CPU. `cpu` forces CPU. `mps` forces MPS.
    We don't use CUDA on the target hardware (Apple Silicon).
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available on this machine.")
        return torch.device("mps")
    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device pref: {prefer}")
