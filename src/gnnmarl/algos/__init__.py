"""Phase-0 algorithm package.

Exposes the four concrete algorithms and a tiny string-keyed factory so the
trainer / CLI can construct them by name without importing each module
individually.
"""

from __future__ import annotations

from typing import Any

from .base import Algo, BaseAlgo
from .gat_qmix import GATQMIX
from .gnn_qmix import GNNQMIX
from .iql import IQL
from .mlp_qmix import MLPQMIX
from .qmix import QMIX
from .vdn import VDN

__all__ = [
    "Algo",
    "BaseAlgo",
    "IQL",
    "VDN",
    "QMIX",
    "GNNQMIX",
    "MLPQMIX",
    "GATQMIX",
    "make_algo",
]


_REGISTRY: dict[str, type[BaseAlgo]] = {
    "iql": IQL,
    "vdn": VDN,
    "qmix": QMIX,
    "gnn_qmix": GNNQMIX,
    "mlp_qmix": MLPQMIX,
    "gat_qmix": GATQMIX,
}


def make_algo(name: str, **kwargs: Any) -> BaseAlgo:
    """Instantiate an algorithm by name.

    Args:
        name: one of ``"iql" | "vdn" | "qmix" | "gnn_qmix"``.
        **kwargs: forwarded to the algorithm constructor (see
            :class:`BaseAlgo.__init__` for the shared signature).

    Raises:
        KeyError: if ``name`` is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown algo {name!r}; expected one of {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](**kwargs)
