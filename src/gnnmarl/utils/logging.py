"""Per-run CSV logger.

A run directory holds:

- ``config.json`` — the full TrainConfig (or any dict) for the run, written
  once at construction. Non-JSON-serializable values are stringified.
- ``episodes.csv`` — one row per training episode, header fixed by
  ``docs/INTERFACES.md``.

The CSV is flushed (and ``os.fsync``-ed where possible) after every row so that
crashed or killed runs still leave a usable partial log.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

HEADER: tuple[str, ...] = (
    "run_id",
    "algo",
    "env",
    "seed",
    "episode",
    "env_step",
    "return",
    "episode_len",
    "epsilon",
    "loss",
    "grad_norm",
    "wallclock_s",
)


def _json_safe(value: Any) -> Any:
    """Best-effort coercion to a JSON-serializable structure.

    Containers are walked recursively; anything we can't serialize directly is
    converted with ``str(...)`` so the sidecar is always writable.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


class CSVLogger:
    """Append-only CSV logger with a JSON config sidecar.

    Args:
        run_dir: directory for this run's artefacts. Created (with parents)
            if it does not exist.
        run_id: human-readable identifier; written as the first column of
            every row.
        config: arbitrary run config; dumped to ``run_dir/config.json``.
    """

    def __init__(self, run_dir: Path, run_id: str, config: dict):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Config sidecar — always recoverable, even if logging never produces
        # a single row.
        config_path = self.run_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(config), fh, indent=2, sort_keys=True)

        # Pull the canonical columns out of config so callers don't have to
        # pass them on every row. Missing keys default to empty strings —
        # the trainer always supplies them, but tests sometimes don't.
        self._algo = str(config.get("algo", ""))
        self._env = str(config.get("env", ""))
        self._seed = config.get("seed", "")

        self._path = self.run_dir / "episodes.csv"
        self._fh = self._path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(HEADER)
        self._flush()

        self._wallclock_start = _monotonic()

    # ------------------------------------------------------------------ utils
    def _flush(self) -> None:
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except (OSError, AttributeError, ValueError):
            # fsync can fail on some FS / when the fd is detached during
            # interpreter shutdown; flush() is still enough to satisfy the
            # "another reader can see the row immediately" contract.
            pass

    # ----------------------------------------------------------------- public
    def log_episode(
        self,
        episode: int,
        env_step: int,
        ret: float,
        ep_len: int,
        epsilon: float,
        loss: float | None,
        grad_norm: float | None,
    ) -> None:
        wallclock_s = _monotonic() - self._wallclock_start
        row = [
            self.run_id,
            self._algo,
            self._env,
            self._seed,
            episode,
            env_step,
            ret,
            ep_len,
            epsilon,
            "" if loss is None else loss,
            "" if grad_norm is None else grad_norm,
            wallclock_s,
        ]
        self._writer.writerow(row)
        self._flush()

    def close(self) -> None:
        if not self._fh.closed:
            try:
                self._flush()
            finally:
                self._fh.close()

    # Context-manager sugar — handy in scripts and tests.
    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _monotonic() -> float:
    """Indirection so tests can monkey-patch wallclock if they ever need to."""
    import time
    return time.monotonic()
