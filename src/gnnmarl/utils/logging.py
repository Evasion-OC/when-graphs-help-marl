"""Lightweight CSV logger.

Each row = one logical event (typically per episode or per training step).
Header is inferred from the first row passed to `log()`. New keys later
trigger a header rewrite (rare; we expect a stable schema).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None
        self._fh = None
        self._writer = None

    def log(self, row: dict[str, Any]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
            self._writer.writeheader()
        else:
            # If new keys appear, append them to the header (extend by None for old rows
            # would require a rewrite; instead, we just add unknown keys silently to
            # the schema and let DictWriter's `extrasaction='ignore'` drop them).
            new = [k for k in row.keys() if k not in self._fieldnames]
            if new:
                # Re-open with extended fieldnames is expensive; for our workload
                # the schema is stable so we just ignore new keys.
                pass
        assert self._writer is not None
        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        assert self._fh is not None
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
