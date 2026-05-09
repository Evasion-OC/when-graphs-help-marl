"""Lightweight CSV logger with a fixed schema.

The schema is supplied at construction time so that heterogeneous rows
(training rows vs evaluation rows) all map cleanly into one CSV file.
Missing keys in a row land as empty strings; unknown keys raise.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = list(fieldnames)
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._fh.flush()

    def log(self, row: dict[str, Any]) -> None:
        unknown = [k for k in row if k not in self._fieldnames]
        if unknown:
            raise KeyError(
                f"CSVLogger row contains keys not in schema: {unknown}. "
                f"Schema: {self._fieldnames}"
            )
        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
