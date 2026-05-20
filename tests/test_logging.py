"""Tests for `gnnmarl.utils.logging.CSVLogger`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from gnnmarl.utils.logging import HEADER, CSVLogger


def test_writes_header_and_row(tmp_path: Path) -> None:
    logger = CSVLogger(
        run_dir=tmp_path / "run0",
        run_id="run0",
        config={"algo": "qmix", "env": "coord_grid", "seed": 0},
    )
    logger.log_episode(
        episode=1, env_step=100, ret=2.5, ep_len=50,
        epsilon=0.95, loss=0.123, grad_norm=4.56,
    )
    logger.close()

    csv_path = tmp_path / "run0" / "episodes.csv"
    with csv_path.open() as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == HEADER
    assert len(rows) == 2

    data = rows[1]
    # First four cells are the per-run constants pulled from `config`.
    assert data[0] == "run0"
    assert data[1] == "qmix"
    assert data[2] == "coord_grid"
    assert data[3] == "0"
    assert data[4] == "1"           # episode
    assert data[5] == "100"         # env_step
    assert float(data[6]) == 2.5    # return
    assert data[7] == "50"          # episode_len
    assert float(data[8]) == 0.95   # epsilon
    assert float(data[9]) == 0.123  # loss
    assert float(data[10]) == 4.56  # grad_norm
    assert float(data[11]) >= 0.0   # wallclock_s


def test_loss_none_renders_empty(tmp_path: Path) -> None:
    logger = CSVLogger(
        run_dir=tmp_path / "run1",
        run_id="run1",
        config={"algo": "iql", "env": "coord_grid", "seed": 1},
    )
    logger.log_episode(
        episode=0, env_step=0, ret=0.0, ep_len=1,
        epsilon=1.0, loss=None, grad_norm=None,
    )
    logger.close()

    with (tmp_path / "run1" / "episodes.csv").open() as fh:
        rows = list(csv.reader(fh))
    data = rows[1]
    # `loss` and `grad_norm` columns must be empty strings, not "None".
    assert data[9] == ""
    assert data[10] == ""


def test_config_sidecar(tmp_path: Path) -> None:
    cfg = {"algo": "qmix", "lr": 5e-4, "extra": Path("/tmp")}
    CSVLogger(run_dir=tmp_path / "run2", run_id="run2", config=cfg).close()

    sidecar = tmp_path / "run2" / "config.json"
    assert sidecar.exists()
    parsed = json.loads(sidecar.read_text())
    assert parsed["algo"] == "qmix"
    assert parsed["lr"] == 5e-4
    # Path must have been stringified — exact string form is platform-dependent
    # (`/tmp` on POSIX, `\\tmp` on Windows) so just assert it's a string and
    # contains "tmp".
    assert isinstance(parsed["extra"], str)
    assert "tmp" in parsed["extra"]


def test_flush_per_row(tmp_path: Path) -> None:
    """Another reader must see the row before `close()` is called."""
    logger = CSVLogger(
        run_dir=tmp_path / "run3",
        run_id="run3",
        config={"algo": "vdn", "env": "coord_grid", "seed": 7},
    )
    logger.log_episode(
        episode=0, env_step=10, ret=1.0, ep_len=10,
        epsilon=0.5, loss=0.01, grad_norm=0.2,
    )

    csv_path = tmp_path / "run3" / "episodes.csv"
    content = csv_path.read_text()
    assert "run3" in content
    # Two lines (header + data) plus the trailing newline from csv.writer.
    assert content.count("\n") >= 2

    logger.close()


def test_run_dir_created(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "run4"
    assert not nested.exists()
    logger = CSVLogger(run_dir=nested, run_id="run4", config={"algo": "iql"})
    logger.close()
    assert nested.is_dir()
    assert (nested / "episodes.csv").exists()
    assert (nested / "config.json").exists()
