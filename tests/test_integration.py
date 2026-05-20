"""End-to-end Phase-0 integration test.

For each of the four algorithms we run a tiny training job against a real
CoordGrid and verify the trainer writes a well-formed ``episodes.csv``. This is
the test the Phase-0 acceptance criterion ("smoke config trains all four
algorithms for 1k steps without crashing, logs land in ``results/``") collapses
to — kept at 200 env steps here so it runs in CI.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gnnmarl.training import TrainConfig, train
from gnnmarl.utils.logging import HEADER


@pytest.mark.parametrize("algo", ["iql", "vdn", "qmix", "gnn_qmix"])
def test_end_to_end_smoke(algo: str, tmp_path: Path) -> None:
    cfg = TrainConfig(
        env="coord_grid",
        env_kwargs={"n_agents": 4, "grid_size": 4, "episode_steps": 10, "graph": "ring"},
        algo=algo,
        algo_kwargs={
            "buffer_capacity": 500,
            "batch_size": 16,
            "target_update_interval": 50,
        },
        total_env_steps=200,
        eps_start=1.0,
        eps_end=0.1,
        eps_anneal_steps=150,
        seed=7,
        log_dir=tmp_path,
    )
    run_dir = train(cfg)

    csv_path = run_dir / "episodes.csv"
    assert csv_path.exists(), f"missing {csv_path}"

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == HEADER, "csv header drifted from spec"
    # 200 env steps / 10-step episodes => 20 episodes exactly.
    assert len(rows) - 1 == 20, f"expected 20 episode rows, got {len(rows) - 1}"

    cfg_path = run_dir / "config.json"
    assert cfg_path.exists(), "config sidecar missing"


def test_train_returns_run_dir_under_log_dir(tmp_path: Path) -> None:
    cfg = TrainConfig(
        env="coord_grid",
        env_kwargs={"n_agents": 2, "grid_size": 3, "episode_steps": 5, "graph": "line"},
        algo="iql",
        algo_kwargs={"buffer_capacity": 200, "batch_size": 8, "target_update_interval": 25},
        total_env_steps=50,
        eps_anneal_steps=40,
        seed=1,
        log_dir=tmp_path,
        run_id="custom_id_42",
    )
    run_dir = train(cfg)
    assert run_dir == tmp_path / "custom_id_42"
