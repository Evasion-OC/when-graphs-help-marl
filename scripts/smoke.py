"""Phase-0 smoke test: train each algo for 1k steps and verify CSV output.

Usage::

    python scripts/smoke.py [--config configs/coord_grid_smoke.yaml]

Exit code 0 iff all four algos complete and each writes a non-empty episodes.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make the src layout importable when running as a script without install.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.training import TrainConfig, train  # noqa: E402


ALGOS = ("iql", "vdn", "qmix", "gnn_qmix")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "coord_grid_smoke.yaml",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    base["log_dir"] = Path(base.get("log_dir", "results/smoke"))

    failures: list[str] = []
    for algo in ALGOS:
        cfg = TrainConfig(**{**base, "algo": algo})
        print(f"[smoke] training {algo} ...", flush=True)
        run_dir = train(cfg)
        csv_path = run_dir / "episodes.csv"
        if not csv_path.exists():
            failures.append(f"{algo}: missing {csv_path}")
            continue
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            failures.append(f"{algo}: episodes.csv has no data rows ({len(lines)} lines)")
            continue
        print(f"[smoke] {algo} OK ({len(lines) - 1} episodes) -> {run_dir}")

    if failures:
        print("\n[smoke] FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\n[smoke] all four algos completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
