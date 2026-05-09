"""Run the smoke-test config: every algorithm, short training, results to CSV.

Usage: python scripts/smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gnnmarl.train import Args, main  # noqa: E402


def run_smoke(config_path: str = "configs/coord_grid_smoke.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    defaults = cfg["defaults"]
    runs = cfg["runs"]
    print(f"Smoke test: {len(runs)} runs from {config_path}")
    for i, run in enumerate(runs):
        merged = {**defaults, **run}
        # Drop keys not in Args (yaml may carry extras)
        accepted = {k: v for k, v in merged.items() if k in Args.__annotations__}
        args = Args(**accepted)
        print(f"\n=== [{i+1}/{len(runs)}] {args.algo} ===")
        t0 = time.time()
        path = main(args)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s -> {path}")


if __name__ == "__main__":
    run_smoke()
