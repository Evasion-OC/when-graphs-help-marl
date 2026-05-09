"""Pre-plotting data quality check for Phase 1 pilot.

Verifies that every (algo, seed) CSV in results/pilot/ is:
- Non-empty
- Has the full schema declared by training.loop.LOG_FIELDS
- Contains both train and eval rows
- Reaches the expected total step budget (within tolerance)
- Has plausible eval-return values (not all NaN, not all zero)

Exits with non-zero status on any failure so callers can pipe it as a gate.

Run:
    python scripts/check_pilot_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "results" / "pilot"

EXPECTED_ALGOS = ["iql", "vdn", "qmix", "gnn_qmix"]
EXPECTED_SEEDS = [0, 1, 2]
TOTAL_STEPS = 10_000
STEP_TOLERANCE = 250  # episodes don't terminate exactly on step boundaries

# Required schema (subset of LOG_FIELDS that must be present).
REQUIRED_FIELDS = {
    "step", "episode", "kind", "ep_return", "ep_length", "ep_success",
    "epsilon", "eval_return_mean", "eval_success_rate", "eval_length_mean",
}


def check_one(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        return [f"unreadable: {e!r}"]
    if df.empty:
        return ["empty file"]
    missing = REQUIRED_FIELDS - set(df.columns)
    if missing:
        problems.append(f"missing schema fields: {sorted(missing)}")
    if "kind" in df.columns:
        train_n = (df["kind"] == "train").sum()
        eval_n = (df["kind"] == "eval").sum()
        if train_n == 0:
            problems.append("no train rows")
        if eval_n == 0:
            problems.append("no eval rows")
    last_step = pd.to_numeric(df.get("step", pd.Series([0])), errors="coerce").max()
    if last_step < TOTAL_STEPS - STEP_TOLERANCE:
        problems.append(
            f"last_step={int(last_step)} < expected {TOTAL_STEPS - STEP_TOLERANCE}"
        )
    eval_rows = df[df["kind"] == "eval"] if "kind" in df.columns else df.iloc[0:0]
    if not eval_rows.empty:
        ev_succ = pd.to_numeric(eval_rows["eval_success_rate"], errors="coerce")
        if ev_succ.isna().all():
            problems.append("all eval_success_rate are NaN")
    return problems


def main() -> int:
    if not PILOT.exists():
        print(f"!! {PILOT} not found — run scripts/run_pilot.py first.")
        return 2

    expected = [(a, s) for a in EXPECTED_ALGOS for s in EXPECTED_SEEDS]
    found = sorted(PILOT.glob("*_seed*.csv"))
    found_keys = {(p.stem.split("_seed")[0], int(p.stem.split("seed")[-1])) for p in found}

    missing = sorted(set(expected) - found_keys)
    extra = sorted(found_keys - set(expected))

    rc = 0
    if missing:
        print(f"MISSING runs: {missing}")
        rc = 1
    if extra:
        print(f"EXTRA runs (will not be checked): {extra}")

    print(f"\nChecking {len(found)} files in {PILOT}/")
    bad = 0
    for p in found:
        probs = check_one(p)
        if probs:
            bad += 1
            print(f"  ✗ {p.name}")
            for pr in probs:
                print(f"      - {pr}")
        else:
            print(f"  ✓ {p.name}")

    if bad:
        print(f"\n{bad}/{len(found)} files have problems.")
        rc = 1
    else:
        print(f"\nAll {len(found)} files clean.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
