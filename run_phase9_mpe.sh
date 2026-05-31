#!/usr/bin/env bash
# Phase 9: external-validity replication on MPE simple_spread.
# Same algos, same locked hyperparameters, same 150k-step / 10-seed
# protocol as Phase 8 -- only the environment changes (CoordGrid -> MPE).
# 5 algos x N in {3,6} x 10 seeds = 100 runs, ~4 h on M-series CPU.
#
# Launch detached + caffeinated so it survives logout / display sleep:
#   nohup caffeinate -dimsu bash run_phase9_mpe.sh > phase9_mpe.out 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

OUT=results/phase9_mpe
STATUS="$OUT/RUN_STATUS.txt"
mkdir -p "$OUT"
echo "RUNNING  started=$(date '+%Y-%m-%d %H:%M:%S')  pid=$$" > "$STATUS"

echo "[phase9] === MPE simple_spread external-validity replication ==="
python scripts/phase9_mpe_sweep.py \
    --steps 150000 --seeds 10 \
    --algos qmix,mlp_qmix,gnn_qmix,vdn,iql --ns 3,6 \
    --k 2 --log-dir "$OUT"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FAILED stage=sweep rc=$RC at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
    exit "$RC"
fi

python scripts/phase9_mpe_analysis.py --results "$OUT"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FAILED stage=analysis rc=$RC at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
    exit "$RC"
fi

echo "DONE finished=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
echo "[phase9] === ALL DONE ==="
