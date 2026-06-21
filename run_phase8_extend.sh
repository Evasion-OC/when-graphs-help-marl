#!/usr/bin/env bash
# Phase 8 extension — add seeds 5..9 (n=5 -> n=10) to the existing run,
# then re-run analysis over the full n=10 set. Same env/config as the
# original sweep; writes into the same results/phase8 dir.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

STATUS="results/phase8/RUN_STATUS.txt"
echo "RUNNING_EXTENSION  started=$(date '+%Y-%m-%d %H:%M:%S')  pid=$$" > "$STATUS"

echo "[extend] === seeds 5..9 started $(date) ==="
python scripts/phase8_confound_sweep.py \
    --steps 150000 --seeds 5 --seed-start 5 \
    --algos qmix,mlp_qmix,gnn_qmix,vdn,iql --ns 4,8 \
    --log-dir results/phase8
RC=$?
echo "[extend] sweep exit code: $RC"
if [ "$RC" -ne 0 ]; then
    echo "FAILED  stage=extend-sweep  rc=$RC  at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
    exit "$RC"
fi

echo "[extend] === analysis over full n=10 set $(date) ==="
python scripts/phase8_analysis.py --results results/phase8
ARC=$?
if [ "$ARC" -ne 0 ]; then
    echo "FAILED  stage=analysis  rc=$ARC  at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
    exit "$ARC"
fi

echo "DONE_N10  finished=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
echo "[extend] === ALL DONE (n=10) $(date) ==="
