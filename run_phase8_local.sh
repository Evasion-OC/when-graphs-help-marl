#!/usr/bin/env bash
# Phase 8 local runner — sweep -> analysis -> status marker.
# Launched under `nohup caffeinate ...` so it survives sleep + terminal close.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

STATUS="results/phase8/RUN_STATUS.txt"
mkdir -p results/phase8
echo "RUNNING  started=$(date '+%Y-%m-%d %H:%M:%S')  pid=$$" > "$STATUS"

echo "[local-run] === Phase 8 sweep started $(date) ==="
python scripts/phase8_confound_sweep.py \
    --steps 150000 --seeds 5 \
    --algos qmix,mlp_qmix,gnn_qmix,vdn,iql --ns 4,8 \
    --log-dir results/phase8
SWEEP_RC=$?
echo "[local-run] sweep exit code: $SWEEP_RC"

if [ "$SWEEP_RC" -ne 0 ]; then
    echo "FAILED  stage=sweep  rc=$SWEEP_RC  at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
    echo "[local-run] ABORTED at sweep stage"
    exit "$SWEEP_RC"
fi

echo "[local-run] === analysis started $(date) ==="
python scripts/phase8_analysis.py --results results/phase8
ANA_RC=$?
echo "[local-run] analysis exit code: $ANA_RC"

if [ "$ANA_RC" -ne 0 ]; then
    echo "FAILED  stage=analysis  rc=$ANA_RC  at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
    exit "$ANA_RC"
fi

echo "DONE  finished=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
echo "[local-run] === ALL DONE $(date) ==="
