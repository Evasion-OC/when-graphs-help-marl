#!/usr/bin/env bash
# Phase 8 robustness extension: capacity control on the LINE topology
# (a second graph regime, larger diameter than ring) at n=10. Writes to a
# separate dir so the headline ring results stay pristine.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

OUT=results/phase8_line
STATUS="$OUT/RUN_STATUS.txt"
mkdir -p "$OUT"
echo "RUNNING  started=$(date '+%Y-%m-%d %H:%M:%S')  pid=$$" > "$STATUS"

echo "[line] === capacity control on line topology, n=10 ==="
python scripts/phase8_confound_sweep.py \
    --steps 150000 --seeds 10 --graph line \
    --algos qmix,mlp_qmix,gnn_qmix --ns 4,8 \
    --log-dir "$OUT"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FAILED stage=sweep rc=$RC at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
    exit "$RC"
fi

python scripts/phase8_analysis.py --results "$OUT"
echo "DONE finished=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
echo "[line] === ALL DONE ==="
