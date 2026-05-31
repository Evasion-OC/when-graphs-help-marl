#!/usr/bin/env bash
# Phase 11: larger-N scaling of the graph penalty on CoordGrid ring.
# Extends the decisive cells to N in {12,16} (qmix/mlp/gnn, 10 seeds, 150k)
# to show the graph penalty (gnn - mlp) keeps growing with team size:
#   N=4: -10.6  ->  N=8: -86.6  ->  N=12, N=16 (this run).
# WAITS for the Phase 10 GAT job to finish first (CPU is the bottleneck;
# running concurrently would just halve both), then runs sequentially.
#
# Launch detached + caffeinated:
#   nohup caffeinate -dimsu bash run_phase11_scale.sh > phase11_scale.out 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

GATSTATUS=results/phase10_gat/RUN_STATUS.txt
OUT=results/phase11_scale
STATUS="$OUT/RUN_STATUS.txt"
mkdir -p "$OUT"
echo "WAITING for phase10_gat to finish  pid=$$  at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"

# Block until the GAT job reports DONE (abort if it FAILED).
while true; do
    s=$(cat "$GATSTATUS" 2>/dev/null || echo "")
    case "$s" in
        *DONE*)   break ;;
        *FAILED*) echo "ABORT: phase10_gat FAILED ($s)" > "$STATUS"; exit 1 ;;
    esac
    sleep 60
done

echo "RUNNING  started=$(date '+%Y-%m-%d %H:%M:%S')  pid=$$" > "$STATUS"
echo "[phase11] === graph-penalty scaling: CoordGrid ring N in {12,16} ==="
python scripts/phase8_confound_sweep.py \
    --algos qmix,mlp_qmix,gnn_qmix --ns 12,16 --graph ring --seeds 10 --steps 150000 \
    --log-dir "$OUT"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FAILED stage=sweep rc=$RC at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"; exit "$RC"
fi

python scripts/phase8_analysis.py --results "$OUT"
echo "DONE finished=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
echo "[phase11] === ALL DONE ==="
