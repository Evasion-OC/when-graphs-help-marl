#!/usr/bin/env bash
# Phase 12: DGN-style multi-head graph attention robustness check.
# Runs ONLY dgn_qmix on the decisive cells, into the SAME dirs as the
# existing qmix/mlp/gnn/gat runs (additive -- new dgn_qmix__* subdirs only),
# so it is directly comparable. Same locked config, 150k steps, 10 seeds.
#   CoordGrid ring N in {4,8}   -> results/phase8
#   MPE simple_spread N in {3,6} -> results/phase9_mpe
#
# Launch detached + caffeinated:
#   nohup caffeinate -dimsu bash run_phase12_dgn.sh > phase12_dgn.out 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

OUT_CG=results/phase8
OUT_MPE=results/phase9_mpe
STATUSDIR=results/phase12_dgn
STATUS="$STATUSDIR/RUN_STATUS.txt"
mkdir -p "$STATUSDIR"
echo "RUNNING  started=$(date '+%Y-%m-%d %H:%M:%S')  pid=$$" > "$STATUS"

echo "[phase12] === DGN-QMIX on CoordGrid ring N in {4,8} ==="
python scripts/phase8_confound_sweep.py \
    --algos dgn_qmix --ns 4,8 --graph ring --seeds 10 --steps 150000 \
    --log-dir "$OUT_CG"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FAILED stage=coordgrid rc=$RC at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"; exit "$RC"
fi

echo "[phase12] === DGN-QMIX on MPE simple_spread N in {3,6} ==="
python scripts/phase9_mpe_sweep.py \
    --algos dgn_qmix --ns 3,6 --seeds 10 --steps 150000 --k 2 \
    --log-dir "$OUT_MPE"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FAILED stage=mpe rc=$RC at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"; exit "$RC"
fi

echo "[phase12] === analyses ==="
python scripts/phase12_dgn_analysis.py --results "$OUT_CG"  --tag coordgrid_ring
python scripts/phase12_dgn_analysis.py --results "$OUT_MPE" --tag mpe_spread

echo "DONE finished=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
echo "[phase12] === ALL DONE ==="
