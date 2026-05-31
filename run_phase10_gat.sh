#!/usr/bin/env bash
# Phase 10: second GNN family (GAT-QMIX) robustness check.
# Runs ONLY gat_qmix on the decisive cells, into the SAME dirs as the
# existing qmix/mlp/gnn runs (additive -- new gat_qmix__* subdirs only),
# so GAT is directly comparable. Same locked config, 150k steps, 10 seeds.
#   CoordGrid ring N in {4,8}  -> results/phase8     (vs Phase 8 qmix/mlp/gnn)
#   MPE simple_spread N in {3,6} -> results/phase9_mpe (vs Phase 9 qmix/mlp/gnn)
#
# Launch detached + caffeinated:
#   nohup caffeinate -dimsu bash run_phase10_gat.sh > phase10_gat.out 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

OUT_CG=results/phase8
OUT_MPE=results/phase9_mpe
STATUSDIR=results/phase10_gat
STATUS="$STATUSDIR/RUN_STATUS.txt"
mkdir -p "$STATUSDIR"
echo "RUNNING  started=$(date '+%Y-%m-%d %H:%M:%S')  pid=$$" > "$STATUS"

echo "[phase10] === GAT-QMIX on CoordGrid ring N in {4,8} ==="
python scripts/phase8_confound_sweep.py \
    --algos gat_qmix --ns 4,8 --graph ring --seeds 10 --steps 150000 \
    --log-dir "$OUT_CG"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FAILED stage=coordgrid rc=$RC at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"; exit "$RC"
fi

echo "[phase10] === GAT-QMIX on MPE simple_spread N in {3,6} ==="
python scripts/phase9_mpe_sweep.py \
    --algos gat_qmix --ns 3,6 --seeds 10 --steps 150000 --k 2 \
    --log-dir "$OUT_MPE"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "FAILED stage=mpe rc=$RC at=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"; exit "$RC"
fi

echo "[phase10] === analyses ==="
python scripts/phase10_gat_analysis.py --results "$OUT_CG"  --tag coordgrid_ring
python scripts/phase10_gat_analysis.py --results "$OUT_MPE" --tag mpe_spread

echo "DONE finished=$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
echo "[phase10] === ALL DONE ==="
