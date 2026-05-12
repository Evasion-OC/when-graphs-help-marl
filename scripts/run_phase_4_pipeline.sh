#!/usr/bin/env bash
# Phase 4 overnight pipeline: sweep -> plot -> STATUS file.
#
# Usage (standalone):
#   cd ~/Documents/wgh-marl-phase4
#   nohup bash scripts/run_phase_4_pipeline.sh > /tmp/phase_4.log 2>&1 &
#
# Usage (after Phase 3 finishes — chained):
#   ./scripts/wait_for_phase_3_then_run.sh

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="results/phase_4_main"
STATUS_FILE="$OUT_DIR/STATUS"
mkdir -p "$OUT_DIR"

START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Phase 4 pipeline started at $START_TS" | tee "$STATUS_FILE"

echo ""
echo "=== STAGE 1: Phase 4 sweep ==="
PYTHONPATH=src PYTHONUNBUFFERED=1 python scripts/run_phase_4.py
SWEEP_RC=$?
if [ "$SWEEP_RC" -ne 0 ]; then
    echo "sweep_failed (rc=$SWEEP_RC)" | tee "$STATUS_FILE"
    exit "$SWEEP_RC"
fi

echo ""
echo "=== STAGE 2: plot + summarize ==="
PYTHONPATH=src PYTHONUNBUFFERED=1 python scripts/plot_phase_4.py
PLOT_RC=$?
if [ "$PLOT_RC" -ne 0 ]; then
    echo "plot_failed (rc=$PLOT_RC)" | tee "$STATUS_FILE"
    exit "$PLOT_RC"
fi

END_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
    echo "ok"
    echo "started: $START_TS"
    echo "finished: $END_TS"
} | tee "$STATUS_FILE"
