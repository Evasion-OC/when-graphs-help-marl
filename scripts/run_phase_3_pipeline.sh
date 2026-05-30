#!/usr/bin/env bash
# Phase 3 overnight pipeline: sweep -> plot -> status file.
#
# Usage:
#   cd ~/Documents/wgh-marl-phase3
#   nohup bash scripts/run_phase_3_pipeline.sh > /tmp/phase_3.log 2>&1 &
#
# When the pipeline finishes, results/phase_3_mpe/STATUS contains one of:
# 'sweep_failed', 'plot_failed', or 'ok'.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="results/phase_3_mpe"
STATUS_FILE="$OUT_DIR/STATUS"
mkdir -p "$OUT_DIR"

START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Phase 3 pipeline started at $START_TS" | tee "$STATUS_FILE"

# ---- 1. Main sweep --------------------------------------------------------
echo ""
echo "=== STAGE 1: Phase 3 sweep ==="
PYTHONPATH=src PYTHONUNBUFFERED=1 python scripts/run_phase_3.py
SWEEP_RC=$?
if [ "$SWEEP_RC" -ne 0 ]; then
    echo "sweep_failed (rc=$SWEEP_RC)" | tee "$STATUS_FILE"
    exit "$SWEEP_RC"
fi
echo "  sweep ok"

# ---- 2. Plotting + summary ----------------------------------------------
echo ""
echo "=== STAGE 2: plot + summarize ==="
PYTHONPATH=src PYTHONUNBUFFERED=1 python scripts/plot_phase_3.py
PLOT_RC=$?
if [ "$PLOT_RC" -ne 0 ]; then
    echo "plot_failed (rc=$PLOT_RC)" | tee "$STATUS_FILE"
    exit "$PLOT_RC"
fi
echo "  plot ok"

# ---- Done -----------------------------------------------------------------
END_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "=== ALL STAGES OK ==="
{
    echo "ok"
    echo "started: $START_TS"
    echo "finished: $END_TS"
} | tee "$STATUS_FILE"
