#!/usr/bin/env bash
# Phase 2C overnight pipeline: sweep -> plots -> H1 stats -> status file.
#
# Designed to run detached from an interactive shell:
#
#     cd ~/Documents/when-graphs-help-marl
#     nohup bash scripts/run_phase_2c_pipeline.sh > /tmp/phase_2c.log 2>&1 &
#
# When the pipeline finishes, results/phase_2_main/STATUS will contain one
# of: 'sweep_failed', 'plot_failed', 'h1_failed', or 'ok'. A non-zero exit
# code from any stage halts the chain.

set -u  # treat unset variables as errors; do NOT set -e (we want explicit
        # status reporting per stage)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="results/phase_2_main"
STATUS_FILE="$OUT_DIR/STATUS"
mkdir -p "$OUT_DIR"

START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Phase 2C pipeline started at $START_TS" | tee "$STATUS_FILE"
echo "  PID=$$"
echo "  ROOT=$ROOT"

# ---- 1. Main sweep --------------------------------------------------------
echo ""
echo "=== STAGE 1: main sweep ==="
PYTHONUNBUFFERED=1 python scripts/run_main_sweep.py
SWEEP_RC=$?
if [ "$SWEEP_RC" -ne 0 ]; then
    echo "sweep_failed (rc=$SWEEP_RC)" | tee "$STATUS_FILE"
    echo "FAILED at sweep stage; aborting"
    exit "$SWEEP_RC"
fi
echo "  sweep ok"

# ---- 2. Plot --------------------------------------------------------------
echo ""
echo "=== STAGE 2: plotting ==="
PYTHONUNBUFFERED=1 python scripts/plot_main_sweep.py
PLOT_RC=$?
if [ "$PLOT_RC" -ne 0 ]; then
    echo "plot_failed (rc=$PLOT_RC)" | tee "$STATUS_FILE"
    exit "$PLOT_RC"
fi
echo "  plot ok"

# ---- 3. H1 statistical tests ---------------------------------------------
echo ""
echo "=== STAGE 3: H1 tests ==="
PYTHONUNBUFFERED=1 python scripts/h1_test.py
H1_RC=$?
if [ "$H1_RC" -ne 0 ]; then
    echo "h1_failed (rc=$H1_RC)" | tee "$STATUS_FILE"
    exit "$H1_RC"
fi
echo "  H1 tests ok"

# ---- Done -----------------------------------------------------------------
END_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "=== ALL STAGES OK ==="
{
    echo "ok"
    echo "started: $START_TS"
    echo "finished: $END_TS"
    echo ""
    echo "Outputs:"
    ls -la "$OUT_DIR" | sed 's/^/  /'
} | tee "$STATUS_FILE"
