#!/usr/bin/env bash
# Wait for Phase 3 to write its STATUS file, then kick off Phase 4.
#
# Polls every 5 minutes for the Phase 3 STATUS file. When it appears with
# 'ok', launches the Phase 4 pipeline. If Phase 3 ends in a failure
# state, refuses to launch Phase 4 (better to investigate first).
#
# Usage:
#   cd ~/Documents/wgh-marl-phase4
#   nohup bash scripts/wait_for_phase_3_then_run.sh > /tmp/phase_4_chain.log 2>&1 &

set -u

PHASE_3_STATUS="$HOME/Documents/wgh-marl-phase3/results/phase_3_mpe/STATUS"
CHAIN_LOG="/tmp/phase_4.log"
WAIT_INTERVAL=300  # 5 minutes between checks

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Waiting for Phase 3 to finish before launching Phase 4..."
echo "  Watching: $PHASE_3_STATUS"
echo "  Poll interval: ${WAIT_INTERVAL}s"
echo ""

while true; do
    if [ -f "$PHASE_3_STATUS" ]; then
        FIRST_LINE="$(head -n1 "$PHASE_3_STATUS" 2>/dev/null || echo '?')"
        if [ "$FIRST_LINE" = "ok" ]; then
            echo "[$(date -u +%FT%TZ)] Phase 3 STATUS = ok. Launching Phase 4."
            break
        elif echo "$FIRST_LINE" | grep -q "_failed"; then
            echo "[$(date -u +%FT%TZ)] Phase 3 STATUS = $FIRST_LINE. Refusing to launch Phase 4."
            exit 1
        else
            # STATUS file exists but still says "started at..." — Phase 3 still
            # running. (The pipeline overwrites this with 'ok' on completion.)
            :
        fi
    fi
    sleep "$WAIT_INTERVAL"
done

echo ""
echo "=== Launching Phase 4 pipeline ==="
cd "$ROOT"
exec bash scripts/run_phase_4_pipeline.sh
