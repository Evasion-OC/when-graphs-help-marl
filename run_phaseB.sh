#!/usr/bin/env bash
# Stage B: run the diameter sweep in parallel, then analyse H5.
# Usage:  ./run_phaseB.sh [SEEDS] [WORKERS]
# Example (16-core desktop):  ./run_phaseB.sh 10 16
set -e
SEEDS="${1:-10}"
WORKERS="${2:-$(python -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')}"

cd "$(dirname "$0")"
echo "==> Stage B sweep: seeds=$SEEDS workers=$WORKERS (CPU; the GPU does not help here)"
python scripts/phaseB_diameter_sweep.py --seeds "$SEEDS" --workers "$WORKERS"
echo "==> Stage B analysis (H5)"
python scripts/phaseB_analysis.py
