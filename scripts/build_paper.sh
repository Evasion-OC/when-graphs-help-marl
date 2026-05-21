#!/usr/bin/env bash
# Full paper-build pipeline.
#
# Runs all analyses for phases that have data, regenerates the LaTeX inserts,
# and compiles paper/main.pdf. Idempotent — safe to re-run anytime.
#
# Requires the experiments (scripts/phase{1,2,4}_*) to have produced
# CSVs under results/.
set -e
export PYTHONIOENCODING=utf-8

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ran_any=0
if [ -d "results/phase1" ] && [ "$(ls results/phase1/*/episodes.csv 2>/dev/null | wc -l)" -ge 1 ]; then
  echo "==> phase 1 analysis"
  python scripts/phase1_analysis.py
  ran_any=1
fi
if [ -d "results/phase2" ] && [ "$(ls results/phase2/*/episodes.csv 2>/dev/null | wc -l)" -ge 1 ]; then
  echo "==> phase 2 analysis"
  python scripts/phase2_analysis.py
  ran_any=1
fi
if [ -d "results/phase4" ] && [ "$(ls results/phase4/*/episodes.csv 2>/dev/null | wc -l)" -ge 1 ]; then
  echo "==> phase 4 analysis"
  python scripts/phase4_analysis.py
  ran_any=1
fi

echo "==> building paper inserts"
python scripts/build_paper_inserts.py

echo "==> compiling paper"
cd paper
pdflatex -interaction=nonstopmode main.tex > /tmp/lp.txt 2>&1 || { tail -20 /tmp/lp.txt; exit 1; }
bibtex main > /tmp/bib.txt 2>&1 || { cat /tmp/bib.txt; exit 1; }
pdflatex -interaction=nonstopmode main.tex > /tmp/lp.txt 2>&1 || { tail -20 /tmp/lp.txt; exit 1; }
pdflatex -interaction=nonstopmode main.tex > /tmp/lp.txt 2>&1 || { tail -20 /tmp/lp.txt; exit 1; }

pages=$(grep -oP 'Output written.*\(\K[0-9]+' /tmp/lp.txt | head -1)
echo "==> paper/main.pdf built ($pages pages)"
