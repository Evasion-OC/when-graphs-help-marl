"""Summarize results from `scripts/tune_mixer_sanity.py`.

Reads results/pilot_tune/{algo}_emb{N}.csv and reports:
- peak success rate
- final-window success rate
- late-training q_tot magnitude (the divergence indicator from Phase 1)

Then makes a one-line verdict on whether shrinking embed_dim fixed the
mixer-family algorithms.

Run after `scripts/tune_mixer_sanity.py`:
    python scripts/check_diagnostic.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TUNE = ROOT / "results" / "pilot_tune"


def summarize(path: Path) -> dict:
    df = pd.read_csv(path)
    ev = df[df["kind"] == "eval"].copy()
    ev["eval_success_rate"] = pd.to_numeric(ev["eval_success_rate"])
    train = df[df["kind"] == "train"].copy()
    train["q_tot_mean"] = pd.to_numeric(train["q_tot_mean"], errors="coerce")
    train["loss"] = pd.to_numeric(train["loss"], errors="coerce")
    n = len(train)
    late_q_tot = train.tail(max(1, n // 5))["q_tot_mean"].mean()
    late_loss = train.tail(max(1, n // 5))["loss"].mean()
    peak = ev["eval_success_rate"].max() if not ev.empty else float("nan")
    final = ev.tail(max(1, len(ev) // 4))["eval_success_rate"].mean() if not ev.empty else float("nan")
    return {
        "file": path.name,
        "peak": peak,
        "final": final,
        "late_q_tot": late_q_tot,
        "late_loss": late_loss,
    }


def main() -> int:
    if not TUNE.exists():
        print(f"!! {TUNE} not found — run scripts/tune_mixer_sanity.py first.")
        return 2

    rows = [summarize(p) for p in sorted(TUNE.glob("*.csv"))]
    if not rows:
        print(f"No CSVs in {TUNE}")
        return 1
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

    # Verdict logic
    qmix8 = df[df["file"] == "qmix_emb8.csv"]
    qmix16 = df[df["file"] == "qmix_emb16.csv"]
    gnn8 = df[df["file"] == "gnn_qmix_emb8.csv"]
    gnn16 = df[df["file"] == "gnn_qmix_emb16.csv"]

    threshold = 0.30  # peak success of "actually learned"

    def fmt(label, sub):
        if sub.empty:
            return f"  {label:<22}  (no data)"
        peak = sub["peak"].iloc[0]
        verdict = "✓ learned" if peak >= threshold else "✗ did not learn"
        return f"  {label:<22}  peak={peak:.3f}  q_tot={sub['late_q_tot'].iloc[0]:+.2f}  → {verdict}"

    print("Verdict (threshold: peak success ≥ 0.30 = 'learned')")
    print(fmt("qmix     embed_dim=8",  qmix8))
    print(fmt("qmix     embed_dim=16", qmix16))
    print(fmt("gnn_qmix embed_dim=8",  gnn8))
    print(fmt("gnn_qmix embed_dim=16", gnn16))
    print()
    print("Pilot baseline (Phase 1, embed_dim=32, seed 0):")
    print("  qmix     embed_dim=32  peak=0.050  q_tot=+12.44  → ✗ did not learn")
    print("  gnn_qmix embed_dim=32  peak=0.100  q_tot=+4.12   → ✗ did not learn")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
