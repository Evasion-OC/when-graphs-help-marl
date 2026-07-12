"""Learning-curve evidence for the F1 (150k matched-budget) attention check.

Bins each run's per-episode return into 5 equal training-progress quintiles and
averages over the 12 seeds. Shows the complete-graph attention arms sit at the
random floor for the ENTIRE 150k budget while the true-graph arms (GCN/GAT/DGN)
climb steeply -- i.e. the complete-arm floor is structural, not under-training.

Usage: python scripts/phaseC_150k_trajectory.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

BASE = "results/phaseC_150k/pair_N6_ego_attention"
ARMS = ["gnn_true", "gat_true", "dgn_true", "gat_complete", "dgn_complete",
        "gcn_complete", "mlp"]
NQ = 5


def main() -> int:
    rows = []
    for a in ARMS:
        dirs = sorted(glob.glob(f"{BASE}/{a}__pair_N6_ego_attention__seed*"))
        per = []
        for d in dirs:
            r = pd.read_csv(os.path.join(d, "episodes.csv"))["return"].to_numpy(float)
            per.append([c.mean() for c in np.array_split(r, NQ)])
        m = np.mean(per, axis=0)
        rows.append({"arm": a, "n_seeds": len(dirs),
                     **{f"Q{i+1}": round(float(m[i]), 2) for i in range(NQ)}})
    df = pd.DataFrame(rows)
    out = "results/phaseC_150k/trajectory_quintiles.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
