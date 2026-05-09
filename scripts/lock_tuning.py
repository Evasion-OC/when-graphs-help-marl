"""Phase 2A — pick the locked hyperparameters per algorithm.

Reads results/phase_2_tune/{algo}/*.csv, computes peak and
final-window success rates averaged across seeds for each (algo, config)
cell, and writes the top configuration per algorithm to:

    configs/phase_2_locked.yaml
    results/phase_2_tune/locked.md   (human-readable summary)

Decision metric: mean **final-window** success rate (last 25% of eval
checkpoints), with mean peak success as a tiebreaker.

Why final-window and not peak: Phase 2A surfaced configs that hit
high peaks but collapse by end of training. Peak is a single best
moment vulnerable to transient noise; final-window averages over many
late checkpoints and is what readers will see in published learning
curves. Phase 2C's H1 test also uses final-window as the headline
metric, so locking on the same criterion avoids selecting on the
training-time noise floor.

Usage (after run_tuning.py finishes):
    python scripts/lock_tuning.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
TUNE = ROOT / "results" / "phase_2_tune"
ALGOS = ("qmix", "gnn_qmix")


def parse_cfg_id(cfg_id: str) -> dict:
    """Inverse of run_tuning.py's cfg_id(). Returns the param dict."""
    parts = cfg_id.split("__")
    out: dict = {}
    for p in parts[1:]:  # parts[0] is the algo name
        m = re.match(r"^([a-z_]+)=(.+)$", p)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if k in {"embed_dim"}:
            out[k] = int(v)
        elif k in {"mixer_lr", "init_scale"}:
            out[k] = float(v)
        else:
            out[k] = v
    return out


def per_seed_metrics(algo_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(algo_dir.glob("*.csv")):
        name = f.stem  # e.g. qmix__embed_dim=8__mixer_lr=0.0001__mixer_init=default_seed0
        m = re.match(r"^(.+?)_seed(\d+)$", name)
        if not m:
            continue
        cfg_id = m.group(1)
        seed = int(m.group(2))
        df = pd.read_csv(f)
        ev = df[df["kind"] == "eval"].copy()
        ev["eval_success_rate"] = pd.to_numeric(ev["eval_success_rate"])
        ev["eval_return_mean"] = pd.to_numeric(ev["eval_return_mean"])
        if ev.empty:
            continue
        peak = float(ev["eval_success_rate"].max())
        late = ev.sort_values("episode").tail(max(1, len(ev) // 4))
        final = float(late["eval_success_rate"].mean())
        final_ret = float(late["eval_return_mean"].mean())
        rows.append(
            {
                "cfg_id": cfg_id,
                "seed": seed,
                "peak_success": peak,
                "final_success": final,
                "final_return": final_ret,
            }
        )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby("cfg_id")
        .agg(
            n_seeds=("seed", "count"),
            peak_mean=("peak_success", "mean"),
            peak_std=("peak_success", "std"),
            final_mean=("final_success", "mean"),
            final_std=("final_success", "std"),
            return_mean=("final_return", "mean"),
        )
        .reset_index()
        .sort_values(["final_mean", "peak_mean"], ascending=False)
    )


def main() -> int:
    if not TUNE.exists():
        print(f"!! {TUNE} not found — run scripts/run_tuning.py first.")
        return 2

    locked: dict[str, dict] = {}
    md_lines = ["# Phase 2A locked hyperparameters", ""]

    for algo in ALGOS:
        algo_dir = TUNE / algo
        if not algo_dir.exists():
            print(f"!! {algo_dir} missing.")
            continue

        per_seed = per_seed_metrics(algo_dir)
        agg = aggregate(per_seed)
        if agg.empty:
            print(f"!! No data for {algo}.")
            continue

        print(f"\n=== {algo.upper()} ===")
        print(agg.to_string(index=False))
        winner = agg.iloc[0]
        params = parse_cfg_id(winner["cfg_id"])
        locked[algo] = params
        md_lines += [
            f"## {algo}",
            "",
            f"**Locked config:** `{winner['cfg_id']}`",
            "",
            "Mean over seeds:",
            f"- peak success: {winner['peak_mean']:.3f} ± {winner['peak_std']:.3f}",
            f"- final-window: {winner['final_mean']:.3f} ± {winner['final_std']:.3f}",
            "",
            "Top 5 configurations by final-window success rate:",
            "",
            "| config | final (mean) | final (std) | peak (mean) |",
            "|---|---|---|---|",
        ]
        for _, r in agg.head(5).iterrows():
            md_lines.append(
                f"| `{r['cfg_id']}` | {r['final_mean']:.3f} | {r['final_std']:.3f} | {r['peak_mean']:.3f} |"
            )
        md_lines += ["", ""]

    # IQL and VDN: locked at Phase 1 defaults (not part of tuning sweep).
    locked["iql"] = {}
    locked["vdn"] = {}
    md_lines += [
        "## iql, vdn",
        "",
        "Not tuned in Phase 2A (Phase 1 confirmed defaults work). Lock to:",
        "",
        "- `lr=5e-4`, `hidden=64` (PyTorch defaults), 30k steps in Phase 2C",
        "",
    ]

    # Write the YAML
    out_yaml = ROOT / "configs" / "phase_2_locked.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(locked, out_yaml.open("w"), sort_keys=False)
    print(f"\nLocked hyperparameters → {out_yaml}")

    out_md = TUNE / "locked.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Summary → {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
