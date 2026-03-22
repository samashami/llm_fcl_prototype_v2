#!/usr/bin/env python
"""
Aggregate CIFAR-100 FCL results across seeds for:
- baseline (fixed)
- controller_v4 (heuristic)
- controller_sft (LLM-guided)

Folder structure expected:

cifar100/
  baseline/
    s42/
    s43/
    s44/
  controller_v4/
    s42/
    s43/
    s44/
  controller_sft/
    s42/
    s43/
    s44/
  aggregated/   <-- will be created here
"""

import pandas as pd
from pathlib import Path

# Since this script is INSIDE the cifar100 folder:
BASE = Path(__file__).resolve().parent

CONTROLLERS = {
    "baseline": "baseline_20251120",          # fixed controller
    "v4": "controller_v4_20251120",           # heuristic v4 controller
    "sft": "controller_sft",         # LLM-guided controller
}

def load_all_summary_rows():
    rows = []

    for ctrl_name, folder_name in CONTROLLERS.items():
        ctrl_dir = BASE / folder_name
        if not ctrl_dir.exists():
            print(f"WARNING: controller folder missing: {ctrl_dir}")
            continue

        for seed_dir in sorted(ctrl_dir.glob("s*")):
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name  # "s42"

            summary_files = list(seed_dir.glob("fcl_run_summary_*.csv"))
            if not summary_files:
                print(f"WARNING: no summary CSV in {seed_dir}")
                continue
            summary_path = summary_files[0]

            df = pd.read_csv(summary_path)

            for _, r in df.iterrows():
                rows.append({
                    "controller": ctrl_name,
                    "seed": seed,
                    "round": int(r["round"]),
                    "global_acc": r["global_acc"],
                    "forget_mean": r.get("forget_mean", float("nan")),
                    "forget_max": r.get("forget_max", float("nan")),
                    "aulc_running": r.get("aulc_running", float("nan")),
                    "comm_bytes_round": r.get("comm_bytes_round", float("nan")),
                    "comm_bytes_cum": r.get("comm_bytes_cum", float("nan")),
                })

    if not rows:
        raise RuntimeError("No rows loaded – check your paths and CSVs.")
    return pd.DataFrame(rows)


def main():
    df_all = load_all_summary_rows()

    out_dir = BASE / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Per-round aggregated stats ===
    per_round_stats = (
        df_all
        .groupby(["controller", "round"])
        .agg(
            acc_mean=("global_acc", "mean"),
            acc_std=("global_acc", "std"),
            forget_mean_mean=("forget_mean", "mean"),
            forget_mean_std=("forget_mean", "std"),
            forget_max_mean=("forget_max", "mean"),
            forget_max_std=("forget_max", "std"),
            aulc_mean=("aulc_running", "mean"),
            aulc_std=("aulc_running", "std"),
            comm_round_mean=("comm_bytes_round", "mean"),
            comm_round_std=("comm_bytes_round", "std"),
            comm_cum_mean=("comm_bytes_cum", "mean"),
            comm_cum_std=("comm_bytes_cum", "std"),
        )
        .reset_index()
        .sort_values(["controller", "round"])
    )

    per_round_stats.to_csv(out_dir / "cifar100_per_round_stats.csv", index=False)

    # === Final stats (one per seed, last round) ===
    last_round_rows = (
        df_all
        .sort_values(["controller", "seed", "round"])
        .groupby(["controller", "seed"])
        .tail(1)
    )

    final_stats = (
        last_round_rows
        .groupby("controller")
        .agg(
            acc_mean=("global_acc", "mean"),
            acc_std=("global_acc", "std"),
            forget_mean_mean=("forget_mean", "mean"),
            forget_mean_std=("forget_mean", "std"),
            forget_max_mean=("forget_max", "mean"),
            forget_max_std=("forget_max", "std"),
            aulc_mean=("aulc_running", "mean"),
            aulc_std=("aulc_running", "std"),
            comm_cum_mean=("comm_bytes_cum", "mean"),
            comm_cum_std=("comm_bytes_cum", "std"),
        )
        .reset_index()
        .sort_values("controller")
    )

    final_stats.to_csv(out_dir / "cifar100_final_stats.csv", index=False)

    print("Wrote aggregated CSVs:")
    print("  - cifar100_per_round_stats.csv")
    print("  - cifar100_final_stats.csv")


if __name__ == "__main__":
    main()