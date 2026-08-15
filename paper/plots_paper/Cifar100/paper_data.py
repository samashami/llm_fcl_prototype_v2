"""Load and aggregate the canonical CIFAR-100 paper-result CSVs."""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "experiments" / "results" / "cifar100" / "paper_results"
OUTPUT_ROOT = Path(__file__).resolve().parent

METHODS = {
    "fixed": "Fixed",
    "v4": "Heuristic V4",
    "lmss": "LMSS (local)",
    "openrouter": "LMSS (OpenRouter)",
}
SEEDS = ("seed42", "seed43", "seed44")
CSV_TYPES = ("summary", "results", "cl_batches")


def validate_canonical_results() -> None:
    """Require exactly one CSV of every canonical type per method and seed."""
    problems = []
    for method in METHODS:
        for seed in SEEDS:
            run_dir = RESULTS_ROOT / method / seed
            for csv_type in CSV_TYPES:
                matches = sorted(run_dir.glob(f"fcl_run_{csv_type}_*.csv"))
                if len(matches) != 1:
                    problems.append(
                        f"{method}/{seed}: expected one fcl_run_{csv_type}_*.csv, "
                        f"found {len(matches)}"
                    )
    if problems:
        raise RuntimeError("Invalid canonical paper-result set:\n" + "\n".join(problems))


def load_summaries() -> pd.DataFrame:
    """Return all canonical per-round summaries with method and seed labels."""
    validate_canonical_results()
    frames = []
    for method in METHODS:
        for seed in SEEDS:
            summary_path = next(
                (RESULTS_ROOT / method / seed).glob("fcl_run_summary_*.csv")
            )
            frame = pd.read_csv(summary_path)
            frame.insert(0, "seed", seed)
            frame.insert(0, "method", method)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def per_round_stats() -> pd.DataFrame:
    """Compute mean and sample standard deviation across the three seeds."""
    summaries = load_summaries()
    metrics = ("global_acc", "forget_mean", "aulc_running", "comm_bytes_cum")
    return (
        summaries.groupby(["method", "round"])[list(metrics)]
        .agg(["mean", "std"])
        .reset_index()
    )


def final_stats() -> pd.DataFrame:
    """Compute final-round paper metrics across seeds."""
    summaries = load_summaries()
    final_rows = (
        summaries.sort_values(["method", "seed", "round"])
        .groupby(["method", "seed"], as_index=False)
        .tail(1)
    )
    metrics = (
        "global_acc",
        "forget_mean",
        "forget_max",
        "aulc_running",
        "comm_bytes_cum",
    )
    return final_rows.groupby("method")[list(metrics)].agg(["mean", "std"])
