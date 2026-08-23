"""Analyze the seed-42 LMSS-Geo Stage 1 fixed-lambda sweep."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
LMSS_GEO_ROOT = (
    REPO_ROOT / "experiments" / "results" / "cifar100" / "lmss_geo"
)
STAGE1_ROOT = LMSS_GEO_ROOT / "stage1_fixed_lambda"
OUTPUT_ROOT = STAGE1_ROOT / "analysis"

LAMBDA_RUNS = {
    0.0: LMSS_GEO_ROOT / "stage0_measurement" / "dirichlet_a01" / "seed42",
    0.25: STAGE1_ROOT / "dirichlet_a01" / "lambda025" / "seed42",
    0.5: STAGE1_ROOT / "dirichlet_a01" / "lambda050" / "seed42",
    0.75: STAGE1_ROOT / "dirichlet_a01" / "lambda075" / "seed42",
    1.0: STAGE1_ROOT / "dirichlet_a01" / "lambda100" / "seed42",
}

COLUMN_MAP = {
    "round": "round",
    "accuracy": "global_acc",
    "forgetting": "forget_mean",
    "aulc": "aulc_running",
    "divergence": "divergence",
    "beta_hat": "beta_hat",
    "rho_hat": "rho_hat",
    "subspace_rank": "protected_basis_rank",
    "measurement_overhead_seconds": "measurement_overhead_seconds",
}

STYLES = {
    0.0: {"color": "#333333", "marker": "o", "linestyle": "--"},
    0.25: {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
    0.5: {"color": "#2ca02c", "marker": "D", "linestyle": "-"},
    0.75: {"color": "#ff7f0e", "marker": "v", "linestyle": "-"},
    1.0: {"color": "#d62728", "marker": "^", "linestyle": "-"},
}


def find_summary(run_directory: Path) -> Path:
    matches = sorted(run_directory.glob("fcl_run_summary_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one summary CSV in {run_directory}, found {len(matches)}"
        )
    return matches[0]


def load_comparison() -> pd.DataFrame:
    frames = []
    expected_rounds = None
    required = set(COLUMN_MAP.values())
    for lambda_value, run_directory in LAMBDA_RUNS.items():
        source = find_summary(run_directory)
        raw = pd.read_csv(source)
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise RuntimeError(f"{source} is missing required columns: {missing}")
        if raw["round"].duplicated().any():
            raise RuntimeError(f"{source} contains duplicate round values")

        rounds = tuple(raw["round"].tolist())
        if expected_rounds is None:
            expected_rounds = rounds
        elif rounds != expected_rounds:
            raise RuntimeError(f"{source} does not contain the same rounds as the baseline")

        frame = raw[list(COLUMN_MAP.values())].rename(
            columns={source_name: name for name, source_name in COLUMN_MAP.items()}
        )
        frame.insert(0, "lambda", lambda_value)
        frame["source_file"] = source.relative_to(REPO_ROOT).as_posix()
        # Projection cannot act before the first protected basis exists.
        frame.loc[frame["round"] == 0, ["beta_hat", "rho_hat"]] = np.nan
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def final_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lambda_value in LAMBDA_RUNS:
        data = comparison[comparison["lambda"] == lambda_value].sort_values("round")
        final = data.iloc[-1]
        rows.append(
            {
                "lambda": lambda_value,
                "final_accuracy": final["accuracy"],
                "final_forgetting": final["forgetting"],
                "final_aulc": final["aulc"],
                "final_divergence": final["divergence"],
                "final_beta_hat": final["beta_hat"],
                "final_rho_hat": final["rho_hat"],
                "average_measurement_overhead_seconds": data[
                    "measurement_overhead_seconds"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_metric(
    comparison: pd.DataFrame,
    column: str,
    ylabel: str,
    filename: str,
    *,
    percent: bool = False,
    lower_zero: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    axis.set_facecolor("white")
    for lambda_value in LAMBDA_RUNS:
        data = comparison[comparison["lambda"] == lambda_value].sort_values("round")
        values = data[column] * 100 if percent else data[column]
        axis.plot(
            data["round"],
            values,
            label=f"λ={lambda_value:g}",
            linewidth=1.9,
            markersize=4.5,
            **STYLES[lambda_value],
        )

    axis.set_xlabel("Round")
    axis.set_ylabel(ylabel)
    axis.set_xticks(sorted(comparison["round"].unique()))
    if lower_zero:
        axis.set_ylim(bottom=0.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def monotonic_description(values: pd.Series) -> str:
    differences = np.diff(values.to_numpy(dtype=float))
    if np.all(differences >= 0.0):
        return "non-decreasing"
    if np.all(differences <= 0.0):
        return "non-increasing"
    return "non-monotonic"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison = load_comparison()
    final = final_metrics(comparison)

    comparison.to_csv(OUTPUT_ROOT / "stage1_round_comparison.csv", index=False)
    final.to_csv(OUTPUT_ROOT / "stage1_final_metrics.csv", index=False)

    plot_metric(
        comparison,
        "accuracy",
        "Global accuracy (%)",
        "stage1_accuracy_vs_round.png",
        percent=True,
    )
    plot_metric(
        comparison,
        "forgetting",
        "Mean forgetting (%)",
        "stage1_forgetting_vs_round.png",
        percent=True,
        lower_zero=True,
    )
    plot_metric(
        comparison,
        "aulc",
        "Running AULC (%)",
        "stage1_aulc_vs_round.png",
        percent=True,
    )
    plot_metric(
        comparison,
        "divergence",
        "Client divergence",
        "stage1_divergence_vs_round.png",
    )
    plot_metric(
        comparison,
        "beta_hat",
        "β̂ (gradient-energy fraction inside protected subspace)",
        "stage1_beta_hat_vs_round.png",
    )
    plot_metric(
        comparison,
        "rho_hat",
        "ρ̂ (relative gradient residual)",
        "stage1_rho_hat_vs_round.png",
    )

    print("Confirmed source-column mapping:")
    for analysis_name, source_name in COLUMN_MAP.items():
        print(f"  {analysis_name}: {source_name}")
    print("\nFinal metrics:")
    print(final.to_string(index=False))
    print("\nFinal-metric response directions as lambda increases:")
    for column in (
        "final_accuracy",
        "final_forgetting",
        "final_aulc",
        "final_divergence",
        "final_beta_hat",
        "final_rho_hat",
    ):
        print(f"  {column}: {monotonic_description(final[column])}")
    print("\nBest fixed lambda by requested objective:")
    objectives = {
        "final accuracy": ("final_accuracy", "max"),
        "lowest forgetting": ("final_forgetting", "min"),
        "highest AULC": ("final_aulc", "max"),
        "lowest divergence": ("final_divergence", "min"),
    }
    for label, (column, direction) in objectives.items():
        index = final[column].idxmax() if direction == "max" else final[column].idxmin()
        print(f"  {label}: lambda={final.loc[index, 'lambda']:g}")


if __name__ == "__main__":
    main()
