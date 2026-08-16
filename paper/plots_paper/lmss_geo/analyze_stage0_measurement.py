"""Compare the three seed-42 LMSS-Geo Stage 0 measurement protocols."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE0_ROOT = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "cifar100"
    / "lmss_geo"
    / "stage0_measurement"
)
OUTPUT_ROOT = STAGE0_ROOT / "analysis"

PROTOCOLS = {
    "equal": ("equal_split", "Equal"),
    "dirichlet_alpha_0.5": ("dirichlet_a05", "Dirichlet α=0.5"),
    "dirichlet_alpha_0.1": ("dirichlet_a01", "Dirichlet α=0.1"),
}

COLUMN_MAP = {
    "round": "round",
    "accuracy": "global_acc",
    "beta_hat": "beta_hat",
    "rho_hat": "rho_hat",
    "subspace_rank": "protected_basis_rank",
    "orthonormality_error": "protected_orthonormality_error",
    "measurement_overhead_seconds": "measurement_overhead_seconds",
}

# A five-percentage-point span is used only as a transparent descriptive flag,
# not as a hypothesis test or claim of statistical significance.
MEANINGFUL_BETA_RANGE = 0.05

STYLES = {
    "equal": {"color": "#333333", "marker": "o", "linestyle": "--"},
    "dirichlet_alpha_0.5": {
        "color": "#1f77b4",
        "marker": "s",
        "linestyle": "-",
    },
    "dirichlet_alpha_0.1": {
        "color": "#d62728",
        "marker": "^",
        "linestyle": "-",
    },
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
    required = set(COLUMN_MAP.values())
    for protocol, (directory, label) in PROTOCOLS.items():
        source = find_summary(STAGE0_ROOT / directory / "seed42")
        raw = pd.read_csv(source)
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise RuntimeError(f"{source} is missing required columns: {missing}")
        if raw["round"].duplicated().any():
            raise RuntimeError(f"{source} contains duplicate round values")

        frame = raw[list(COLUMN_MAP.values())].rename(
            columns={value: key for key, value in COLUMN_MAP.items()}
        )
        frame.insert(0, "protocol_label", label)
        frame.insert(0, "protocol", protocol)
        frame["source_file"] = source.relative_to(REPO_ROOT).as_posix()

        # No protected subspace exists at the start of round 0.
        frame.loc[frame["round"] == 0, ["beta_hat", "rho_hat"]] = np.nan
        frames.append(frame)

    comparison = pd.concat(frames, ignore_index=True)
    return comparison.reset_index(drop=True)


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol, (_, label) in PROTOCOLS.items():
        data = comparison[comparison["protocol"] == protocol].sort_values("round")
        valid = data.dropna(subset=["beta_hat", "rho_hat"])
        beta = valid["beta_hat"]
        rho = valid["rho_hat"]
        beta_range = float(beta.max() - beta.min())
        beta_changes = beta.diff().abs().dropna()
        rows.append(
            {
                "protocol": protocol,
                "protocol_label": label,
                "valid_signal_rounds": len(valid),
                "beta_hat_min": beta.min(),
                "beta_hat_max": beta.max(),
                "beta_hat_mean": beta.mean(),
                "beta_hat_std_population": beta.std(ddof=0),
                "beta_hat_range": beta_range,
                "beta_hat_mean_abs_round_change": beta_changes.mean(),
                "beta_hat_max_abs_round_change": beta_changes.max(),
                "beta_meaningful_round_variation": beta_range
                >= MEANINGFUL_BETA_RANGE,
                "rho_hat_min": rho.min(),
                "rho_hat_max": rho.max(),
                "rho_hat_mean": rho.mean(),
                "rho_hat_std_population": rho.std(ddof=0),
                "rho_hat_range": rho.max() - rho.min(),
                "final_accuracy": data.iloc[-1]["accuracy"],
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
    unit_interval: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    axis.set_facecolor("white")
    for protocol, (_, label) in PROTOCOLS.items():
        data = comparison[comparison["protocol"] == protocol].sort_values("round")
        values = data[column] * 100 if percent else data[column]
        axis.plot(
            data["round"],
            values,
            label=label,
            linewidth=2.0,
            markersize=5,
            **STYLES[protocol],
        )

    axis.set_xlabel("Round")
    axis.set_ylabel(ylabel)
    axis.set_xticks(sorted(comparison["round"].unique()))
    if unit_interval:
        axis.set_ylim(0.0, 1.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison = load_comparison()
    summary = summarize(comparison)

    comparison.to_csv(OUTPUT_ROOT / "stage0_round_comparison.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "stage0_protocol_statistics.csv", index=False)

    plot_metric(
        comparison,
        "beta_hat",
        "β̂ (gradient-energy fraction inside protected subspace)",
        "stage0_beta_hat_vs_round.png",
        unit_interval=True,
    )
    plot_metric(
        comparison,
        "rho_hat",
        "ρ̂ (relative gradient residual)",
        "stage0_rho_hat_vs_round.png",
        unit_interval=True,
    )
    plot_metric(
        comparison,
        "accuracy",
        "Global accuracy (%)",
        "stage0_accuracy_vs_round.png",
        percent=True,
    )
    plot_metric(
        comparison,
        "measurement_overhead_seconds",
        "Measurement overhead (seconds)",
        "stage0_measurement_overhead_vs_round.png",
    )

    print("Confirmed source-column mapping:")
    for analysis_name, source_name in COLUMN_MAP.items():
        print(f"  {analysis_name}: {source_name}")
    print("\nProtocol statistics (population standard deviation):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
