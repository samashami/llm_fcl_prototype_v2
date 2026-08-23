"""Validate Stage 1 norm matching and compare learned-Phi with shrinkage."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE1_ROOT = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "cifar100"
    / "lmss_geo"
    / "stage1_fixed_lambda"
)
PROJECTION_DIR = STAGE1_ROOT / "dirichlet_a01" / "lambda075_energycal" / "seed42"
SHRINKAGE_DIR = (
    STAGE1_ROOT / "dirichlet_a01" / "norm_matched_shrinkage" / "seed42"
)
PRIOR_PROJECTION_DIR = STAGE1_ROOT / "dirichlet_a01" / "lambda075" / "seed42"
SCHEDULE_PATH = SHRINKAGE_DIR / "norm_matched_schedule_from_lambda075.csv"
OUTPUT_DIR = STAGE1_ROOT / "analysis_mechanism"

PROJECTION_TAG = "full_s42_fixed_epochs5_pat2_dirichlet_a01_lambda075_energycal"
SHRINKAGE_TAG = "full_s42_fixed_epochs5_pat2_dirichlet_a01_normmatched_lambda075"
CONTROLLED_ROUNDS = tuple(range(1, 7))
PROTECTED_LAYERS = ("layer4_1_conv2", "fc")
ENERGY_KEYS = ["round", "client", "layer"]

# Explicit viability tolerances for this protocol check. These are engineering
# match tolerances, not statistical thresholds and were not preregistered.
MAX_RETENTION_FRACTION_ABS_ERROR = 1e-6
MAX_GLOBAL_ENERGY_REL_ERROR = 0.01
MAX_CELL_ENERGY_REL_ERROR = 0.05

SUMMARY_COLUMNS = {
    "accuracy": "global_acc",
    "forgetting_mean": "forget_mean",
    "forgetting_max": "forget_max",
    "aulc": "aulc_running",
    "divergence": "divergence",
    "beta_hat": "beta_hat",
    "rho_hat": "rho_hat",
    "basis_rank": "protected_basis_rank",
    "measurement_overhead_seconds": "measurement_overhead_seconds",
}


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {pattern!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(directory: Path, expected_tag: str) -> dict[str, object]:
    paths = {
        "results": find_one(directory, "fcl_run_results_*.csv"),
        "summary": find_one(directory, "fcl_run_summary_*.csv"),
        "cl_batches": find_one(directory, "fcl_run_cl_batches_*.csv"),
        "energy_steps": find_one(directory, "fcl_run_update_energy_steps_*.csv"),
        "energy_rounds": find_one(directory, "fcl_run_update_energy_rounds_*.csv"),
    }
    run = {name: pd.read_csv(path) for name, path in paths.items()}
    run["paths"] = paths

    expected_columns = {
        "results": {
            "run_id", "tag", "round", "client", "epoch", "lr",
            "replay_ratio", "cl_batch", "cl_batch_size", "train_loss",
            "train_acc", "val_loss", "val_acc",
        },
        "summary": {"run_id", "tag", "round", *SUMMARY_COLUMNS.values()},
        "cl_batches": {"run_id", "client", "cl_batch", "size"},
        "energy_steps": {
            "run_id", "tag", "round", "client", "layer", "update_control",
            "projection_lambda", "shrinkage_factor", "raw_update_energy",
            "projected_update_energy", "retained_energy_fraction",
            "local_optimizer_step", "epoch", "batch",
        },
        "energy_rounds": {
            "run_id", "tag", "round", "update_control", "projection_lambda",
            "step_layer_count", "raw_update_energy_sum",
            "projected_update_energy_sum", "retained_energy_fraction",
        },
    }
    for name, required in expected_columns.items():
        missing = sorted(required.difference(run[name].columns))
        if missing:
            raise RuntimeError(f"{paths[name]} is missing columns: {missing}")
    for name in ("results", "summary", "energy_steps", "energy_rounds"):
        tags = set(run[name]["tag"].dropna().unique())
        if tags != {expected_tag}:
            raise RuntimeError(f"Unexpected tag(s) in {paths[name]}: {tags}")
    if run["summary"]["round"].duplicated().any():
        raise RuntimeError(f"Duplicate summary rounds in {paths['summary']}")
    return run


def validate_comparability(projection: dict, shrinkage: dict) -> dict[str, bool]:
    p_results = projection["results"]
    s_results = shrinkage["results"]
    result_protocol_columns = [
        "round", "client", "epoch", "lr", "replay_ratio", "cl_batch",
        "cl_batch_size",
    ]
    p_batches = projection["cl_batches"].drop(columns="run_id")
    s_batches = shrinkage["cl_batches"].drop(columns="run_id")
    p_steps = projection["energy_steps"]
    s_steps = shrinkage["energy_steps"]
    p_summary = projection["summary"]
    s_summary = shrinkage["summary"]

    checks = {
        "same_epoch_client_schedule": p_results[result_protocol_columns].equals(
            s_results[result_protocol_columns]
        ),
        "same_cl_batch_sizes": p_batches.equals(s_batches),
        "same_rounds": p_summary["round"].equals(s_summary["round"]),
        "same_lr": p_results["lr"].equals(s_results["lr"]),
        "same_replay": p_results["replay_ratio"].equals(s_results["replay_ratio"]),
        "same_protected_layers": set(p_steps["layer"].unique())
        == set(s_steps["layer"].unique())
        == set(PROTECTED_LAYERS),
        "same_basis_rank_trace": p_summary["protected_basis_rank"].equals(
            s_summary["protected_basis_rank"]
        ),
        "same_gradient_measurement_count": p_summary[
            "gradient_measurement_count"
        ].equals(s_summary["gradient_measurement_count"]),
        "projection_mode_verified": set(p_steps["update_control"].unique())
        == {"projection"},
        "projection_lambda_verified": set(p_steps["projection_lambda"].unique())
        == {0.75},
        "shrinkage_mode_verified": set(s_steps["update_control"].unique())
        == {"shrinkage"},
        "shrinkage_projection_lambda_zero": set(
            s_steps["projection_lambda"].unique()
        ) == {0.0},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Run comparability checks failed: {failed}")
    return checks


def aggregate_energy(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if group_columns:
        result = (
            frame.groupby(group_columns, as_index=False)
            .agg(
                raw_update_energy=("raw_update_energy", "sum"),
                retained_update_energy=("projected_update_energy", "sum"),
                step_count=("raw_update_energy", "size"),
            )
        )
    else:
        result = pd.DataFrame(
            {
                "raw_update_energy": [frame["raw_update_energy"].sum()],
                "retained_update_energy": [frame["projected_update_energy"].sum()],
                "step_count": [len(frame)],
            }
        )
    result["retained_energy_fraction"] = (
        result["retained_update_energy"] / result["raw_update_energy"]
    )
    return result


def compare_energy_level(
    projection: pd.DataFrame,
    shrinkage: pd.DataFrame,
    level: str,
    group_columns: list[str],
) -> pd.DataFrame:
    p = aggregate_energy(projection, group_columns).rename(
        columns={
            "raw_update_energy": "learned_phi_raw_update_energy",
            "retained_update_energy": "learned_phi_retained_update_energy",
            "retained_energy_fraction": "learned_phi_retained_energy_fraction",
            "step_count": "learned_phi_step_count",
        }
    )
    s = aggregate_energy(shrinkage, group_columns).rename(
        columns={
            "raw_update_energy": "shrinkage_raw_update_energy",
            "retained_update_energy": "shrinkage_retained_update_energy",
            "retained_energy_fraction": "shrinkage_retained_energy_fraction",
            "step_count": "shrinkage_step_count",
        }
    )
    if group_columns:
        result = p.merge(s, on=group_columns, how="outer", validate="one_to_one")
    else:
        result = pd.concat([p, s], axis=1)
    result.insert(0, "analysis_level", level)
    for key in ENERGY_KEYS:
        if key not in result:
            result[key] = np.nan

    for metric in ("raw_update_energy", "retained_update_energy"):
        learned = result[f"learned_phi_{metric}"]
        control = result[f"shrinkage_{metric}"]
        result[f"{metric}_absolute_difference"] = (learned - control).abs()
        result[f"{metric}_relative_difference"] = (
            result[f"{metric}_absolute_difference"] / learned.abs()
        )
    learned_fraction = result["learned_phi_retained_energy_fraction"]
    control_fraction = result["shrinkage_retained_energy_fraction"]
    result["retained_energy_fraction_absolute_difference"] = (
        learned_fraction - control_fraction
    ).abs()
    result["retained_energy_fraction_relative_difference"] = (
        result["retained_energy_fraction_absolute_difference"]
        / learned_fraction.abs()
    )
    result["step_count_difference"] = (
        result["learned_phi_step_count"] - result["shrinkage_step_count"]
    )
    return result


def validate_schedule(schedule: pd.DataFrame, projection: dict) -> None:
    required = {
        "round", "client", "layer", "shrinkage_factor", "source_file",
        "source_sha256", "raw_update_energy_sum", "retained_update_energy_sum",
        "retained_energy_fraction", "projected_step_count",
    }
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise RuntimeError(f"Schedule is missing columns: {missing}")
    if schedule.duplicated(ENERGY_KEYS).any() or len(schedule) != 56:
        raise RuntimeError("Schedule must contain 56 unique round/client/layer entries")
    expected_keys = {
        (round_id, client, layer)
        for round_id in range(7)
        for client in range(4)
        for layer in PROTECTED_LAYERS
    }
    actual_keys = set(map(tuple, schedule[ENERGY_KEYS].itertuples(index=False, name=None)))
    if actual_keys != expected_keys:
        raise RuntimeError("Schedule keys do not cover the expected 7x4x2 grid")
    source_path = projection["paths"]["energy_steps"]
    if set(schedule["source_file"].unique()) != {source_path.name}:
        raise RuntimeError("Schedule source filename does not match calibration log")
    checksum = sha256_file(source_path)
    if set(schedule["source_sha256"].unique()) != {checksum}:
        raise RuntimeError("Schedule source checksum does not match calibration log")
    round_zero = schedule[schedule["round"] == 0]
    if not np.allclose(round_zero["shrinkage_factor"], 1.0, atol=0.0, rtol=0.0):
        raise RuntimeError("Round-0 schedule entries must use s=1")

    p = aggregate_energy(
        projection["energy_steps"].query("round > 0"), ENERGY_KEYS
    ).rename(
        columns={
            "raw_update_energy": "computed_raw",
            "retained_update_energy": "computed_retained",
            "retained_energy_fraction": "computed_fraction",
            "step_count": "computed_steps",
        }
    )
    joined = schedule.query("round > 0").merge(
        p, on=ENERGY_KEYS, validate="one_to_one"
    )
    comparisons = (
        np.allclose(joined["raw_update_energy_sum"], joined["computed_raw"]),
        np.allclose(joined["retained_update_energy_sum"], joined["computed_retained"]),
        np.allclose(joined["retained_energy_fraction"], joined["computed_fraction"]),
        np.array_equal(joined["projected_step_count"], joined["computed_steps"]),
        np.allclose(
            joined["shrinkage_factor"].pow(2),
            joined["computed_fraction"],
            atol=1e-12,
            rtol=1e-7,
        ),
    )
    if not all(comparisons):
        raise RuntimeError("Schedule values do not reproduce calibration energy sums")


def build_energy_validation(projection: dict, shrinkage: dict) -> tuple[pd.DataFrame, bool]:
    p = projection["energy_steps"].query("round in @CONTROLLED_ROUNDS").copy()
    s = shrinkage["energy_steps"].query("round in @CONTROLLED_ROUNDS").copy()
    levels = [
        ("round_client_layer", ["round", "client", "layer"]),
        ("round", ["round"]),
        ("client", ["client"]),
        ("layer", ["layer"]),
        ("global", []),
    ]
    validation = pd.concat(
        [compare_energy_level(p, s, level, columns) for level, columns in levels],
        ignore_index=True,
    )
    detail = validation[validation["analysis_level"] == "round_client_layer"]
    global_row = validation[validation["analysis_level"] == "global"].iloc[0]
    valid = bool(
        (detail["step_count_difference"] == 0).all()
        and detail["retained_energy_fraction_absolute_difference"].max()
        <= MAX_RETENTION_FRACTION_ABS_ERROR
        and global_row["raw_update_energy_relative_difference"]
        <= MAX_GLOBAL_ENERGY_REL_ERROR
        and global_row["retained_update_energy_relative_difference"]
        <= MAX_GLOBAL_ENERGY_REL_ERROR
        and detail["raw_update_energy_relative_difference"].max()
        <= MAX_CELL_ENERGY_REL_ERROR
        and detail["retained_update_energy_relative_difference"].max()
        <= MAX_CELL_ENERGY_REL_ERROR
    )
    validation["energy_match_valid"] = valid
    return validation, valid


def build_round_comparison(
    projection: dict, shrinkage: dict, prior_projection: pd.DataFrame
) -> pd.DataFrame:
    def select(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
        selected = summary[["round", *SUMMARY_COLUMNS.values()]].rename(
            columns={source: f"{prefix}_{name}" for name, source in SUMMARY_COLUMNS.items()}
        )
        return selected

    comparison = select(projection["summary"], "learned_phi").merge(
        select(shrinkage["summary"], "shrinkage"), on="round", validate="one_to_one"
    )
    repeatability = select(prior_projection, "prior_projection")
    comparison = comparison.merge(repeatability, on="round", validate="one_to_one")
    for metric in SUMMARY_COLUMNS:
        comparison[f"learned_phi_minus_shrinkage_{metric}"] = (
            comparison[f"learned_phi_{metric}"] - comparison[f"shrinkage_{metric}"]
        )
        comparison[f"prior_repeatability_absolute_difference_{metric}"] = (
            comparison[f"prior_projection_{metric}"]
            - comparison[f"learned_phi_{metric}"]
        ).abs()
        comparison[f"difference_exceeds_observed_repeatability_{metric}"] = (
            comparison[f"learned_phi_minus_shrinkage_{metric}"].abs()
            > comparison[f"prior_repeatability_absolute_difference_{metric}"]
        )
    return comparison


def build_final_comparison(
    round_comparison: pd.DataFrame,
    projection: dict,
    shrinkage: dict,
) -> pd.DataFrame:
    final_round = round_comparison.sort_values("round").iloc[-1]
    rows = []
    for method, run in (("learned_phi", projection), ("shrinkage", shrinkage)):
        controlled = run["energy_steps"].query("round in @CONTROLLED_ROUNDS")
        retention = (
            controlled["projected_update_energy"].sum()
            / controlled["raw_update_energy"].sum()
        )
        rows.append(
            {
                "row_type": "method",
                "method": method,
                "final_accuracy": final_round[f"{method}_accuracy"],
                "final_forgetting_mean": final_round[f"{method}_forgetting_mean"],
                "final_forgetting_max": final_round[f"{method}_forgetting_max"],
                "final_aulc": final_round[f"{method}_aulc"],
                "final_divergence": final_round[f"{method}_divergence"],
                "final_beta_hat": final_round[f"{method}_beta_hat"],
                "final_rho_hat": final_round[f"{method}_rho_hat"],
                "average_update_energy_retention": retention,
                "average_measurement_overhead_seconds": run["summary"][
                    "measurement_overhead_seconds"
                ].mean(),
            }
        )
    methods = pd.DataFrame(rows)
    difference = {
        "row_type": "difference",
        "method": "learned_phi_minus_shrinkage",
    }
    for column in methods.columns.difference(["row_type", "method"]):
        difference[column] = methods.loc[0, column] - methods.loc[1, column]
    return pd.concat([methods, pd.DataFrame([difference])], ignore_index=True)


def plot_tradeoff(round_comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharex=True)
    styles = {
        "learned_phi": {"label": "Learned-$\\Phi$ projection", "marker": "o"},
        "shrinkage": {"label": "Norm-matched shrinkage", "marker": "s"},
    }
    for method, style in styles.items():
        axes[0].plot(
            round_comparison["round"],
            100 * round_comparison[f"{method}_accuracy"],
            linewidth=2,
            **style,
        )
        axes[1].plot(
            round_comparison["round"],
            100 * round_comparison[f"{method}_forgetting_mean"],
            linewidth=2,
            **style,
        )
    axes[0].set_ylabel("Global accuracy (%)")
    axes[1].set_ylabel("Mean forgetting (%)")
    for axis in axes:
        axis.set_xlabel("Round")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Stage 1 mechanism validation: stability–plasticity trade-off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mechanism_tradeoff_by_round.png", dpi=180)
    plt.close(fig)


def plot_energy_match(energy_validation: pd.DataFrame) -> None:
    rounds = energy_validation[energy_validation["analysis_level"] == "round"].sort_values(
        "round"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    axes[0].plot(
        rounds["round"],
        100 * rounds["learned_phi_retained_energy_fraction"],
        marker="o",
        linewidth=2,
        label="Learned-$\\Phi$ projection",
    )
    axes[0].plot(
        rounds["round"],
        100 * rounds["shrinkage_retained_energy_fraction"],
        marker="s",
        linewidth=2,
        label="Norm-matched shrinkage",
    )
    axes[0].set_ylabel("Retained update energy (%)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        rounds["round"],
        100 * rounds["raw_update_energy_relative_difference"],
        marker="o",
        linewidth=2,
        label="Raw energy",
    )
    axes[1].plot(
        rounds["round"],
        100 * rounds["retained_update_energy_relative_difference"],
        marker="s",
        linewidth=2,
        label="Retained energy",
    )
    axes[1].set_ylabel("Absolute relative difference (%)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Round")
        axis.grid(alpha=0.25)
    fig.suptitle("Norm-matched control: realized energy validation")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.82, wspace=0.32)
    fig.savefig(OUTPUT_DIR / "energy_match_by_round.png", dpi=180)
    plt.close(fig)


def main() -> None:
    projection = load_run(PROJECTION_DIR, PROJECTION_TAG)
    shrinkage = load_run(SHRINKAGE_DIR, SHRINKAGE_TAG)
    comparability = validate_comparability(projection, shrinkage)
    schedule = pd.read_csv(SCHEDULE_PATH)
    validate_schedule(schedule, projection)

    prior_summary_path = find_one(PRIOR_PROJECTION_DIR, "fcl_run_summary_*.csv")
    prior_summary = pd.read_csv(prior_summary_path)
    energy_validation, energy_valid = build_energy_validation(projection, shrinkage)
    if not energy_valid:
        raise RuntimeError(
            "Energy matching failed the declared tolerances; mechanism comparison is invalid"
        )
    round_comparison = build_round_comparison(
        projection, shrinkage, prior_summary
    )
    final_comparison = build_final_comparison(
        round_comparison, projection, shrinkage
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    round_comparison.to_csv(
        OUTPUT_DIR / "stage1_mechanism_round_comparison.csv", index=False
    )
    final_comparison.to_csv(
        OUTPUT_DIR / "stage1_mechanism_final_comparison.csv", index=False
    )
    energy_validation.to_csv(
        OUTPUT_DIR / "stage1_mechanism_energy_match_validation.csv", index=False
    )
    plot_tradeoff(round_comparison)
    plot_energy_match(energy_validation)

    detail = energy_validation.query("analysis_level == 'round_client_layer'")
    global_row = energy_validation.query("analysis_level == 'global'").iloc[0]
    final = final_comparison.set_index("method")
    print("Schemas and protocol artifacts validated.")
    print(f"Comparability checks passed: {len(comparability)}/{len(comparability)}")
    print(f"Schedule provenance SHA-256: {schedule['source_sha256'].iloc[0]}")
    print(f"Energy matching valid: {'YES' if energy_valid else 'NO'}")
    print(
        "Global raw/retained relative error: "
        f"{global_row['raw_update_energy_relative_difference']:.6%} / "
        f"{global_row['retained_update_energy_relative_difference']:.6%}"
    )
    print(
        "Cell mean/max raw-energy relative error: "
        f"{detail['raw_update_energy_relative_difference'].mean():.6%} / "
        f"{detail['raw_update_energy_relative_difference'].max():.6%}"
    )
    print(
        "Cell mean/max retention-fraction absolute error: "
        f"{detail['retained_energy_fraction_absolute_difference'].mean():.3e} / "
        f"{detail['retained_energy_fraction_absolute_difference'].max():.3e}"
    )
    print("Final learned-Phi minus shrinkage:")
    for metric in (
        "final_accuracy", "final_forgetting_mean", "final_aulc", "final_divergence"
    ):
        print(f"  {metric}: {final.loc['learned_phi_minus_shrinkage', metric]:+.6f}")
    round_zero = round_comparison.loc[round_comparison["round"] == 0].iloc[0]
    print(
        "Round-0 nominal-no-op accuracy difference: "
        f"{round_zero['learned_phi_minus_shrinkage_accuracy']:+.6f}"
    )
    print(f"Outputs: {OUTPUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
