"""Descriptive Stage 2 headroom diagnostic from fixed-lambda trajectories.

This analysis deliberately does not stitch rounds into an oracle trajectory.
Every per-round preference compares separate fixed-lambda model trajectories at
the same round number.
"""

from __future__ import annotations

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
OUTPUT_DIR = STAGE1_ROOT / "analysis" / "headroom"

LAMBDA_RUNS = {
    0.0: LMSS_GEO_ROOT / "stage0_measurement" / "dirichlet_a01" / "seed42",
    0.25: STAGE1_ROOT / "dirichlet_a01" / "lambda025" / "seed42",
    0.5: STAGE1_ROOT / "dirichlet_a01" / "lambda050" / "seed42",
    0.75: STAGE1_ROOT / "dirichlet_a01" / "lambda075" / "seed42",
    1.0: STAGE1_ROOT / "dirichlet_a01" / "lambda100" / "seed42",
}
KAPPAS = (0.5, 1.0, 2.0, 4.0)
NUMERICAL_TIE_ATOL_PP = 1e-6
NEGLIGIBLE_GAP_PP = 0.05

REQUIRED_COLUMNS = {
    "round",
    "global_acc",
    "forget_mean",
    "aulc_running",
    "beta_hat",
}


def find_summary(directory: Path) -> Path:
    matches = sorted(directory.glob("fcl_run_summary_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one summary CSV in {directory}, found {len(matches)}"
        )
    return matches[0]


def load_trajectories() -> pd.DataFrame:
    frames = []
    expected_rounds = None
    for lambda_value, directory in LAMBDA_RUNS.items():
        source = find_summary(directory)
        raw = pd.read_csv(source)
        missing = sorted(REQUIRED_COLUMNS.difference(raw.columns))
        if missing:
            raise RuntimeError(f"{source} is missing required columns: {missing}")
        if raw["round"].duplicated().any():
            raise RuntimeError(f"{source} has duplicate rounds")
        rounds = tuple(raw["round"].tolist())
        if expected_rounds is None:
            expected_rounds = rounds
        elif rounds != expected_rounds:
            raise RuntimeError("Fixed-lambda runs do not contain identical rounds")
        frame = raw[
            ["round", "global_acc", "forget_mean", "aulc_running", "beta_hat"]
        ].copy()
        frame.insert(0, "lambda", lambda_value)
        frame["accuracy_pp"] = 100.0 * frame["global_acc"]
        frame["forgetting_pp"] = 100.0 * frame["forget_mean"]
        frame["aulc_pp"] = 100.0 * frame["aulc_running"]
        frame["source_file"] = source.relative_to(REPO_ROOT).as_posix()
        frames.append(frame)
    trajectories = pd.concat(frames, ignore_index=True)
    # Round 0 has no previous protected basis, so beta_hat is undefined.
    trajectories.loc[trajectories["round"] == 0, "beta_hat"] = np.nan
    return trajectories


def rank_trajectories(trajectories: pd.DataFrame) -> pd.DataFrame:
    ranked = trajectories.copy()
    ranked["accuracy_rank"] = ranked.groupby("round")["accuracy_pp"].rank(
        method="min", ascending=False
    )
    ranked["forgetting_rank"] = ranked.groupby("round")["forgetting_pp"].rank(
        method="min", ascending=True
    )
    ranked["aulc_rank"] = ranked.groupby("round")["aulc_pp"].rank(
        method="min", ascending=False
    )
    stage0_beta = (
        ranked[ranked["lambda"] == 0.0][["round", "beta_hat"]]
        .rename(columns={"beta_hat": "stage0_alpha01_beta_hat"})
    )
    ranked = ranked.merge(stage0_beta, on="round", validate="many_to_one")
    return ranked.sort_values(["round", "lambda"]).reset_index(drop=True)


def lambda_list(values) -> str:
    return "|".join(f"{float(value):g}" for value in sorted(values))


def utility_preferences(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    beta_by_round = (
        ranked[["round", "stage0_alpha01_beta_hat"]]
        .drop_duplicates("round")
        .set_index("round")["stage0_alpha01_beta_hat"]
    )
    for kappa in KAPPAS:
        work = ranked.copy()
        work["utility_pp"] = work["accuracy_pp"] - kappa * work["forgetting_pp"]
        for round_id, group in work.groupby("round"):
            ordered = group.sort_values(
                ["utility_pp", "lambda"], ascending=[False, True]
            ).reset_index(drop=True)
            best_utility = float(ordered.loc[0, "utility_pp"])
            best_mask = np.isclose(
                ordered["utility_pp"],
                best_utility,
                atol=NUMERICAL_TIE_ATOL_PP,
                rtol=0.0,
            )
            best_lambdas = ordered.loc[best_mask, "lambda"].tolist()
            second_utility = float(ordered.loc[1, "utility_pp"])
            gap = best_utility - second_utility
            spread = best_utility - float(ordered.iloc[-1]["utility_pp"])
            ranking = ";".join(
                f"{row['lambda']:g}:{row['utility_pp']:.6f}"
                for _, row in ordered.iterrows()
            )
            rows.append(
                {
                    "kappa": kappa,
                    "round": int(round_id),
                    "best_lambda": lambda_list(best_lambdas),
                    "best_lambda_numeric_mean": float(np.mean(best_lambdas)),
                    "best_utility_pp": best_utility,
                    "second_best_lambda": f"{ordered.loc[1, 'lambda']:g}",
                    "second_best_utility_pp": second_utility,
                    "best_second_gap_pp": gap,
                    "best_worst_spread_pp": spread,
                    "best_second_gap_negligible": gap <= NEGLIGIBLE_GAP_PP,
                    "stage0_alpha01_beta_hat": beta_by_round.loc[round_id],
                    "utility_ranking_lambda_colon_utility_pp": ranking,
                    "interpretation": "descriptive_cross_trajectory_preference",
                }
            )
    return pd.DataFrame(rows)


def summarize_preferences(preferences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kappa, group in preferences.groupby("kappa"):
        controlled = group[group["round"] > 0].sort_values("round")
        preferred = controlled["best_lambda_numeric_mean"]
        beta = controlled["stage0_alpha01_beta_hat"]
        changes = int((controlled["best_lambda"].shift() != controlled["best_lambda"]).sum() - 1)
        rows.append(
            {
                "kappa": kappa,
                "controlled_round_preference_sequence": ";".join(
                    f"r{int(row['round'])}:{row['best_lambda']}"
                    for _, row in controlled.iterrows()
                ),
                "preference_change_count": changes,
                "unique_preferred_lambda_count": controlled["best_lambda"].nunique(),
                "effectively_constant": controlled["best_lambda"].nunique() == 1,
                "mean_best_second_gap_pp": controlled["best_second_gap_pp"].mean(),
                "min_best_second_gap_pp": controlled["best_second_gap_pp"].min(),
                "max_best_second_gap_pp": controlled["best_second_gap_pp"].max(),
                "mean_best_worst_spread_pp": controlled["best_worst_spread_pp"].mean(),
                "negligible_gap_round_count": int(
                    controlled["best_second_gap_negligible"].sum()
                ),
                "spearman_preferred_lambda_vs_round": preferred.corr(
                    controlled["round"], method="spearman"
                ),
                "pearson_preferred_lambda_vs_stage0_beta_hat": preferred.corr(beta),
                "spearman_preferred_lambda_vs_stage0_beta_hat": preferred.corr(
                    beta, method="spearman"
                ),
                "association_is_descriptive_only": True,
            }
        )
    return pd.DataFrame(rows).sort_values("kappa").reset_index(drop=True)


def plot_preferences(preferences: pd.DataFrame) -> None:
    controlled = preferences[preferences["round"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for kappa, group in controlled.groupby("kappa"):
        group = group.sort_values("round")
        axes[0].plot(
            group["round"],
            group["best_lambda_numeric_mean"],
            marker="o",
            linewidth=2,
            label=f"κ={kappa:g}",
        )
        axes[1].plot(
            group["round"],
            group["best_second_gap_pp"],
            marker="o",
            linewidth=2,
            label=f"κ={kappa:g}",
        )
    axes[0].set_ylabel("Descriptively preferred λ")
    axes[1].set_ylabel("Best − second-best utility (pp)")
    for axis in axes:
        axis.set_xlabel("Round")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Fixed-trajectory preference diagnostic (not an adaptive oracle)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stage2_headroom_preferences.png", dpi=180)
    plt.close(fig)


def plot_beta_association(preferences: pd.DataFrame) -> None:
    controlled = preferences[preferences["round"] > 0]
    beta = controlled[
        ["round", "stage0_alpha01_beta_hat"]
    ].drop_duplicates("round").sort_values("round")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(beta["round"], beta["stage0_alpha01_beta_hat"], marker="o")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Stage 0 α=0.1 beta_hat")
    axes[0].grid(alpha=0.25)
    for kappa, group in controlled.groupby("kappa"):
        axes[1].scatter(
            group["stage0_alpha01_beta_hat"],
            group["best_lambda_numeric_mean"],
            s=45,
            label=f"κ={kappa:g}",
        )
    axes[1].set_xlabel("Stage 0 α=0.1 beta_hat")
    axes[1].set_ylabel("Descriptively preferred λ")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Descriptive beta_hat association; no causal interpretation")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.80, wspace=0.35)
    fig.savefig(OUTPUT_DIR / "stage2_headroom_beta_association.png", dpi=180)
    plt.close(fig)


def main() -> None:
    trajectories = load_trajectories()
    ranked = rank_trajectories(trajectories)
    preferences = utility_preferences(ranked)
    summary = summarize_preferences(preferences)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUTPUT_DIR / "per_round_lambda_rankings.csv", index=False)
    preferences.to_csv(OUTPUT_DIR / "utility_preferences.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "utility_preference_summary.csv", index=False)
    plot_preferences(preferences)
    plot_beta_association(preferences)

    print("Descriptive fixed-trajectory headroom analysis complete.")
    print("No cross-trajectory stitching or adaptive-oracle claim was made.")
    print(summary.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
