"""Derive a frozen round/client/layer norm-matched shrinkage schedule."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import pandas as pd


DEFAULT_LAYERS = ("layer4_1_conv2", "fc")
REQUIRED_COLUMNS = {
    "round",
    "client",
    "layer",
    "update_control",
    "projection_lambda",
    "raw_update_energy",
    "projected_update_energy",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_schedule(
    step_log: Path,
    output: Path,
    rounds: int = 7,
    clients: int = 4,
    layers=DEFAULT_LAYERS,
    projection_lambda: float = 0.75,
    calibration_round: int | None = None,
) -> pd.DataFrame:
    """Derive and exclusively create the frozen schedule CSV."""
    if rounds <= 0 or clients <= 0:
        raise ValueError("rounds and clients must be positive")
    if calibration_round is not None and not 0 <= calibration_round < rounds:
        raise ValueError("calibration_round must satisfy 0 <= value < rounds")
    source = pd.read_csv(step_log)
    missing = sorted(REQUIRED_COLUMNS.difference(source.columns))
    if missing:
        raise ValueError(f"calibration step log is missing columns: {missing}")
    if set(source["update_control"].dropna().unique()) != {"projection"}:
        raise ValueError("calibration log must contain projection updates only")
    lambdas = source["projection_lambda"].dropna().astype(float).unique()
    if len(lambdas) != 1 or not math.isclose(
        float(lambdas[0]), projection_lambda, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "calibration log projection lambda does not match "
            f"--projection-lambda={projection_lambda:g}"
        )
    source_rounds = set(source["round"].dropna().astype(int).unique())
    if calibration_round is not None and source_rounds != {calibration_round}:
        raise ValueError(
            "single-round calibration log must contain only round "
            f"{calibration_round}, found {sorted(source_rounds)}"
        )

    grouped = (
        source.groupby(["round", "client", "layer"], as_index=False)
        .agg(
            raw_update_energy_sum=("raw_update_energy", "sum"),
            retained_update_energy_sum=("projected_update_energy", "sum"),
            projected_step_count=("raw_update_energy", "size"),
        )
    )
    grouped_by_key = {
        (int(row["round"]), int(row["client"]), str(row["layer"])): row
        for _, row in grouped.iterrows()
    }

    source_checksum = sha256_file(step_log)
    rows = []
    for round_id in range(rounds):
        for client_id in range(clients):
            for layer in layers:
                key = (round_id, client_id, layer)
                use_calibration = (
                    round_id != 0
                    and (calibration_round is None or round_id == calibration_round)
                )
                if not use_calibration:
                    raw = retained = 0.0
                    count = 0
                    retained_fraction = 1.0
                    factor = 1.0
                else:
                    if key not in grouped_by_key:
                        raise ValueError(f"calibration log is missing projected steps for {key}")
                    values = grouped_by_key[key]
                    raw = float(values["raw_update_energy_sum"])
                    retained = float(values["retained_update_energy_sum"])
                    count = int(values["projected_step_count"])
                    if raw < 0.0 or retained < 0.0:
                        raise ValueError(f"negative update energy for {key}")
                    if raw == 0.0:
                        retained_fraction = 1.0
                        factor = 1.0
                    else:
                        retained_fraction = retained / raw
                        if retained_fraction > 1.0 + 1e-6:
                            raise ValueError(f"projection increased aggregate energy for {key}")
                        retained_fraction = min(1.0, max(0.0, retained_fraction))
                        factor = math.sqrt(retained_fraction)
                rows.append(
                    {
                        "round": round_id,
                        "client": client_id,
                        "layer": layer,
                        "shrinkage_factor": factor,
                        "raw_update_energy_sum": raw,
                        "retained_update_energy_sum": retained,
                        "retained_energy_fraction": retained_fraction,
                        "projected_step_count": count,
                        "projection_lambda": float(projection_lambda),
                        "calibration_round": calibration_round,
                        "used_for_calibration": use_calibration,
                        "source_file": step_log.name,
                        "source_sha256": source_checksum,
                        "derivation": "sqrt(sum(projected_update_energy)/sum(raw_update_energy))",
                    }
                )

    schedule = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(output, index=False, mode="x")
    return schedule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--layers", nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--projection-lambda", type=float, default=0.75)
    parser.add_argument("--calibration-round", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    schedule = derive_schedule(
        args.step_log,
        args.output,
        rounds=args.rounds,
        clients=args.clients,
        layers=tuple(args.layers),
        projection_lambda=args.projection_lambda,
        calibration_round=args.calibration_round,
    )
    print(f"Wrote immutable schedule: {args.output} ({len(schedule)} entries)")


if __name__ == "__main__":
    main()
