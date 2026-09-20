"""CLI wrapper for the isolated random-Phi fixed-compute branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:  # Support both ``python -m experiments...`` and direct script execution.
    from experiments.random_phi_control import RandomPhiPatch
except ModuleNotFoundError:
    from random_phi_control import RandomPhiPatch


BASE_COMMIT = "79c507dac97cfd4aa13a558679659d169da767a8"


def _option_value(arguments: list[str], option: str, default=None):
    for index, value in enumerate(arguments):
        if value == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return default


def _replace_update_control(arguments: list[str]) -> list[str]:
    result = list(arguments)
    for index, value in enumerate(result):
        if value == "--update_control" and index + 1 < len(result):
            if result[index + 1] != "random_projection_norm_matched":
                raise ValueError("wrapper requires --update_control random_projection_norm_matched")
            result[index + 1] = "shrinkage_online"
            return result
        if value == "--update_control=random_projection_norm_matched":
            result[index] = "--update_control=shrinkage_online"
            return result
    raise ValueError("wrapper requires --update_control random_projection_norm_matched")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--random_phi_experiment_seed", type=int, default=None)
    parser.add_argument("--random_phi_metadata_path", type=str, default=None)
    wrapper_args, controller_args = parser.parse_known_args()
    controller_args = _replace_update_control(controller_args)
    if _option_value(controller_args, "--projection_lambda") not in {"1", "1.0"}:
        raise ValueError("random_projection_norm_matched requires --projection_lambda 1.0")
    for required in ("--resume_checkpoint", "--one_round", "--branch_local_epochs", "--determinism_gate_dir"):
        if required not in controller_args:
            raise ValueError(f"random_projection_norm_matched requires {required}")
    experiment_seed = wrapper_args.random_phi_experiment_seed
    if experiment_seed is None:
        experiment_seed = int(_option_value(controller_args, "--seed", "42"))
    output_dir = Path(_option_value(controller_args, "--output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    if _option_value(controller_args, "--tag") is None:
        controller_args.extend(["--tag", "random_projection_norm_matched"])
    else:
        tag_index = controller_args.index("--tag")
        controller_args[tag_index + 1] = "random_projection_norm_matched__" + controller_args[tag_index + 1]

    checkpoint_path = Path(_option_value(controller_args, "--resume_checkpoint"))
    from src.checkpointing import load_checkpoint

    checkpoint_payload, checkpoint_metadata = load_checkpoint(checkpoint_path)
    gate_dir = Path(_option_value(controller_args, "--determinism_gate_dir"))
    gate_path = gate_dir / f"determinism_round_{int(checkpoint_payload['next_round']):02d}_lambda000_PASS.json"
    gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
    patch = RandomPhiPatch(experiment_seed).install()
    original_argv = sys.argv
    try:
        sys.argv = ["src/run_llm_fcl_controller.py", *controller_args]
        from src import run_llm_fcl_controller

        run_llm_fcl_controller.main()
    finally:
        sys.argv = original_argv
        patch.restore()

    metadata_path = Path(wrapper_args.random_phi_metadata_path) if wrapper_args.random_phi_metadata_path else output_dir / "random_phi_experiment_metadata.json"
    source_files = [
        Path("experiments/__init__.py"),
        Path("experiments/random_phi_control.py"),
        Path("experiments/run_random_phi_branch.py"),
        Path("tests/test_random_phi_control.py"),
    ]
    metadata = {
        "experiment_mode": "random_projection_norm_matched",
        "base_commit": BASE_COMMIT,
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_metadata["sha256"],
        "starting_state_hash": checkpoint_metadata["starting_state_hash"],
        "parent_run_id": checkpoint_payload["parent_run_id"],
        "executed_round": checkpoint_payload["next_round"],
        "determinism_pass_gate_path": str(gate_path),
        "determinism_pass_gate": gate_document,
        "experiment_seed": experiment_seed,
        "branch_local_epochs": int(_option_value(controller_args, "--branch_local_epochs")),
        "projection_lambda": 1.0,
        "protected_layers": sorted(patch.bases),
        "learned_ranks": {name: item["rank"] for name, item in patch.bases.items()},
        "random_ranks": {name: item["rank"] for name, item in patch.bases.items()},
        "random_basis_seeds": {name: item["seed"] for name, item in patch.bases.items()},
        "random_basis_sha256": {name: item["sha256"] for name, item in patch.bases.items()},
        "experimental_file_sha256": {str(path): _sha256(path) for path in source_files},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
