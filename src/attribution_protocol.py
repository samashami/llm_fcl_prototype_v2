"""Pure helpers for the feedback-attribution experiment protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


DEVELOPMENT_DOMAIN_ORDER: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
HELDOUT_DOMAIN_ORDER: Tuple[int, ...] = (0, 4, 2, 3, 1, 5, 6)
DOMAIN_ORDERS = {
    "development": DEVELOPMENT_DOMAIN_ORDER,
    "heldout": HELDOUT_DOMAIN_ORDER,
}

CONTROL_MODES = ("joint", "eta_only", "rho_only")


def accumulate_post_initialization_utility(
    total: float,
    count: int,
    *,
    stage_id: int,
    mean_seen_domain_accuracy: float,
) -> Tuple[float, int]:
    """Accumulate trajectory utility only after the fixed initialization stage."""
    if int(stage_id) <= 0:
        return float(total), int(count)
    return float(total) + float(mean_seen_domain_accuracy), int(count) + 1


def nearest_strategy_id(
    *,
    target_lr: float,
    target_replay_ratio: float,
    palette: Dict[int, Dict[str, Any]],
) -> int:
    """Project a requested joint action onto a fixed discrete palette."""
    if not palette:
        raise ValueError("strategy palette must not be empty")
    lr_values = [float(row["lr"]) for row in palette.values()]
    replay_values = [float(row["replay_ratio"]) for row in palette.values()]
    lr_span = max(lr_values) - min(lr_values) or 1.0
    replay_span = max(replay_values) - min(replay_values) or 1.0

    def distance(strategy_id: int):
        row = palette[strategy_id]
        lr_delta = (float(row["lr"]) - float(target_lr)) / lr_span
        replay_delta = (
            float(row["replay_ratio"]) - float(target_replay_ratio)
        ) / replay_span
        return lr_delta * lr_delta + replay_delta * replay_delta, int(strategy_id)

    return min(palette, key=distance)


def load_frozen_action_schedule(path: str, required_rounds: Sequence[int]):
    """Load an immutable action schedule together with its source manifest."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError("frozen action schedule must be a run action directory")
    actions: Dict[int, Dict[str, Any]] = {}
    digest = hashlib.sha256()
    provenance_path = root / "run_protocol.json"
    if not provenance_path.is_file():
        raise ValueError("frozen schedule is missing run_protocol.json")
    provenance_payload = provenance_path.read_bytes()
    digest.update(provenance_path.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(provenance_payload)
    provenance = json.loads(provenance_payload)
    if not isinstance(provenance, dict):
        raise ValueError("run_protocol.json must contain a JSON object")
    for round_id in sorted(int(value) for value in required_rounds):
        action_path = root / f"action_round_{round_id}.json"
        if not action_path.is_file():
            raise ValueError(f"frozen schedule is missing {action_path.name}")
        payload = action_path.read_bytes()
        digest.update(action_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError(f"{action_path.name} must contain a JSON object")
        actions[round_id] = parsed
    return actions, digest.hexdigest(), provenance


def validate_frozen_schedule_provenance(
    actions: Dict[int, Dict[str, Any]],
    provenance: Dict[str, Any],
    *,
    expected_model: str,
    expected_palette: Dict[int, Dict[str, Any]],
    expected_policy_sha256: str,
    expected_seed: int = 40,
) -> None:
    """Reject schedules that are not the preregistered live joint LMSS source."""
    manifest = provenance.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("frozen schedule provenance has no manifest")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("frozen schedule provenance has no protocol")
    expected_fields = {
        "attribution_protocol": True,
        "controller": "lmss_openrouter",
        "control_mode": "joint",
        "domain_order": "development",
        "evaluation_source": "validation",
        "seed": int(expected_seed),
        "resolved_lmss_model": str(expected_model),
    }
    mismatches = {
        key: {"expected": expected, "found": protocol.get(key)}
        for key, expected in expected_fields.items()
        if protocol.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"frozen schedule source protocol differs: {mismatches}")

    source_palette = provenance.get("strategy_palette")
    normalized_palette = {str(key): value for key, value in expected_palette.items()}
    if source_palette != normalized_palette:
        raise ValueError("frozen schedule strategy palette differs")

    source_hash = (
        manifest.get("code", {})
        .get("files_sha256", {})
        .get("src/policy/lmss_openrouter.py")
    )
    if source_hash != expected_policy_sha256:
        raise ValueError("frozen schedule LMSS policy code differs")

    for round_id, action in sorted(actions.items()):
        metadata = action.get("controller_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"round {round_id} has no controller metadata")
        if metadata.get("fallback") is not False:
            raise ValueError(f"round {round_id} used an LMSS fallback")
        if int(metadata.get("call_count", 0) or 0) != 1:
            raise ValueError(f"round {round_id} was not produced by one live LMSS call")
        if metadata.get("requested_model") != expected_model:
            raise ValueError(f"round {round_id} used a different requested model")
        prompt_hash = metadata.get("prompt_sha256")
        if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
            raise ValueError(f"round {round_id} has no valid prompt hash")
        if action.get("control_mode") != "joint":
            raise ValueError(f"round {round_id} is not a joint-control action")
        try:
            strategy_id = int(action["strategy_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"round {round_id} has no valid strategy id") from exc
        if strategy_id not in expected_palette:
            raise ValueError(f"round {round_id} uses an unknown strategy id")
        expected_strategy = expected_palette[strategy_id]
        if float(action.get("lr", float("nan"))) != float(expected_strategy["lr"]):
            raise ValueError(f"round {round_id} learning rate differs from its strategy")
        clients = action.get("client_params")
        if not isinstance(clients, list) or not clients:
            raise ValueError(f"round {round_id} has no client action rows")
        expected_replay = float(expected_strategy["replay_ratio"])
        if any(
            float(client.get("replay_ratio", float("nan"))) != expected_replay
            for client in clients
        ):
            raise ValueError(f"round {round_id} replay ratio differs from its strategy")


def resolve_domain_order(name: str, domain_count: int) -> Tuple[int, ...]:
    """Return a frozen domain order and reject incompatible stream definitions."""
    try:
        order = DOMAIN_ORDERS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown domain order: {name!r}") from exc
    if len(order) != int(domain_count) or set(order) != set(range(int(domain_count))):
        raise ValueError("domain order must be a permutation of every configured domain")
    return order


def stage_and_block(round_id: int, blocks_per_stage: int) -> Tuple[int, int]:
    """Map a communication round to its continual stage and within-stage block."""
    round_id = int(round_id)
    blocks_per_stage = int(blocks_per_stage)
    if round_id < 0:
        raise ValueError("round_id must be non-negative")
    if blocks_per_stage <= 0:
        raise ValueError("blocks_per_stage must be positive")
    return divmod(round_id, blocks_per_stage)


def block_epoch_budget(total_stage_epochs: int, blocks_per_stage: int, block_id: int) -> int:
    """Split an existing stage epoch budget without increasing it."""
    total_stage_epochs = int(total_stage_epochs)
    blocks_per_stage = int(blocks_per_stage)
    block_id = int(block_id)
    if total_stage_epochs < blocks_per_stage:
        raise ValueError("total_stage_epochs must allow at least one epoch per block")
    if not 0 <= block_id < blocks_per_stage:
        raise ValueError("block_id is outside the stage")
    quotient, remainder = divmod(total_stage_epochs, blocks_per_stage)
    return quotient + (1 if block_id < remainder else 0)


def restrict_action_axis(
    action: Dict[str, Any],
    *,
    mode: str,
    n_clients: int,
    fixed_lr: float,
    fixed_replay_ratio: float,
) -> Dict[str, Any]:
    """Apply a joint, eta-only, or rho-only actuator restriction.

    The selector's requested action is retained for provenance. Every arm trains
    every client with unit client-specific LR scaling so the only treatment
    variables are the global learning rate and replay allocation.
    """
    if mode not in CONTROL_MODES:
        raise ValueError(f"unsupported control mode: {mode!r}")
    if int(n_clients) <= 0:
        raise ValueError("n_clients must be positive")

    requested = copy.deepcopy(action)
    requested_clients: Sequence[Dict[str, Any]] = requested.get("client_params", [])
    requested_replay = (
        float(requested_clients[0].get("replay_ratio", fixed_replay_ratio))
        if requested_clients
        else float(fixed_replay_ratio)
    )
    requested_lr = float(requested.get("lr", fixed_lr))

    applied_lr = float(fixed_lr) if mode == "rho_only" else requested_lr
    applied_replay = (
        float(fixed_replay_ratio) if mode == "eta_only" else requested_replay
    )

    return {
        "lr": applied_lr,
        "client_selection_k": int(n_clients),
        "aggregation": {"method": "FedAvg"},
        "client_params": [
            {
                "id": client_id,
                "replay_ratio": applied_replay,
                "lr_scale": 1.0,
                "ewc_lambda": 0.0,
            }
            for client_id in range(int(n_clients))
        ],
        "policy_source": requested.get("policy_source", "Unknown"),
        "control_mode": mode,
        "requested_lr": requested_lr,
        "requested_replay_ratio": requested_replay,
        "strategy_id": requested.get("strategy_id"),
        "reasoning": requested.get("reasoning", ""),
        "controller_metadata": copy.deepcopy(
            requested.get("controller_metadata", {})
        ),
    }
