"""Deterministic replay-only controller policy for Adaptive-rho experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


REPLAY_PALETTE = (0.3, 0.4, 0.5, 0.6, 0.7)
FORGETTING_THRESHOLD = 0.05
WARMUP_ROUNDS = 2
USES_ROLLBACK = False


@dataclass(frozen=True)
class AdaptiveRhoDecision:
    """One auditable replay-ratio decision made before a training round."""

    forgetting: float
    previous_replay_ratio: float
    replay_ratio: float
    reason: str


def _canonical_palette_value(value: float) -> float:
    value = float(value)
    for candidate in REPLAY_PALETTE:
        if math.isclose(value, candidate, rel_tol=0.0, abs_tol=1e-12):
            return candidate
    raise ValueError(f"previous replay ratio must be one of {REPLAY_PALETTE}, got {value}")


def select_replay_ratio(
    *,
    round_id: int,
    previous_replay_ratio: float,
    forgetting: float,
) -> AdaptiveRhoDecision:
    """Select a replay level without changing LR, client scales, or model state."""
    previous = _canonical_palette_value(previous_replay_ratio)
    forgetting = float(forgetting)

    if int(round_id) < WARMUP_ROUNDS:
        return AdaptiveRhoDecision(forgetting, previous, 0.5, "warmup")
    if not math.isfinite(forgetting):
        return AdaptiveRhoDecision(forgetting, previous, previous, "retain_nonfinite")
    if forgetting == FORGETTING_THRESHOLD:
        return AdaptiveRhoDecision(forgetting, previous, previous, "retain_equal")

    index = REPLAY_PALETTE.index(previous)
    if forgetting > FORGETTING_THRESHOLD:
        if index == len(REPLAY_PALETTE) - 1:
            return AdaptiveRhoDecision(forgetting, previous, previous, "at_upper_bound")
        return AdaptiveRhoDecision(
            forgetting, previous, REPLAY_PALETTE[index + 1], "increase_forgetting"
        )

    if index == 0:
        return AdaptiveRhoDecision(forgetting, previous, previous, "at_lower_bound")
    return AdaptiveRhoDecision(
        forgetting, previous, REPLAY_PALETTE[index - 1], "decrease_forgetting"
    )


def build_action(*, n_clients: int, fixed_lr: float, decision: AdaptiveRhoDecision) -> Dict:
    """Build the all-client FedAvg action with immutable LR and unit scales."""
    if int(n_clients) <= 0:
        raise ValueError("n_clients must be positive")
    fixed_lr = float(fixed_lr)
    if not math.isfinite(fixed_lr) or fixed_lr <= 0.0:
        raise ValueError("fixed_lr must be finite and positive")
    return {
        "lr": fixed_lr,
        "client_selection_k": int(n_clients),
        "aggregation": {"method": "FedAvg"},
        "client_params": [
            {
                "id": int(client_id),
                "replay_ratio": decision.replay_ratio,
                "lr_scale": 1.0,
                "ewc_lambda": 0.0,
            }
            for client_id in range(int(n_clients))
        ],
        "policy_source": "AdaptiveRho",
    }
