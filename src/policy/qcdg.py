"""Pure queue and drift-gate helpers for optional QCDG-LMSS experiments."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class QueueState:
    """Virtual queue state derived from current and replay losses."""

    Q: float = 0.0
    P: float = 0.0
    delta_q: float = 0.0
    ema_l_new: float = 0.0


def update_queue(
    state: QueueState,
    *,
    l_new: float,
    l_buffer: float,
    beta: float = 0.9,
) -> QueueState:
    """Apply Q=max(0,Q_prev+L_buffer-L_new-delta), with delta=EMA(L_new)."""
    l_new, l_buffer, beta = float(l_new), float(l_buffer), float(beta)
    if not all(math.isfinite(value) for value in (l_new, l_buffer, beta)):
        raise ValueError("QCDG queue inputs must be finite")
    if not 0.0 <= beta < 1.0:
        raise ValueError("QCDG EMA beta must be in [0, 1)")
    ema = l_new if state.ema_l_new == 0.0 else beta * state.ema_l_new + (1.0 - beta) * l_new
    queue = max(0.0, state.Q + l_buffer - l_new - ema)
    return QueueState(Q=queue, P=l_new, delta_q=queue - state.Q, ema_l_new=ema)


def queue_fields_for_state(state: QueueState) -> Dict[str, float]:
    """Finite prompt/state features, including bounded variants."""
    return {
        "Q": float(state.Q),
        "P": float(state.P),
        "delta_q": float(state.delta_q),
        "ema_l_new": float(state.ema_l_new),
        "Q_norm": float(math.tanh(max(0.0, min(state.Q, 50.0)))),
        "P_norm": float(math.tanh(max(0.0, min(state.P, 50.0)))),
        "delta_q_norm": float(math.tanh(max(-50.0, min(state.delta_q, 50.0)))),
    }


def should_invoke_lmss(
    *,
    delta_q: float,
    divergence: float,
    delta_acc: float,
    round_id: int,
    tau_q: float = 0.02,
    tau_d: float = 0.05,
    tau_a: float = 0.01,
    last_invoked_round: Optional[int] = None,
    cooldown_rounds: int = 1,
) -> bool:
    """Call on round zero or when queue, divergence, or accuracy drift triggers."""
    if round_id == 0:
        return True
    values = (delta_q, divergence, delta_acc, tau_q, tau_d, tau_a)
    if not all(math.isfinite(float(value)) for value in values):
        return True
    triggered = delta_q > tau_q or divergence > tau_d or abs(delta_acc) > tau_a
    if not triggered:
        return False
    return last_invoked_round is None or round_id - last_invoked_round > cooldown_rounds


def env_flag_true(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default) if value is None or not value.strip() else float(value)
