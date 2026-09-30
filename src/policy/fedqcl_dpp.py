"""FedQCL drift-plus-penalty primitives for shared-head Data-IL adaptation.

The adapted controller keeps client-local queues indexed by historical domain.
Each stage fixes reference losses on the memory available at stage start. The
reference term is constant with respect to the current model, while the
queue-weighted replay losses contribute gradients to local optimization.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch.nn import functional as F


def fedqcl_objective(
    current_loss: torch.Tensor,
    replay_losses: Mapping[int, torch.Tensor],
    queues: Mapping[int, float],
    reference_losses: Mapping[int, float],
    *,
    penalty_weight: float = 200.0,
) -> torch.Tensor:
    """Compute V*current loss + sum_k Q_k*(current replay loss-reference).

    The scalar reference is deliberately detached: it affects the objective
    value and queue interpretation, but has no gradient with respect to the
    model being optimized.
    """
    if penalty_weight <= 0:
        raise ValueError("FedQCL penalty_weight must be positive")
    total = float(penalty_weight) * current_loss
    for group_id, loss in replay_losses.items():
        q = float(queues.get(int(group_id), 0.0))
        if q < 0:
            raise ValueError("FedQCL queues must be nonnegative")
        if q:
            total = total + q * (
                loss - float(reference_losses.get(int(group_id), 0.0))
            )
    return total


def update_queue(
    previous: float, current_loss: float, reference_loss: float, delta: float
) -> Tuple[float, float]:
    """Return ``(max(0,Q+loss-reference-delta), loss-reference-delta)``."""
    values = (previous, current_loss, reference_loss, delta)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("FedQCL queue inputs must be finite")
    if previous < 0 or delta < 0:
        raise ValueError("FedQCL queue and delta must be nonnegative")
    violation = float(current_loss) - float(reference_loss) - float(delta)
    return max(0.0, float(previous) + violation), violation


@dataclass
class FedQCLState:
    """Per-client controller state; queue values persist across stages."""

    queues: Dict[int, float] = field(default_factory=dict)
    reference_stage: Optional[int] = None
    reference_losses: Dict[int, float] = field(default_factory=dict)
    reference_model_sha256: str = ""

    def begin_stage(
        self, stage: int, reference_losses: Mapping[int, float], model_sha256: str
    ) -> bool:
        """Fix stage reference once; return True only when a new stage begins."""
        stage = int(stage)
        if self.reference_stage == stage:
            return False
        if self.reference_stage is not None and stage < self.reference_stage:
            raise ValueError("FedQCL stage ids must be monotone")
        self.reference_stage = stage
        self.reference_losses = {
            int(k): float(v) for k, v in reference_losses.items()
        }
        self.reference_model_sha256 = str(model_sha256)
        for group_id in self.reference_losses:
            self.queues.setdefault(group_id, 0.0)
        return True

    def update(self, current_losses: Mapping[int, float], delta: float):
        records = {}
        for group_id, reference_loss in self.reference_losses.items():
            if int(group_id) not in current_losses:
                continue
            group_id = int(group_id)
            before = float(self.queues.get(group_id, 0.0))
            after, violation = update_queue(
                before, float(current_losses[group_id]), reference_loss, delta
            )
            self.queues[group_id] = after
            records[group_id] = {
                "reference_loss": reference_loss,
                "current_loss": float(current_losses[group_id]),
                "violation": violation,
                "queue_before": before,
                "queue_after": after,
            }
        return records

    def state_dict(self):
        return {
            "queues": {int(k): float(v) for k, v in self.queues.items()},
            "reference_stage": self.reference_stage,
            "reference_losses": {
                int(k): float(v) for k, v in self.reference_losses.items()
            },
            "reference_model_sha256": self.reference_model_sha256,
        }

    def load_state_dict(self, payload):
        self.queues = {int(k): float(v) for k, v in payload["queues"].items()}
        self.reference_stage = payload["reference_stage"]
        self.reference_losses = {
            int(k): float(v) for k, v in payload["reference_losses"].items()
        }
        self.reference_model_sha256 = str(payload["reference_model_sha256"])


class FedQCLMemory:
    """Bounded, task/domain-stratified client memory with deterministic sampling."""

    def __init__(self, capacity: int = 2000, seed: int = 0):
        if capacity < 1:
            raise ValueError("FedQCL memory capacity must be positive")
        self.capacity = int(capacity)
        self._rng = random.Random(int(seed))
        self._groups: Dict[int, list] = {}

    @property
    def group_ids(self):
        return tuple(sorted(self._groups))

    def __len__(self):
        return sum(len(rows) for rows in self._groups.values())

    def add_domain_batch(self, group_id: int, x: torch.Tensor, y: torch.Tensor):
        group_id = int(group_id)
        if x.size(0) != y.size(0):
            raise ValueError("memory features and labels must have equal lengths")
        if group_id in self._groups:
            raise ValueError(f"FedQCL domain {group_id} was already admitted")
        rows = [(x[i].detach().cpu().clone(), y[i].detach().cpu().clone())
                for i in range(len(y))]
        self._groups[group_id] = rows
        quota = max(1, self.capacity // len(self._groups))
        for gid in list(self._groups):
            group = self._groups[gid]
            if len(group) > quota:
                self._groups[gid] = self._rng.sample(group, quota)
        # If fewer examples exist than slots, retain all available examples.
        while len(self) > self.capacity:
            largest = max(self._groups, key=lambda k: len(self._groups[k]))
            self._groups[largest].pop(self._rng.randrange(len(self._groups[largest])))

def add_domain_batches(memory: FedQCLMemory, group_id: int, batches) -> int:
    """Admit all minibatches for one domain as a single memory group."""
    xs, ys = [], []
    for x, y in batches:
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError(f"FedQCL domain {int(group_id)} has no memory examples")
    all_x = torch.cat(xs, dim=0)
    all_y = torch.cat(ys, dim=0)
    memory.add_domain_batch(group_id, all_x, all_y)
    return int(all_y.numel())


    def sample_by_group(self, count: int, device) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        groups = [gid for gid in self.group_ids if self._groups[gid]]
        count = min(max(0, int(count)), len(self))
        if not groups or count == 0:
            return {}
        base, remainder = divmod(count, len(groups))
        result = {}
        for index, gid in enumerate(groups):
            n = base + (index < remainder)
            if n == 0:
                continue
            rows = self._groups[gid]
            chosen = self._rng.sample(rows, min(n, len(rows)))
            result[gid] = (
                torch.stack([row[0] for row in chosen]).to(device),
                torch.stack([row[1] for row in chosen]).to(device),
            )
        return result

    def iter_group_batches(self, batch_size: int = 256):
        for gid in self.group_ids:
            rows = self._groups[gid]
            for start in range(0, len(rows), batch_size):
                selected = rows[start:start + batch_size]
                if selected:
                    yield gid, (
                        torch.stack([row[0] for row in selected]),
                        torch.stack([row[1] for row in selected]),
                    )

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "groups": {int(k): list(v) for k, v in self._groups.items()},
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, payload):
        self.capacity = int(payload["capacity"])
        self._groups = {int(k): list(v) for k, v in payload["groups"].items()}
        self._rng.setstate(payload["rng_state"])


@torch.no_grad()
def evaluate_group_losses(model, memory: FedQCLMemory, device, batch_size: int = 256):
    """Evaluate each historical group without changing model or global RNG."""
    was_training = model.training
    model.eval()
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    try:
        for group_id, (x, y) in memory.iter_group_batches(batch_size):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, reduction="sum")
            sums[group_id] = sums.get(group_id, 0.0) + float(loss.item())
            counts[group_id] = counts.get(group_id, 0) + int(y.numel())
    finally:
        model.train(was_training)
    return {gid: sums[gid] / counts[gid] for gid in sums if counts[gid]}
