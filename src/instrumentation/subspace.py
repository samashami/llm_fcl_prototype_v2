"""Read-only gradient/subspace measurements for continual learning.

The code in this module deliberately never assigns to ``parameter.grad`` and
never participates in an optimizer step.  A future projection implementation
can reuse :class:`SubspaceBank`, whose bases are stored in the input-feature
coordinates of the corresponding weight tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LayerTarget:
    """A module whose input features protect its weight-gradient columns."""

    name: str
    module_name: str


DEFAULT_RESNET18_TARGETS = (
    LayerTarget("layer4_1_conv2", "layer4.1.conv2"),
    LayerTarget("fc", "fc"),
)

SUPPORTED_PROJECTION_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


class SubspaceBank:
    """Incrementally aggregates orthonormal GPM-style feature bases on CPU."""

    def __init__(self, explained_energy: float = 0.95, max_rank: int = 32):
        if not 0.0 < explained_energy <= 1.0:
            raise ValueError("explained_energy must be in (0, 1]")
        if max_rank <= 0:
            raise ValueError("max_rank must be positive")
        self.explained_energy = float(explained_energy)
        self.max_rank = int(max_rank)
        self.bases: Dict[str, torch.Tensor] = {}

    def basis(self, name: str, *, device=None, dtype=None) -> Optional[torch.Tensor]:
        basis = self.bases.get(name)
        if basis is None:
            return None
        return basis.to(device=device, dtype=dtype)

    def ranks(self) -> Dict[str, int]:
        return {name: int(phi.shape[1]) for name, phi in self.bases.items()}

    def orthonormality_errors(self) -> Dict[str, float]:
        errors = {}
        for name, phi in self.bases.items():
            eye = torch.eye(phi.shape[1], dtype=phi.dtype, device=phi.device)
            errors[name] = float(torch.linalg.matrix_norm(phi.T @ phi - eye).item())
        return errors

    @torch.no_grad()
    def update(self, activation_matrices: Mapping[str, torch.Tensor]) -> float:
        """Append residual SVD directions needed to represent a new phase."""
        started = time.perf_counter()
        for name, matrix in activation_matrices.items():
            if matrix is None or matrix.numel() == 0:
                continue
            a = matrix.detach().to(device="cpu", dtype=torch.float32)
            if a.ndim != 2:
                raise ValueError(f"activation matrix for {name!r} must be 2-D")

            old = self.bases.get(name)
            if old is not None and old.shape[0] != a.shape[0]:
                raise ValueError(f"feature dimension changed for {name!r}")

            total_energy = float(a.square().sum().item())
            if total_energy <= torch.finfo(a.dtype).eps:
                continue

            if old is None:
                residual = a
                represented_energy = 0.0
                old_rank = 0
            else:
                coefficients = old.T @ a
                residual = a - old @ coefficients
                represented_energy = float(coefficients.square().sum().item())
                old_rank = int(old.shape[1])

            available = min(self.max_rank - old_rank, residual.shape[0], residual.shape[1])
            needed_energy = self.explained_energy * total_energy - represented_energy
            if available <= 0 or needed_energy <= 0.0:
                continue

            u, singular_values, _ = torch.linalg.svd(residual, full_matrices=False)
            cumulative = torch.cumsum(singular_values.square(), dim=0)
            needed = torch.tensor(needed_energy, dtype=cumulative.dtype)
            add_rank = int(torch.searchsorted(cumulative, needed).item()) + 1
            add_rank = min(add_rank, available)
            additions = u[:, :add_rank]
            combined = additions if old is None else torch.cat((old, additions), dim=1)
            # Residual SVD is theoretically orthogonal to old. QR removes drift.
            self.bases[name] = torch.linalg.qr(combined, mode="reduced").Q.contiguous()

        return time.perf_counter() - started


def soft_project_rows(
    matrix: torch.Tensor, phi: torch.Tensor, lambda_value: float
) -> torch.Tensor:
    """Soft-project matrix rows away from the column span of ``phi``."""
    if not isinstance(matrix, torch.Tensor) or not isinstance(phi, torch.Tensor):
        raise TypeError("matrix and phi must be torch.Tensor instances")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D [output_features, input_features]")
    if phi.ndim != 2:
        raise ValueError("phi must be 2-D [input_features, rank]")
    if phi.numel() == 0 or phi.shape[1] == 0:
        raise ValueError("phi must contain at least one basis vector")
    if phi.shape[0] != matrix.shape[1]:
        raise ValueError(
            "phi input dimension must match matrix input_features "
            f"({phi.shape[0]} != {matrix.shape[1]})"
        )
    if not matrix.is_floating_point() or not phi.is_floating_point():
        raise TypeError("matrix and phi must have floating-point dtypes")
    if matrix.dtype != phi.dtype:
        raise ValueError("matrix and phi must have the same dtype")
    if matrix.device != phi.device:
        raise ValueError("matrix and phi must be on the same device")
    if (
        not isinstance(lambda_value, Real)
        or isinstance(lambda_value, bool)
        or float(lambda_value) not in SUPPORTED_PROJECTION_LAMBDAS
    ):
        raise ValueError(
            f"lambda_value must be one of {SUPPORTED_PROJECTION_LAMBDAS}"
        )

    lambda_value = float(lambda_value)
    if lambda_value == 0.0:
        return matrix.clone()
    return matrix - lambda_value * (matrix @ phi) @ phi.T


def scalar_shrink_rows(matrix: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale an optimizer displacement without changing its direction."""
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D [output_features, input_features]")
    if not matrix.is_floating_point():
        raise TypeError("matrix must have a floating-point dtype")
    if (
        not isinstance(factor, Real)
        or isinstance(factor, bool)
        or not math.isfinite(float(factor))
        or not 0.0 <= float(factor) <= 1.0
    ):
        raise ValueError("shrinkage factor must be a finite value in [0, 1]")
    return matrix * float(factor)


def realized_update_energy(
    raw_update: torch.Tensor, projected_update: torch.Tensor
) -> Dict[str, float]:
    """Account for the realized raw and applied optimizer displacements."""
    if raw_update.shape != projected_update.shape:
        raise ValueError("raw and projected updates must have identical shapes")
    if raw_update.device != projected_update.device:
        raise ValueError("raw and projected updates must be on the same device")
    if raw_update.dtype != projected_update.dtype:
        raise ValueError("raw and projected updates must have the same dtype")

    raw_energy = float(raw_update.detach().square().sum().item())
    projected_energy = float(projected_update.detach().square().sum().item())
    removed = raw_update.detach() - projected_update.detach()
    removed_displacement_energy = float(removed.square().sum().item())
    raw_norm = math.sqrt(raw_energy)
    projected_norm = math.sqrt(projected_energy)
    if raw_energy == 0.0:
        retained_norm_fraction = float("nan")
        retained_energy_fraction = float("nan")
    else:
        retained_norm_fraction = projected_norm / raw_norm
        retained_energy_fraction = projected_energy / raw_energy
    return {
        "raw_update_norm": raw_norm,
        "raw_update_energy": raw_energy,
        "projected_update_norm": projected_norm,
        "projected_update_energy": projected_energy,
        "removed_displacement_norm": math.sqrt(removed_displacement_energy),
        "removed_displacement_energy": removed_displacement_energy,
        "removed_energy": max(0.0, raw_energy - projected_energy),
        "retained_norm_fraction": retained_norm_fraction,
        "retained_energy_fraction": retained_energy_fraction,
    }


def aggregate_update_energy_records(
    records: Iterable[Mapping[str, object]], rounds: Optional[Iterable[int]] = None
) -> List[Dict[str, float]]:
    """Aggregate step/layer energies by round using ratios of energy sums."""
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for record in records:
        grouped.setdefault(int(record["round"]), []).append(record)
    if rounds is not None:
        for round_id in rounds:
            grouped.setdefault(int(round_id), [])

    result = []
    energy_fields = (
        "raw_update_energy",
        "projected_update_energy",
        "removed_displacement_energy",
        "removed_energy",
    )
    for round_id in sorted(grouped):
        rows = grouped[round_id]
        sums = {
            f"{field}_sum": sum(float(row[field]) for row in rows)
            for field in energy_fields
        }
        raw = sums["raw_update_energy_sum"]
        retained = sums["projected_update_energy_sum"]
        if raw > 0.0:
            retained_energy_fraction = retained / raw
            retained_norm_fraction = math.sqrt(retained_energy_fraction)
            removed_energy_fraction = sums["removed_energy_sum"] / raw
        else:
            retained_energy_fraction = float("nan")
            retained_norm_fraction = float("nan")
            removed_energy_fraction = float("nan")
        result.append(
            {
                "round": round_id,
                "step_layer_count": len(rows),
                **sums,
                "raw_update_norm": math.sqrt(raw),
                "projected_update_norm": math.sqrt(retained),
                "removed_displacement_norm": math.sqrt(
                    sums["removed_displacement_energy_sum"]
                ),
                "retained_norm_fraction": retained_norm_fraction,
                "retained_energy_fraction": retained_energy_fraction,
                "removed_energy_fraction": removed_energy_fraction,
            }
        )
    return result


def gradient_energy(
    gradient: torch.Tensor, basis: torch.Tensor, eps: float = 1e-12
) -> Dict[str, float]:
    """Return a read-only orthogonal energy decomposition for one weight."""
    g = gradient.detach().reshape(gradient.shape[0], -1)
    phi = basis.to(device=g.device, dtype=g.dtype)
    if phi.shape[0] != g.shape[1]:
        raise ValueError("basis and gradient input-feature dimensions do not match")
    coefficients = g @ phi
    inside = float(coefficients.square().sum().item())
    total = float(g.square().sum().item())
    outside = max(0.0, total - inside)
    beta_hat = inside / (total + eps)
    # For an orthogonal projection this is both ||g_perp||/||g|| and the
    # cosine between g and its hypothetical projected counterpart.
    rho_hat = math.sqrt(outside / (total + eps))
    return {
        "inside": inside,
        "outside": outside,
        "total": total,
        "beta_hat": beta_hat,
        "rho_hat": rho_hat,
    }


class _ModelMonitor:
    def __init__(
        self,
        model: nn.Module,
        bank: SubspaceBank,
        targets: Sequence[LayerTarget],
        samples_per_batch: int,
        samples_per_phase: int,
    ):
        self.model = model
        self.bank = bank
        self.targets = tuple(targets)
        self.samples_per_batch = int(samples_per_batch)
        self.samples_per_phase = int(samples_per_phase)
        modules = dict(model.named_modules())
        self.modules: Dict[str, nn.Module] = {}
        self.handles = []
        for target in self.targets:
            if target.module_name not in modules:
                raise ValueError(f"model has no module {target.module_name!r}")
            module = modules[target.module_name]
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                raise TypeError("subspace targets must be Conv2d or Linear modules")
            self.modules[target.name] = module
            self.handles.append(module.register_forward_pre_hook(self._make_hook(target.name)))
        self.collect_enabled = False
        self.samples: Dict[str, List[torch.Tensor]] = {}
        self.sample_counts: Dict[str, int] = {}
        self.inside = 0.0
        self.outside = 0.0
        self.measurements = 0
        self.overhead_seconds = 0.0

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs):
            if not self.collect_enabled or not self.model.training or not inputs:
                return
            remaining = self.samples_per_phase - self.sample_counts.get(name, 0)
            if remaining <= 0:
                return
            started = time.perf_counter()
            with torch.no_grad():
                x = inputs[0].detach()
                if isinstance(module, nn.Conv2d):
                    # Unfold only a few images; unfolding a full 224px ResNet
                    # batch would create a large temporary matrix before sampling.
                    desired_count = min(self.samples_per_batch, remaining)
                    image_count = min(x.shape[0], desired_count, 4)
                    image_indices = torch.linspace(
                        0, x.shape[0] - 1, steps=image_count, device=x.device
                    ).long()
                    x = x.index_select(0, image_indices)
                    patches = F.unfold(
                        x,
                        kernel_size=module.kernel_size,
                        dilation=module.dilation,
                        padding=module.padding,
                        stride=module.stride,
                    ).transpose(1, 2).reshape(-1, module.weight[0].numel())
                else:
                    patches = x.reshape(-1, x.shape[-1])
                count = min(self.samples_per_batch, remaining, patches.shape[0])
                if count > 0:
                    # Evenly spaced indices are deterministic and cover images/spatial sites.
                    indices = torch.linspace(
                        0, patches.shape[0] - 1, steps=count, device=patches.device
                    ).long()
                    selected = patches.index_select(0, indices).T.to("cpu", torch.float32)
                    self.samples.setdefault(name, []).append(selected)
                    self.sample_counts[name] = self.sample_counts.get(name, 0) + count
            self.overhead_seconds += time.perf_counter() - started

        return hook

    def begin_round(self, collect_activations: bool) -> None:
        self.collect_enabled = bool(collect_activations)
        self.samples = {}
        self.sample_counts = {}
        self.inside = 0.0
        self.outside = 0.0
        self.measurements = 0
        self.overhead_seconds = 0.0

    @torch.no_grad()
    def measure_gradients(self) -> None:
        """Read selected gradients after backward; never mutate them."""
        if not self.bank.bases:
            return
        started = time.perf_counter()
        measured = False
        for target in self.targets:
            module = self.modules[target.name]
            if module.weight.grad is None:
                continue
            phi = self.bank.basis(
                target.name, device=module.weight.grad.device, dtype=module.weight.grad.dtype
            )
            if phi is None:
                continue
            values = gradient_energy(module.weight.grad, phi)
            self.inside += values["inside"]
            self.outside += values["outside"]
            measured = True
        if measured:
            self.measurements += 1
        self.overhead_seconds += time.perf_counter() - started

    @torch.no_grad()
    def snapshot_protected_weights(self) -> Dict[str, torch.Tensor]:
        """Clone weights whose protected bases currently exist."""
        return {
            target.name: self.modules[target.name].weight.detach().clone()
            for target in self.targets
            if self.bank.basis(target.name) is not None
        }

    @torch.no_grad()
    def snapshot_target_weights(
        self, layer_names: Optional[Iterable[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Clone selected controlled weights without consulting the subspace bank."""
        selected = (
            {target.name for target in self.targets}
            if layer_names is None
            else set(layer_names)
        )
        return {
            target.name: self.modules[target.name].weight.detach().clone()
            for target in self.targets
            if target.name in selected
        }

    @torch.no_grad()
    def soft_project_parameter_updates(
        self, weights_before: Mapping[str, torch.Tensor], lambda_value: float
    ) -> List[Dict[str, float]]:
        """Replace realized optimizer displacements with soft-projected ones."""
        records = []
        for target in self.targets:
            weight_before = weights_before.get(target.name)
            if weight_before is None:
                continue
            weight = self.modules[target.name].weight
            displacement = (weight.detach() - weight_before).reshape(weight.shape[0], -1)
            phi = self.bank.basis(
                target.name,
                device=displacement.device,
                dtype=displacement.dtype,
            )
            if phi is None:
                continue
            projected = soft_project_rows(displacement, phi, lambda_value)
            weight.copy_((weight_before.reshape(weight.shape[0], -1) + projected).reshape_as(weight))
            record = realized_update_energy(displacement, projected)
            record.update(
                {
                    "layer": target.name,
                    "basis_rank": int(phi.shape[1]),
                    "projection_lambda": float(lambda_value),
                    "shrinkage_factor": float("nan"),
                }
            )
            records.append(record)
        return records

    @torch.no_grad()
    def shrink_parameter_updates(
        self,
        weights_before: Mapping[str, torch.Tensor],
        factors_by_layer: Mapping[str, float],
    ) -> List[Dict[str, float]]:
        """Scale realized optimizer displacements without reading any basis."""
        records = []
        for target in self.targets:
            if target.name not in factors_by_layer:
                raise ValueError(f"missing shrinkage factor for layer {target.name!r}")
            factor = float(factors_by_layer[target.name])
            if factor == 1.0:
                continue
            weight_before = weights_before.get(target.name)
            if weight_before is None:
                raise ValueError(f"missing pre-step weight for layer {target.name!r}")
            weight = self.modules[target.name].weight
            displacement = (weight.detach() - weight_before).reshape(weight.shape[0], -1)
            shrunk = scalar_shrink_rows(displacement, factor)
            weight.copy_((weight_before.reshape(weight.shape[0], -1) + shrunk).reshape_as(weight))
            record = realized_update_energy(displacement, shrunk)
            record.update(
                {
                    "layer": target.name,
                    "basis_rank": float("nan"),
                    "projection_lambda": 0.0,
                    "shrinkage_factor": factor,
                }
            )
            records.append(record)
        return records

    def activation_matrices(self) -> Dict[str, torch.Tensor]:
        return {
            name: torch.cat(parts, dim=1)
            for name, parts in self.samples.items()
            if parts
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class SubspaceInstrumentation:
    """Coordinates one shared protected bank across federated clients."""

    def __init__(
        self,
        models: Iterable[nn.Module],
        targets: Sequence[LayerTarget] = DEFAULT_RESNET18_TARGETS,
        explained_energy: float = 0.95,
        max_rank: int = 32,
        samples_per_batch: int = 8,
        samples_per_phase: int = 64,
    ):
        self.bank = SubspaceBank(explained_energy=explained_energy, max_rank=max_rank)
        self.monitors = [
            _ModelMonitor(
                model,
                self.bank,
                targets,
                samples_per_batch=samples_per_batch,
                samples_per_phase=samples_per_phase,
            )
            for model in models
        ]
        self.completed_phases = set()
        self._phase_id = None
        self._ranks_at_start: Dict[str, int] = {}
        self._errors_at_start: Dict[str, float] = {}

    def begin_round(self, phase_id: int) -> None:
        self._phase_id = int(phase_id)
        self._ranks_at_start = self.bank.ranks()
        self._errors_at_start = self.bank.orthonormality_errors()
        collect = self._phase_id not in self.completed_phases
        for monitor in self.monitors:
            monitor.begin_round(collect_activations=collect)

    def end_round(self) -> Dict[str, object]:
        if self._phase_id is None:
            raise RuntimeError("begin_round must be called before end_round")
        for monitor in self.monitors:
            monitor.collect_enabled = False

        construction_seconds = 0.0
        if self._phase_id not in self.completed_phases:
            by_layer: Dict[str, List[torch.Tensor]] = {}
            for monitor in self.monitors:
                for name, matrix in monitor.activation_matrices().items():
                    by_layer.setdefault(name, []).append(matrix)
            combined = {name: torch.cat(parts, dim=1) for name, parts in by_layer.items()}
            construction_seconds = self.bank.update(combined)
            self.completed_phases.add(self._phase_id)

        inside = sum(m.inside for m in self.monitors)
        outside = sum(m.outside for m in self.monitors)
        total = inside + outside
        eps = 1e-12
        measurements = sum(m.measurements for m in self.monitors)
        measurement_seconds = sum(m.overhead_seconds for m in self.monitors)
        ranks_after = self.bank.ranks()
        return {
            "protected_basis_exists": bool(self._ranks_at_start),
            "protected_basis_rank": sum(self._ranks_at_start.values()),
            "protected_basis_rank_by_layer": dict(self._ranks_at_start),
            "protected_basis_rank_after_update": sum(ranks_after.values()),
            "protected_orthonormality_error": max(self._errors_at_start.values(), default=0.0),
            "protected_orthonormality_error_by_layer": dict(self._errors_at_start),
            "gradient_energy_inside": inside,
            "gradient_energy_outside": outside,
            "beta_hat": inside / (total + eps) if measurements else float("nan"),
            "rho_hat": math.sqrt(outside / (total + eps)) if measurements else float("nan"),
            "basis_construction_seconds": construction_seconds,
            "measurement_overhead_seconds": measurement_seconds + construction_seconds,
            "gradient_measurement_count": measurements,
        }

    def close(self) -> None:
        for monitor in self.monitors:
            monitor.close()
