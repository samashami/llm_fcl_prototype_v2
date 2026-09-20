"""Deterministic random-subspace orientation control for one resumed branch."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import torch


ENERGY_MATCH_RTOL = 1e-5
ENERGY_MATCH_ATOL = 1e-12


def random_basis_seed(experiment_seed: int, layer_name: str) -> int:
    """Stable private-generator seed; never use Python's randomized hash()."""
    payload = f"random_projection_norm_matched\0{int(experiment_seed)}\0{layer_name}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def random_orthonormal_basis(
    experiment_seed: int, layer_name: str, input_dim: int, rank: int
) -> tuple[torch.Tensor, int, str]:
    """Construct a fixed CPU float64 Q without mutating any global RNG state."""
    if input_dim <= 0 or rank <= 0 or rank > input_dim:
        raise ValueError(f"invalid random basis shape {(input_dim, rank)!r}")
    seed = random_basis_seed(experiment_seed, layer_name)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn(
        (int(input_dim), int(rank)), generator=generator, device="cpu", dtype=torch.float64
    )
    basis = torch.linalg.qr(gaussian, mode="reduced").Q.contiguous()
    return basis, seed, tensor_sha256(basis)


def _energy(value: torch.Tensor) -> float:
    return float(value.detach().square().sum().item())


def _contractivity_tolerance(raw_energy: float) -> float:
    return ENERGY_MATCH_ATOL + ENERGY_MATCH_RTOL * abs(raw_energy)


def random_projection_norm_matched_update(
    raw: torch.Tensor, learned_basis: torch.Tensor, random_basis: torch.Tensor
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Match learned-Phi retained energy while applying the random-Q direction."""
    if raw.ndim != 2:
        raise ValueError("raw update must be a matrix of output rows")
    phi = learned_basis.to(device=raw.device, dtype=raw.dtype)
    q = random_basis.to(device=raw.device, dtype=raw.dtype)
    if phi.shape[0] != raw.shape[1] or q.shape[0] != raw.shape[1]:
        raise ValueError("basis input dimension does not match raw update")
    if phi.shape[1] != q.shape[1]:
        raise ValueError("random basis rank must equal learned basis rank")

    learned = raw - (raw @ phi) @ phi.T
    random_projected = raw - (raw @ q) @ q.T
    raw_energy = _energy(raw)
    learned_energy = _energy(learned)
    random_energy = _energy(random_projected)
    values = (raw_energy, learned_energy, random_energy)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"random-Phi control encountered non-finite energy: {values!r}")
    tolerance = _contractivity_tolerance(raw_energy)
    if learned_energy > raw_energy + tolerance:
        raise RuntimeError(
            "random-Phi control learned projection is non-contractive: "
            f"raw_energy={raw_energy!r}, learned_energy={learned_energy!r}, "
            f"tolerance={tolerance!r}"
        )
    if random_energy > raw_energy + tolerance:
        raise RuntimeError(
            "random-Phi control random projection is non-contractive: "
            f"raw_energy={raw_energy!r}, random_energy={random_energy!r}, "
            f"tolerance={tolerance!r}"
        )

    zero_tolerance = ENERGY_MATCH_ATOL + ENERGY_MATCH_RTOL * max(raw_energy, learned_energy)
    if random_energy <= zero_tolerance:
        if learned_energy > zero_tolerance:
            raise RuntimeError(
                "random-Phi control cannot match nonzero learned energy with an "
                "effectively zero random-projection energy"
            )
        scale = 1.0
    else:
        scale = math.sqrt(learned_energy / random_energy)
    if not math.isfinite(scale):
        raise RuntimeError(f"random-Phi control produced non-finite scale {scale!r}")
    applied = random_projected * scale
    applied_energy = _energy(applied)
    if not math.isfinite(applied_energy) or not math.isclose(
        applied_energy, learned_energy, rel_tol=ENERGY_MATCH_RTOL, abs_tol=ENERGY_MATCH_ATOL
    ):
        raise RuntimeError(
            "random-Phi control energy-match guard failed: "
            f"applied_energy={applied_energy!r}, learned_energy={learned_energy!r}, "
            f"rtol={ENERGY_MATCH_RTOL!r}, atol={ENERGY_MATCH_ATOL!r}"
        )
    return applied, {
        "raw_update_energy": raw_energy,
        "learned_counterfactual_energy": learned_energy,
        "random_projection_energy_before_match": random_energy,
        "applied_random_energy": applied_energy,
        "learned_retained_energy_fraction": learned_energy / raw_energy if raw_energy else 1.0,
        "random_retained_energy_fraction_before_match": random_energy / raw_energy if raw_energy else 1.0,
        "random_rescale_factor": scale,
        "energy_match_abs_error": abs(applied_energy - learned_energy),
        "energy_match_guard_passed": True,
    }


@dataclass
class RandomPhiPatch:
    experiment_seed: int
    original_method: Any = None
    bases: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def install(self) -> "RandomPhiPatch":
        """Install only the treatment hook and retain enough provenance for metadata."""
        from src.instrumentation import subspace

        if self.original_method is not None:
            raise RuntimeError("random-Phi patch is already installed")
        self.original_method = subspace._ModelMonitor.online_shrink_parameter_updates
        patch = self

        @torch.no_grad()
        def replacement(monitor, weights_before: Mapping[str, torch.Tensor], projection_lambda: float):
            if float(projection_lambda) != 1.0:
                raise ValueError("random_projection_norm_matched requires projection_lambda=1")
            records = []
            for target in monitor.targets:
                weight_before = weights_before.get(target.name)
                if weight_before is None:
                    continue
                weight = monitor.modules[target.name].weight
                raw = (weight.detach() - weight_before).reshape(weight.shape[0], -1)
                phi = monitor.bank.basis(target.name, device=raw.device, dtype=raw.dtype)
                if phi is None:
                    continue
                entry = patch.bases.get(target.name)
                if entry is None:
                    cpu_basis, seed, fingerprint = random_orthonormal_basis(
                        patch.experiment_seed, target.name, int(phi.shape[0]), int(phi.shape[1])
                    )
                    entry = {
                        "basis": cpu_basis,
                        "seed": seed,
                        "sha256": fingerprint,
                        "rank": int(phi.shape[1]),
                        "input_dim": int(phi.shape[0]),
                    }
                    patch.bases[target.name] = entry
                applied, record = random_projection_norm_matched_update(
                    raw, phi, entry["basis"]
                )
                from src.instrumentation.subspace import realized_update_energy

                # Preserve the controller's standard energy columns for its usual CSV
                # aggregation, then add the orientation-control-specific accounting.
                record = {**realized_update_energy(raw, applied), **record}
                weight.copy_(
                    (weight_before.reshape(raw.shape[0], -1) + applied).reshape_as(weight)
                )
                record.update(
                    {
                        "layer": target.name,
                        "basis_rank": int(phi.shape[1]),
                        "learned_basis_rank": int(phi.shape[1]),
                        "random_basis_rank": int(entry["rank"]),
                        "random_basis_seed": int(entry["seed"]),
                        "random_basis_sha256": entry["sha256"],
                        "projection_lambda": 1.0,
                        "experimental_mode": "random_projection_norm_matched",
                    }
                )
                records.append(record)
            return records

        subspace._ModelMonitor.online_shrink_parameter_updates = replacement
        return self

    def restore(self) -> None:
        if self.original_method is None:
            return
        from src.instrumentation import subspace

        subspace._ModelMonitor.online_shrink_parameter_updates = self.original_method
        self.original_method = None
