"""FedProTIP projection-component adaptation for controlled shared-head Data-IL.

This module retains the core FedProTIP/GPM mechanism: after each completed
continual stage, representation subspaces are accumulated and subsequent local
gradients are projected onto their orthogonal complement. Task-ID prediction is
not used because the controlled Data-IL protocol has one shared 100-class output
space at every stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SharedGPMState:
    threshold: float = 0.70
    max_columns: int = 512
    bases: Dict[str, np.ndarray] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    update_seconds: float = 0.0


class FedProTIPProjector:
    """GPM projector implementing the protected-gradient part of FedProTIP."""

    EARLY_FROZEN_PREFIXES = ("conv1.", "bn1.", "layer1.", "layer2.")

    def __init__(self, model: nn.Module, shared: SharedGPMState, seed: int = 42):
        self.model = model
        self.shared = shared
        self.seed = int(seed)

    @staticmethod
    def _conv_layers(model: nn.Module):
        return {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Conv2d)
        }

    @staticmethod
    def _weight_name(module_name: str) -> str:
        return f"{module_name}.weight"

    def freeze_early_blocks(self) -> list[str]:
        """Mirror upstream pretrained-ResNet freezing through layer2.

        Upstream freezes the first 30 backbone parameters, corresponding to the
        torchvision ResNet-18 stem, layer1 and layer2. Explicit prefixes are used
        here so the rule is auditable and independent of parameter enumeration.
        """
        frozen = []
        for name, param in self.model.named_parameters():
            if name.startswith(self.EARLY_FROZEN_PREFIXES):
                param.requires_grad_(False)
                frozen.append(name)
        return frozen

    def zero_backbone_1d_gradients(self) -> int:
        """Preserve upstream later-task handling of 1-D backbone parameters.

        FedProTIP zeroes gradients of one-dimensional backbone parameters during
        projected training. For ResNet-18 these are primarily BatchNorm affine
        parameters. Running statistics remain governed by normal train/eval mode.
        The classifier head is intentionally excluded.
        """
        zeroed = 0
        for name, param in self.model.named_parameters():
            if name.startswith("fc.") or not param.requires_grad:
                continue
            if param.grad is not None and param.ndim == 1:
                param.grad.zero_()
                zeroed += 1
        return zeroed

    def project_gradients(self) -> dict:
        """Apply G <- G - G U U^T to protected convolutional gradients."""
        named_params = dict(self.model.named_parameters())
        total_sq = 0.0
        removed_sq = 0.0
        residual_inside_sq = 0.0
        projected_layers = 0
        for module_name in self._conv_layers(self.model):
            weight_name = self._weight_name(module_name)
            param = named_params.get(weight_name)
            basis = self.shared.bases.get(weight_name)
            if param is None or param.grad is None or basis is None or basis.size == 0:
                continue
            if not param.requires_grad:
                continue
            grad = param.grad.data
            flat = grad.reshape(grad.shape[0], -1)
            if flat.shape[1] != basis.shape[0]:
                raise RuntimeError(
                    f"FedProTIP basis/gradient mismatch for {weight_name}: "
                    f"gradient={tuple(flat.shape)}, basis={tuple(basis.shape)}"
                )
            U = torch.as_tensor(basis, dtype=flat.dtype, device=flat.device)
            inside = (flat @ U) @ U.T
            total_sq += float((flat * flat).sum().item())
            removed_sq += float((inside * inside).sum().item())
            flat.sub_(inside)
            remaining_inside = (flat @ U) @ U.T
            residual_inside_sq += float((remaining_inside * remaining_inside).sum().item())
            projected_layers += 1
        return {
            "projected_layers": int(projected_layers),
            "gradient_energy_total": float(total_sq),
            "gradient_energy_removed": float(removed_sq),
            "gradient_energy_residual_inside": float(residual_inside_sq),
            "projection_residual_ratio": float(
                residual_inside_sq / max(removed_sq, 1e-30)
            ),
        }

    @staticmethod
    def _uniform_reservoir_merge(
        previous: np.ndarray | None,
        seen_previous: int,
        new_columns: np.ndarray,
        max_columns: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, int]:
        """Uniformly sample from all representation columns seen so far."""
        n_new = int(new_columns.shape[1])
        total_seen = int(seen_previous) + n_new
        keep = min(int(max_columns), total_seen)
        if previous is None or seen_previous == 0:
            if n_new <= keep:
                return new_columns.copy(), total_seen
            chosen = rng.choice(n_new, keep, replace=False)
            return new_columns[:, chosen].copy(), total_seen

        old_keep = int(
            rng.hypergeometric(
                ngood=int(seen_previous),
                nbad=n_new,
                nsample=keep,
            )
        )
        new_keep = keep - old_keep
        old_keep = min(old_keep, previous.shape[1])
        new_keep = min(new_keep, n_new)
        old_idx = (
            rng.choice(previous.shape[1], old_keep, replace=False)
            if old_keep else np.empty(0, dtype=np.int64)
        )
        new_idx = (
            rng.choice(n_new, new_keep, replace=False)
            if new_keep else np.empty(0, dtype=np.int64)
        )
        pieces = []
        if old_keep:
            pieces.append(previous[:, old_idx])
        if new_keep:
            pieces.append(new_columns[:, new_idx])
        merged = np.concatenate(pieces, axis=1) if pieces else previous[:, :0]
        return merged.copy(), total_seen

    def update_from_loader(
        self,
        loader,
        *,
        stage: int,
        client_id: int,
        device: torch.device,
        max_batches: int = 20,
        activation_microbatch: int = 8,
    ) -> dict:
        """Update shared GPM bases from convolution inputs.

        Representation columns are sampled uniformly across all observed batches
        with a 512-column reservoir, matching the upstream cap without favouring
        later batches. Hooks unfold at most ``activation_microbatch`` examples at
        once, so a training-size loader batch of 256 does not materialize the full
        high-resolution patch matrix.
        """
        started = time.perf_counter()
        rng = np.random.RandomState(self.seed + 1009 * int(stage) + 97 * int(client_id))
        columns: dict[str, np.ndarray] = {}
        seen_columns: dict[str, int] = {}
        handles = []
        named_params = dict(self.model.named_parameters())

        def make_hook(module_name: str, module: nn.Conv2d):
            weight_name = self._weight_name(module_name)

            def hook(_module, inputs):
                param = named_params.get(weight_name)
                if param is None or not param.requires_grad:
                    return
                x = inputs[0].detach()
                if x.shape[0] > int(activation_microbatch):
                    selected_examples = rng.choice(
                        x.shape[0], int(activation_microbatch), replace=False
                    )
                    x = x[selected_examples]
                patches = F.unfold(
                    x,
                    kernel_size=module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                mat = patches.permute(1, 0, 2).reshape(patches.shape[1], -1)
                arr = mat.float().cpu().numpy()
                reservoir, seen = self._uniform_reservoir_merge(
                    columns.get(weight_name),
                    seen_columns.get(weight_name, 0),
                    arr,
                    self.shared.max_columns,
                    rng,
                )
                columns[weight_name] = reservoir
                seen_columns[weight_name] = seen

            return hook

        for name, module in self._conv_layers(self.model).items():
            weight_name = self._weight_name(name)
            param = named_params.get(weight_name)
            if param is not None and param.requires_grad:
                handles.append(module.register_forward_pre_hook(make_hook(name, module)))

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch_index, (x, _y) in enumerate(loader):
                    if batch_index >= int(max_batches):
                        break
                    self.model(x.to(device, non_blocking=True))
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)

        threshold = float(self.shared.threshold + 0.001 * int(stage))
        ranks = {}
        orthogonality_error = {}
        updated_layers = 0
        for weight_name, activation in columns.items():
            if activation.size == 0:
                continue
            old = self.shared.bases.get(weight_name)
            if old is None or old.size == 0:
                U, S, _ = np.linalg.svd(activation, full_matrices=False)
                energy = S ** 2
                total = float(energy.sum())
                if total <= 0.0:
                    continue
                ratio = energy / total
                rank = int(np.sum(np.cumsum(ratio) < threshold))
                if rank == 0:
                    rank = 1
                self.shared.bases[weight_name] = U[:, :rank]
                updated_layers += 1
            else:
                old = np.asarray(old)
                _u1, s1, _ = np.linalg.svd(activation, full_matrices=False)
                total = float((s1 ** 2).sum())
                if total <= 0.0:
                    continue
                residual = activation - old @ (old.T @ activation)
                U, S, _ = np.linalg.svd(residual, full_matrices=False)
                residual_energy = float((S ** 2).sum())
                ratios = (S ** 2) / total
                accumulated = (total - residual_energy) / total
                rank = 0
                for value in ratios:
                    if accumulated < threshold:
                        accumulated += float(value)
                        rank += 1
                    else:
                        break
                if rank > 0:
                    merged = np.hstack((old, U[:, :rank]))
                    q, _ = np.linalg.qr(merged)
                    self.shared.bases[weight_name] = q[:, : min(q.shape[0], q.shape[1])]
                    updated_layers += 1
            basis = self.shared.bases.get(weight_name)
            if basis is not None:
                ranks[weight_name] = int(basis.shape[1])
                gram = basis.T @ basis
                orthogonality_error[weight_name] = float(
                    np.max(np.abs(gram - np.eye(gram.shape[0])))
                )

        elapsed = float(time.perf_counter() - started)
        self.shared.update_seconds += elapsed
        record = {
            "stage": int(stage),
            "client": int(client_id),
            "threshold": threshold,
            "updated_layers": int(updated_layers),
            "ranks": ranks,
            "orthogonality_error": orthogonality_error,
            "max_orthogonality_error": float(
                max(orthogonality_error.values(), default=0.0)
            ),
            "sampled_columns": {
                name: int(value.shape[1]) for name, value in columns.items()
            },
            "seen_columns": {name: int(value) for name, value in seen_columns.items()},
            "seconds": elapsed,
        }
        self.shared.history.append(record)
        return record
