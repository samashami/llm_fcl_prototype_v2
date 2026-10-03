"""FedProTIP-style shared gradient projection for controlled Data-IL.

This adapter keeps the core FedProTIP/GPM mechanism: after each completed
continual stage, convolutional representation subspaces are accumulated and
subsequent local gradients are projected onto their orthogonal complement.
Task-ID prediction is omitted because our Data-IL protocol has one shared
100-class output space at every stage.
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

    def project_gradients(self) -> dict:
        """Apply G <- G - G U U^T to convolutional weight gradients."""
        named_params = dict(self.model.named_parameters())
        total_sq = 0.0
        removed_sq = 0.0
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
            projected_layers += 1
        return {
            "projected_layers": int(projected_layers),
            "gradient_energy_total": float(total_sq),
            "gradient_energy_removed": float(removed_sq),
        }

    def update_from_loader(
        self,
        loader,
        *,
        stage: int,
        client_id: int,
        device: torch.device,
        max_batches: int = 20,
    ) -> dict:
        """Update shared GPM bases from convolution inputs.

        FedProTIP caps sampled representation columns at 512 before SVD.  The
        same cap is used here to keep memory bounded while retaining the method's
        subspace-energy criterion.
        """
        started = time.perf_counter()
        rng = np.random.RandomState(self.seed + 1009 * int(stage) + 97 * int(client_id))
        columns: dict[str, np.ndarray] = {}
        handles = []

        def make_hook(module_name: str, module: nn.Conv2d):
            weight_name = self._weight_name(module_name)

            def hook(_module, inputs):
                x = inputs[0].detach()
                patches = F.unfold(
                    x,
                    kernel_size=module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                mat = patches.permute(1, 0, 2).reshape(patches.shape[1], -1)
                if mat.shape[1] > self.shared.max_columns:
                    chosen = rng.choice(mat.shape[1], self.shared.max_columns, replace=False)
                    mat = mat[:, chosen]
                arr = mat.float().cpu().numpy()
                previous = columns.get(weight_name)
                merged = arr if previous is None else np.concatenate([previous, arr], axis=1)
                if merged.shape[1] > self.shared.max_columns:
                    chosen = rng.choice(merged.shape[1], self.shared.max_columns, replace=False)
                    merged = merged[:, chosen]
                columns[weight_name] = merged

            return hook

        for name, module in self._conv_layers(self.model).items():
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
            ranks[weight_name] = 0 if basis is None else int(basis.shape[1])

        elapsed = float(time.perf_counter() - started)
        self.shared.update_seconds += elapsed
        record = {
            "stage": int(stage),
            "client": int(client_id),
            "threshold": threshold,
            "updated_layers": int(updated_layers),
            "ranks": ranks,
            "seconds": elapsed,
        }
        self.shared.history.append(record)
        return record
