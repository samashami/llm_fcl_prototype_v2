"""BI-style class-balanced memory for the shared-head Data-IL adaptation.

This adapts Online-FCL's Bregman-Information bottom-k memory update. Incoming
samples are admitted once per domain; when capacity is exceeded, BI is computed
over the existing memory plus that domain's incoming samples and the lowest-BI
items are retained under per-label quotas.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from typing import Optional

import numpy as np
import torch
from torchvision.transforms import v2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BIMemory:
    """Fixed-capacity memory with BI bottom-k selection and stage metadata."""

    def __init__(self, capacity: int = 2000, seed: Optional[int] = None,
                 score_batch_size: int = 128):
        if capacity <= 0 or score_batch_size <= 0:
            raise ValueError("BI memory capacity and score batch size must be positive")
        self.capacity = int(capacity)
        self.score_batch_size = int(score_batch_size)
        self.data = []  # (normalized image tensor on CPU, label tensor, domain id)
        self._rng = random.Random(None if seed is None else int(seed))
        self._admitted_stages = set()
        self.samples_offered = 0
        self.last_score_seconds = 0.0
        self.last_scored_candidates = 0
        self.last_kept = 0
        self.total_score_seconds = 0.0

    def __len__(self):
        return len(self.data)

    @property
    def stage_ids(self):
        return tuple(sorted({int(row[2]) for row in self.data}))

    @property
    def labels(self):
        return [int(row[1]) for row in self.data]

    def sample_count_excluding_stage(self, count: int, device, exclude_stage: int):
        candidates = [row for row in self.data if int(row[2]) != int(exclude_stage)]
        k = min(max(0, int(count)), len(candidates))
        if k == 0:
            return None, None
        selected = self._rng.sample(candidates, k)
        xs = torch.stack([row[0] for row in selected]).to(device)
        ys = torch.stack([row[1] for row in selected]).to(device)
        return xs, ys

    def sample_count(self, count: int, device):
        k = min(max(0, int(count)), len(self.data))
        if k == 0:
            return None, None
        selected = self._rng.sample(self.data, k)
        return (torch.stack([row[0] for row in selected]).to(device),
                torch.stack([row[1] for row in selected]).to(device))

    @staticmethod
    def _cutout(images: torch.Tensor, length: int) -> torch.Tensor:
        out = images.clone()
        _, _, height, width = out.shape
        for i in range(out.shape[0]):
            cy, cx = np.random.randint(height), np.random.randint(width)
            y0, y1 = max(0, cy - length // 2), min(height, cy + length // 2)
            x0, x1 = max(0, cx - length // 2), min(width, cx + length // 2)
            out[i, :, y0:y1, x0:x1] = 0.0
        return out

    @classmethod
    def _tta_transforms(cls, images: torch.Tensor):
        """Return the paper's 12 image perturbations on [0,1] RGB tensors."""
        _, _, height, width = images.shape
        crop = v2.RandomResizedCrop(
            size=(height, width), scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True
        )
        return [
            cls._cutout(images, 10),
            cls._cutout(images, 20),
            v2.RandomHorizontalFlip(p=1.0)(images),
            v2.RandomVerticalFlip(p=1.0)(images),
            v2.RandomRotation(degrees=10)(images),
            v2.RandomRotation(degrees=45)(images),
            v2.RandomRotation(degrees=90)(images),
            v2.ColorJitter(brightness=0.1)(images),
            v2.RandomPerspective()(images),
            v2.RandomAffine(degrees=20, translate=(0.1, 0.3), scale=(0.5, 0.75))(images),
            crop(images),
            v2.RandomInvert(p=1.0)(images),
        ]

    @torch.no_grad()
    def _score_candidates(self, model, images: torch.Tensor) -> torch.Tensor:
        """Estimate sample BI = E[LSE(z)] - LSE(E[z]) using 12-view TTA."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("BI image candidates must have shape [N,3,H,W]")
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        score_parts = []
        np_state = np.random.get_state()
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        start = time.perf_counter()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(1729 + len(self._admitted_stages))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(1729 + len(self._admitted_stages))
                np.random.seed(1729 + len(self._admitted_stages))
                for offset in range(0, images.shape[0], self.score_batch_size):
                    normalized = images[offset:offset + self.score_batch_size].to(device)
                    raw = (normalized * std.to(device) + mean.to(device)).clamp(0.0, 1.0)
                    logits_views = [model((view - mean.to(device)) / std.to(device))
                                    for view in self._tta_transforms(raw)]
                    logits = torch.stack(logits_views, dim=0).float()
                    scores = logits.logsumexp(dim=-1).mean(dim=0) - logits.mean(dim=0).logsumexp(dim=-1)
                    score_parts.append(scores.cpu())
        finally:
            np.random.set_state(np_state)
            if was_training:
                model.train()
        self.last_score_seconds = time.perf_counter() - start
        self.total_score_seconds += self.last_score_seconds
        self.last_scored_candidates = int(images.shape[0])
        return torch.cat(score_parts) if score_parts else torch.empty(0)

    def _bottomk_class_balanced(self, candidates, scores):
        labels = [int(row[1]) for row in candidates]
        counts = Counter(labels)
        classes = sorted(counts)
        base, remainder = divmod(self.capacity, len(classes))
        quotas = {label: base + int(i < remainder) for i, label in enumerate(classes)}
        selected_by_class = {}
        selected_count = 0
        for label in classes:
            indices = [i for i, value in enumerate(labels) if value == label]
            indices.sort(key=lambda i: (float(scores[i]), i))  # bottom-k: lowest BI
            take = min(quotas[label], len(indices))
            selected_by_class[label] = indices[:take]
            selected_count += take
        # Reallocate slots when a class has fewer candidates than its quota.
        target = min(self.capacity, len(candidates))
        if selected_count < target:
            already_selected = {
                index for indices in selected_by_class.values() for index in indices
            }
            remaining = [i for i in range(len(candidates)) if i not in already_selected]
            remaining.sort(key=lambda i: (float(scores[i]), i))
            for index in remaining[:target - selected_count]:
                selected_by_class[labels[index]].append(index)
        chosen_indices = [i for label in classes for i in selected_by_class[label]]
        return [candidates[i] for i in chosen_indices]

    def add_domain_batch(self, x: torch.Tensor, y: torch.Tensor, stage_id: int, model):
        """Offer one complete incoming domain batch and update memory once."""
        stage_id = int(stage_id)
        if stage_id in self._admitted_stages:
            raise ValueError(f"BI domain {stage_id} was already admitted")
        if x.size(0) != y.size(0):
            raise ValueError("BI memory features and labels must have equal lengths")
        incoming = [(x[i].detach().cpu().clone(), y[i].detach().cpu().long().clone(), stage_id)
                    for i in range(len(y))]
        self.samples_offered += len(incoming)
        candidates = self.data + incoming
        self._admitted_stages.add(stage_id)
        if len(candidates) <= self.capacity:
            self.data = candidates
            self.last_score_seconds = 0.0
            self.last_scored_candidates = 0
        else:
            candidate_x = torch.stack([row[0] for row in candidates])
            scores = self._score_candidates(model, candidate_x)
            self.data = self._bottomk_class_balanced(candidates, scores)
        self.last_kept = len(self.data)
        if len(self.data) > self.capacity:
            raise RuntimeError("BI memory exceeded its configured capacity")
        return {
            "stage_id": stage_id,
            "incoming": len(incoming),
            "candidates": len(candidates),
            "scored_candidates": self.last_scored_candidates,
            "kept": len(self.data),
            "score_seconds": self.last_score_seconds,
        }

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "score_batch_size": self.score_batch_size,
            "data": self.data,
            "admitted_stages": sorted(self._admitted_stages),
            "rng_state": self._rng.getstate(),
            "samples_offered": self.samples_offered,
            "total_score_seconds": self.total_score_seconds,
        }

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.capacity:
            raise ValueError("BI memory checkpoint capacity differs")
        self.data = [(x.detach().cpu(), y.detach().cpu(), int(stage))
                     for x, y, stage in state["data"]]
        self._admitted_stages = set(map(int, state.get("admitted_stages", [])))
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])
        self.samples_offered = int(state.get("samples_offered", 0))
        self.total_score_seconds = float(state.get("total_score_seconds", 0.0))
