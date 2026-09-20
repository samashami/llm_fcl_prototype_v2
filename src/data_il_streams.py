"""Deterministic Data-IL stream construction for CIFAR-100 experiments."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CONTROLLED_DOMAIN_TRANSFORMS = (
    {"name": "clean"},
    {"name": "brightness", "brightness": 1.25},
    {"name": "contrast", "contrast": 1.25},
    {"name": "gaussian_blur", "kernel_size": 3, "sigma": 0.5},
    {"name": "gaussian_noise", "std": 0.03},
    {"name": "saturation", "saturation": 1.30},
    {
        "name": "brightness_blur",
        "brightness": 1.20,
        "kernel_size": 3,
        "sigma": 0.5,
    },
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cl_batch_sizes(total: int, num_batches: int = 7) -> list[int]:
    """Return the existing 46.6% + evenly split remainder size convention."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    if total < num_batches:
        raise ValueError("each Data-IL stage must contain at least one sample")
    if num_batches == 1:
        return [total]
    initial = int(round(0.466 * total))
    initial = max(1, min(total - (num_batches - 1), initial))
    remainder = total - initial
    per = remainder // (num_batches - 1)
    return [initial] + [per] * (num_batches - 2) + [remainder - per * (num_batches - 2)]


def make_random_chunks(indices: Sequence[int], num_batches: int = 7, seed: int = 42) -> list[list[int]]:
    """The pre-existing random chunk protocol, factored without changing its logic."""
    rng = np.random.RandomState(seed)
    values = np.array(indices, dtype=np.int64)
    rng.shuffle(values)
    sizes = cl_batch_sizes(len(values), num_batches)
    batches, start = [], 0
    for size in sizes:
        batches.append(values[start:start + size].tolist())
        start += size
    return batches


def make_controlled_domain_shift_batches(
    indices: Sequence[int], targets: Sequence[int], num_batches: int = 7, seed: int = 42
) -> list[list[int]]:
    """Create exact-size, class-stratified, non-overlapping Data-IL stages."""
    if num_batches != len(CONTROLLED_DOMAIN_TRANSFORMS):
        raise ValueError("controlled_domain_shift requires exactly seven stages")
    values = np.asarray(indices, dtype=np.int64)
    if len(values) < num_batches:
        raise ValueError("each client needs at least seven samples")
    labels = np.asarray(targets)
    rng = np.random.RandomState(seed)
    target_sizes = cl_batch_sizes(len(values), num_batches)
    capacities = list(target_sizes)
    batches = [[] for _ in range(num_batches)]
    class_queues: Dict[int, list[int]] = {}
    for class_id in sorted(np.unique(labels[values]).tolist()):
        queue = values[labels[values] == class_id].copy()
        rng.shuffle(queue)
        class_queues[int(class_id)] = queue.tolist()

    # Give every sufficiently represented class one sample per domain before
    # filling the remaining exact stage capacities.
    class_order = sorted(class_queues)
    rng.shuffle(class_order)
    for class_id in class_order:
        queue = class_queues[class_id]
        for stage in rng.permutation(num_batches)[: min(num_batches, len(queue))]:
            batches[int(stage)].append(queue.pop())
            capacities[int(stage)] -= 1
            if capacities[int(stage)] < 0:
                raise RuntimeError("controlled stream initial class allocation overflowed a stage")

    # Fill residual capacities proportionally and deterministically. Relative
    # remaining capacity makes every class see the same target proportions,
    # while still producing exact requested stage sizes globally.
    for class_id in class_order:
        remaining = class_queues[class_id]
        rng.shuffle(remaining)
        for item in remaining:
            ratios = [capacity / target for capacity, target in zip(capacities, target_sizes)]
            largest = max(ratios)
            candidates = [stage for stage, value in enumerate(ratios) if value == largest]
            stage = int(candidates[rng.randint(len(candidates))])
            batches[stage].append(int(item))
            capacities[stage] -= 1
            if capacities[stage] < 0:
                raise RuntimeError("controlled stream stage capacity underflow")
    if any(capacities):
        raise RuntimeError(f"controlled stream did not fill stage capacities: {capacities}")
    return batches


def stage_class_counts(batches: Iterable[Sequence[int]], targets: Sequence[int]) -> list[dict[str, int]]:
    labels = np.asarray(targets)
    return [
        {str(key): int(value) for key, value in sorted(Counter(labels[list(batch)]).items())}
        for batch in batches
    ]


def _noise_seed(experiment_seed: int, stage: int, dataset_index: int) -> int:
    value = f"controlled_domain_shift\0{experiment_seed}\0{stage}\0{dataset_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class StageDomainDataset(Dataset):
    """Index subset that applies one deterministic visual domain to CIFAR images."""

    base_dataset: object
    indices: Sequence[int]
    stage: int
    experiment_seed: int
    training: bool = False

    def __post_init__(self):
        if not 0 <= self.stage < len(CONTROLLED_DOMAIN_TRANSFORMS):
            raise ValueError(f"invalid controlled-domain stage {self.stage}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int):
        dataset_index = int(self.indices[item])
        image = Image.fromarray(self.base_dataset.data[dataset_index])
        label = int(self.base_dataset.targets[dataset_index])
        image = transforms.Resize(224)(image)
        if self.training:
            image = transforms.RandomHorizontalFlip()(image)
        spec = CONTROLLED_DOMAIN_TRANSFORMS[self.stage]
        if "brightness" in spec:
            image = transforms.functional.adjust_brightness(image, spec["brightness"])
        if "contrast" in spec:
            image = transforms.functional.adjust_contrast(image, spec["contrast"])
        if "saturation" in spec:
            image = transforms.functional.adjust_saturation(image, spec["saturation"])
        if "kernel_size" in spec:
            image = transforms.GaussianBlur(spec["kernel_size"], sigma=spec["sigma"])(image)
        tensor = transforms.functional.to_tensor(image)
        if "std" in spec:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_noise_seed(self.experiment_seed, self.stage, dataset_index))
            tensor = (tensor + torch.randn(tensor.shape, generator=generator) * spec["std"]).clamp_(0.0, 1.0)
        return transforms.functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD), label
