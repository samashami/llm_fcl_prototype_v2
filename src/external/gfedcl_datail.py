"""Protocol adapter for upstream GFedCL on the frozen CIFAR Data-IL stream."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.attribution_protocol import resolve_domain_order
from src.data_il_streams import (
    CONTROLLED_DOMAIN_TRANSFORMS,
    make_controlled_domain_shift_batches,
    stage_class_counts,
)

GFEDCL_MEAN = (0.5071, 0.4867, 0.4408)
GFEDCL_STD = (0.2675, 0.2565, 0.2761)


def _noise_seed(experiment_seed: int, domain: int, dataset_index: int) -> int:
    value = f"controlled_domain_shift\0{experiment_seed}\0{domain}\0{dataset_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


class GFedCLDomainDataset(Dataset):
    """CIFAR subset with the frozen domain perturbation at GFedCL's native 32x32."""

    def __init__(self, base_dataset, indices, domain, experiment_seed, training=False):
        self.base_dataset = base_dataset
        self.indices = [int(x) for x in indices]
        self.domain = int(domain)
        self.experiment_seed = int(experiment_seed)
        self.training = bool(training)
        if not 0 <= self.domain < len(CONTROLLED_DOMAIN_TRANSFORMS):
            raise ValueError(f"invalid domain {self.domain}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        dataset_index = self.indices[int(item)]
        image = Image.fromarray(self.base_dataset.data[dataset_index])
        label = int(self.base_dataset.targets[dataset_index])
        if self.training:
            image = transforms.RandomHorizontalFlip()(image)
        spec = CONTROLLED_DOMAIN_TRANSFORMS[self.domain]
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
            generator.manual_seed(_noise_seed(self.experiment_seed, self.domain, dataset_index))
            tensor = (tensor + torch.randn(tensor.shape, generator=generator) * spec["std"]).clamp_(0.0, 1.0)
        tensor = transforms.functional.normalize(tensor, GFEDCL_MEAN, GFEDCL_STD)
        return tensor, label


def build_frozen_partition(
    *, data_dir, seed, num_clients=4, val_size=5000,
    subset_per_client=-1, domain_order_name="heldout"
):
    """Reproduce the attribution split, client partition and seven stage allocation."""
    if int(num_clients) != 4:
        raise ValueError("paper GFedCL adapter requires exactly four clients")
    train_full = datasets.CIFAR100(data_dir, train=True, download=True, transform=None)
    test_full = datasets.CIFAR100(data_dir, train=False, download=True, transform=None)

    generator = torch.Generator().manual_seed(int(seed))
    train_size = len(train_full) - int(val_size)
    train_subset, val_subset = torch.utils.data.random_split(
        train_full, [train_size, int(val_size)], generator=generator
    )
    train_indices = np.asarray(train_subset.indices, dtype=np.int64)
    val_indices = np.asarray(val_subset.indices, dtype=np.int64)

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(train_indices)
    sizes = [len(perm) // int(num_clients)] * int(num_clients)
    for i in range(len(perm) % int(num_clients)):
        sizes[i] += 1
    splits, start = [], 0
    for size in sizes:
        splits.append(perm[start:start + size].tolist())
        start += size
    if int(subset_per_client) > 0:
        splits = [split[: int(subset_per_client)] for split in splits]

    targets = np.asarray(train_full.targets)
    schedules = [
        make_controlled_domain_shift_batches(
            split, targets, num_batches=7, seed=int(seed) + cid
        )
        for cid, split in enumerate(splits)
    ]
    domain_order = tuple(resolve_domain_order(domain_order_name, 7))
    return {
        "train_full": train_full,
        "test_full": test_full,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "client_splits": splits,
        "schedules": schedules,
        "domain_order": domain_order,
        "class_counts": [stage_class_counts(schedule, targets) for schedule in schedules],
    }


def make_upstream_loader_factory(
    *, val_size, subset_per_client, domain_order_name, metadata_path=None
):
    """Return setup_cifar100_loaders(opt) with the upstream GFedCL interface."""
    def setup_cifar100_loaders(opt):
        if int(opt.num_clients) != 4 or int(opt.num_task) != 7:
            raise ValueError("GFedCL Data-IL adapter expects four clients and seven stages")
        state = build_frozen_partition(
            data_dir=opt.data_dir,
            seed=int(opt.seed),
            num_clients=int(opt.num_clients),
            val_size=int(val_size),
            subset_per_client=int(subset_per_client),
            domain_order_name=domain_order_name,
        )
        loaders = defaultdict(dict)
        test_indices = list(range(len(state["test_full"])))
        workers = int(getattr(opt, "num_workers", 0))
        pin = bool(getattr(opt, "pin_memory", False))
        for cid in range(int(opt.num_clients)):
            for stage in range(7):
                domain = int(state["domain_order"][stage])
                train_ds = GFedCLDomainDataset(
                    state["train_full"], state["schedules"][cid][stage],
                    domain, int(opt.seed), training=True,
                )
                test_ds = GFedCLDomainDataset(
                    state["test_full"], test_indices,
                    domain, int(opt.seed), training=False,
                )
                loaders[cid][stage] = {
                    "train": DataLoader(
                        train_ds, batch_size=int(opt.batch_size), shuffle=True,
                        num_workers=workers, pin_memory=pin,
                    ),
                    "test": DataLoader(
                        test_ds, batch_size=int(opt.batch_size), shuffle=False,
                        num_workers=workers, pin_memory=pin,
                    ),
                }
        if metadata_path:
            path = Path(metadata_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "adapter": "GFedCL shared-label Data-IL",
                "seed": int(opt.seed),
                "num_clients": 4,
                "num_stages": 7,
                "val_size": int(val_size),
                "subset_per_client": int(subset_per_client),
                "domain_order_name": domain_order_name,
                "domain_order": [int(x) for x in state["domain_order"]],
                "ordered_transforms": [
                    CONTROLLED_DOMAIN_TRANSFORMS[int(x)]["name"]
                    for x in state["domain_order"]
                ],
                "stage_sizes_by_client": [
                    [len(stage) for stage in schedule]
                    for schedule in state["schedules"]
                ],
                "per_stage_class_counts_by_client": state["class_counts"],
                "architecture_note": (
                    "Upstream GFedCL retains its native 32x32 encoder/generator; "
                    "the data split, client split, domain order and stage allocation are matched."
                ),
            }, indent=2))
        return loaders
    return setup_cifar100_loaders
