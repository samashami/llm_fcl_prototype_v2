#!/usr/bin/env python3
"""Run a FedProTIP-style external comparator on the frozen CIFAR Data-IL stream.

This is a protocol adaptation, not an exact reproduction of the authors' CIFAR
Class-IL setup.  It preserves the FedProTIP/GPM gradient-projection mechanism
and replay-free training, while matching this paper's 4-client/7-domain Data-IL
stream, ImageNet-pretrained ResNet-18, Adam optimizer, and 3+2 local-epoch
schedule.  The comparator is evaluated without task labels at inference.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets

from src.attribution_protocol import block_epoch_budget, resolve_domain_order
from src.data_il_streams import (
    CONTROLLED_DOMAIN_TRANSFORMS,
    StageDomainDataset,
    make_controlled_domain_shift_batches,
    stage_class_counts,
)
from src.external.fedprotip_projection import FedProTIPProjector, SharedGPMState
from src.model import build_resnet18

UPSTREAM_REPO = "https://github.com/seohyeon-cha/FedProTIP"
UPSTREAM_COMMIT = "54193fa2d44f6203f39299a0ac3845097559a440"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def average_models(models, device):
    avg = copy.deepcopy(models[0]).to(device)
    with torch.no_grad():
        state = avg.state_dict()
        for key in state:
            values = [m.state_dict()[key].to(device) for m in models]
            if not torch.is_floating_point(values[0]):
                state[key].copy_(values[0])
            else:
                state[key].copy_(sum(values) / len(values))
    return avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.numel())
    return 100.0 * correct / max(1, total)


def build_stream(seed, data_dir, val_size, subset_per_client, domain_order_name):
    train_full = datasets.CIFAR100(data_dir, train=True, download=True, transform=None)
    test_full = datasets.CIFAR100(data_dir, train=False, download=True, transform=None)

    generator = torch.Generator().manual_seed(int(seed))
    train_subset, val_subset = torch.utils.data.random_split(
        train_full,
        [len(train_full) - int(val_size), int(val_size)],
        generator=generator,
    )
    train_indices = np.asarray(train_subset.indices, dtype=np.int64)
    val_indices = np.asarray(val_subset.indices, dtype=np.int64)

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(train_indices)
    sizes = [len(perm) // 4] * 4
    for i in range(len(perm) % 4):
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
            split,
            targets,
            num_batches=7,
            seed=int(seed) + cid,
        )
        for cid, split in enumerate(splits)
    ]
    domain_order = tuple(resolve_domain_order(domain_order_name, 7))
    return train_full, test_full, val_indices, splits, schedules, domain_order


def train_epoch(model, optimizer, loader, projector, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    steps = 0
    samples = 0
    loss_sum = 0.0
    removed = 0.0
    total_energy = 0.0
    projected_layers = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        stats = projector.project_gradients()
        optimizer.step()
        steps += 1
        samples += int(y.numel())
        loss_sum += float(loss.item())
        removed += float(stats["gradient_energy_removed"])
        total_energy += float(stats["gradient_energy_total"])
        projected_layers = max(projected_layers, int(stats["projected_layers"]))
    return {
        "loss": loss_sum / max(1, steps),
        "optimizer_steps": int(steps),
        "presentations": int(samples),
        "projected_layers": int(projected_layers),
        "gradient_energy_removed": float(removed),
        "gradient_energy_total": float(total_energy),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--domain-order", choices=["development", "heldout"], default="heldout")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--gpm-max-batches", type=int, default=20)
    args = ap.parse_args()
    if args.smoke:
        args.val_size = 700
        args.subset_per_client = 140
        args.batch_size = 32
        args.local_epochs = 2
    else:
        args.val_size = 5000
        args.subset_per_client = -1
        args.batch_size = 256
        args.local_epochs = 5
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_full, test_full, val_indices, splits, schedules, domain_order = build_stream(
        args.seed,
        args.data_dir,
        args.val_size,
        args.subset_per_client,
        args.domain_order,
    )
    test_indices = list(range(len(test_full)))

    manifest = {
        "method": "FedProTIP-style GPM Data-IL adaptation",
        "upstream_repository": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "clients": 4,
        "stages": 7,
        "rounds": 14,
        "blocks_per_stage": 2,
        "epochs_per_stage": int(args.local_epochs),
        "block_epoch_budget": [1, 1] if args.smoke else [3, 2],
        "batch_size": int(args.batch_size),
        "optimizer": "Adam",
        "lr": 1e-4,
        "replay": False,
        "threshold": float(args.threshold),
        "gpm_max_batches": int(args.gpm_max_batches),
        "domain_order_name": args.domain_order,
        "domain_order": [int(x) for x in domain_order],
        "adapter_note": (
            "FedProTIP's replay-free GPM gradient projection is retained. "
            "Task-ID prediction/Class-IL head masking is omitted because the controlled Data-IL "
            "protocol uses a shared 100-class label space in every domain."
        ),
        "stage_sizes_by_client": [[len(stage) for stage in schedule] for schedule in schedules],
        "class_counts_by_client": [
            stage_class_counts(schedule, train_full.targets) for schedule in schedules
        ],
    }
    (output_dir / "fedprotip_datail_manifest.json").write_text(json.dumps(manifest, indent=2))

    clients = [build_resnet18(100).to(device) for _ in range(4)]
    optimizers = [optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0) for model in clients]
    global_model = average_models(clients, device)
    shared = SharedGPMState(threshold=float(args.threshold), max_columns=512)
    projectors = [
        FedProTIPProjector(model, shared, seed=args.seed + cid)
        for cid, model in enumerate(clients)
    ]

    round_rows = []
    local_rows = []
    gpm_rows = []
    total_steps = 0
    total_presentations = 0

    round_id = 0
    for stage_position in range(7):
        domain_id = int(domain_order[stage_position])
        for block_id in range(2):
            for cid, model in enumerate(clients):
                model.load_state_dict(global_model.state_dict())
                stage_ds = StageDomainDataset(
                    train_full,
                    schedules[cid][stage_position],
                    stage=domain_id,
                    experiment_seed=args.seed,
                    training=True,
                )
                loader = DataLoader(
                    stage_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"),
                )
                epochs = (
                    1
                    if args.smoke
                    else block_epoch_budget(args.local_epochs, 2, block_id)
                )
                for epoch in range(epochs):
                    stats = train_epoch(
                        model,
                        optimizers[cid],
                        loader,
                        projectors[cid],
                        device,
                    )
                    total_steps += stats["optimizer_steps"]
                    total_presentations += stats["presentations"]
                    local_rows.append({
                        "round": round_id,
                        "stage": stage_position,
                        "block": block_id,
                        "client": cid,
                        "epoch": epoch + 1,
                        **stats,
                    })

            global_model = average_models(clients, device)
            seen_accuracies = []
            domain_accuracies = {}
            for seen_position in range(stage_position + 1):
                seen_domain = int(domain_order[seen_position])
                eval_ds = StageDomainDataset(
                    test_full,
                    test_indices,
                    stage=seen_domain,
                    experiment_seed=args.seed,
                    training=False,
                )
                eval_loader = DataLoader(
                    eval_ds,
                    batch_size=256,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"),
                )
                acc = evaluate(global_model, eval_loader, device)
                seen_accuracies.append(acc)
                domain_accuracies[str(seen_domain)] = acc
            round_rows.append({
                "round": round_id,
                "stage": stage_position,
                "block": block_id,
                "domain_id": domain_id,
                "seen_domain_accuracy": float(np.mean(seen_accuracies)),
                "current_domain_accuracy": float(domain_accuracies[str(domain_id)]),
                "domain_accuracies_json": json.dumps(domain_accuracies, sort_keys=True),
                "optimizer_steps_cumulative": int(total_steps),
                "presentations_cumulative": int(total_presentations),
            })
            print(
                f"[FedProTIP r={round_id} stage={stage_position} block={block_id}] "
                f"seen={round_rows[-1]['seen_domain_accuracy']:.3f}% "
                f"current={round_rows[-1]['current_domain_accuracy']:.3f}% "
                f"steps={total_steps} presentations={total_presentations}",
                flush=True,
            )
            round_id += 1

        # FedProTIP/GPM updates its protected representation subspace after
        # completing a continual stage.  Use each client's stage data and the
        # common global model as the starting point for consistent bases.
        for cid, model in enumerate(clients):
            model.load_state_dict(global_model.state_dict())
            projectors[cid].model = model
            stage_ds = StageDomainDataset(
                train_full,
                schedules[cid][stage_position],
                stage=domain_id,
                experiment_seed=args.seed,
                training=False,
            )
            gpm_loader = DataLoader(
                stage_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            record = projectors[cid].update_from_loader(
                gpm_loader,
                stage=stage_position,
                client_id=cid,
                device=device,
                max_batches=args.gpm_max_batches,
            )
            gpm_rows.append({
                "stage": record["stage"],
                "client": record["client"],
                "threshold": record["threshold"],
                "updated_layers": record["updated_layers"],
                "seconds": record["seconds"],
                "ranks_json": json.dumps(record["ranks"], sort_keys=True),
            })
        if stage_position == 0:
            # The official pretrained-ResNet FedProTIP freezes early blocks
            # after the first task. Preserve that behavior for later stages.
            for model in clients:
                projector = FedProTIPProjector(model, shared, seed=args.seed)
                projector.freeze_early_blocks()
            # Rebuild optimizers so frozen parameters are excluded cleanly.
            optimizers = [
                optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=0.0)
                for model in clients
            ]

    for filename, rows in (
        ("fedprotip_round_metrics.csv", round_rows),
        ("fedprotip_local_training.csv", local_rows),
        ("fedprotip_gpm_updates.csv", gpm_rows),
    ):
        with (output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    post_stage0 = [row["seen_domain_accuracy"] for row in round_rows if row["stage"] > 0]
    final = round_rows[-1]
    summary = {
        "round_averaged_seen_domain_accuracy_post_stage0": float(np.mean(post_stage0)),
        "final_seen_domain_accuracy": float(final["seen_domain_accuracy"]),
        "final_current_domain_accuracy": float(final["current_domain_accuracy"]),
        "optimizer_steps": int(total_steps),
        "example_presentations": int(total_presentations),
        "gpm_update_seconds": float(shared.update_seconds),
        "gpm_basis_ranks": {key: int(value.shape[1]) for key, value in shared.bases.items()},
    }
    (output_dir / "fedprotip_summary.json").write_text(json.dumps(summary, indent=2))
    print("[FedProTIP Data-IL] summary:", summary, flush=True)


if __name__ == "__main__":
    main()
