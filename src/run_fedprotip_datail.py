#!/usr/bin/env python3
"""Run a FedProTIP projection-component comparator on frozen CIFAR Data-IL.

This is an explicit protocol adaptation, not a reproduction of the full
FedProTIP system. It retains replay-free GPM gradient protection while matching
this paper's shared-head 4-client/7-domain Data-IL stream, ImageNet-pretrained
ResNet-18, persistent Adam optimizer, and 3+2 local-epoch schedule. Task-ID
prediction is omitted because every domain uses the same 100-class output head.
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


def train_epoch(model, optimizer, loader, projector, device, projection_active: bool):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    steps = 0
    samples = 0
    loss_sum = 0.0
    removed = 0.0
    total_energy = 0.0
    residual_inside = 0.0
    projected_layers = 0
    zeroed_1d = 0
    max_residual_ratio = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if projection_active:
            stats = projector.project_gradients()
            zeroed_1d += projector.zero_backbone_1d_gradients()
            removed += float(stats["gradient_energy_removed"])
            total_energy += float(stats["gradient_energy_total"])
            residual_inside += float(stats["gradient_energy_residual_inside"])
            max_residual_ratio = max(
                max_residual_ratio, float(stats["projection_residual_ratio"])
            )
            projected_layers = max(projected_layers, int(stats["projected_layers"]))
        optimizer.step()

        steps += 1
        samples += int(y.numel())
        loss_sum += float(loss.item())

    return {
        "loss": loss_sum / max(1, steps),
        "optimizer_steps": int(steps),
        "presentations": int(samples),
        "projection_active": bool(projection_active),
        "projected_layers": int(projected_layers),
        "zeroed_backbone_1d_gradients": int(zeroed_1d),
        "gradient_energy_removed": float(removed),
        "gradient_energy_total": float(total_energy),
        "gradient_energy_residual_inside": float(residual_inside),
        "max_projection_residual_ratio": float(max_residual_ratio),
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
    ap.add_argument("--gpm-activation-microbatch", type=int, default=8)
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
        "method": "FedProTIP projection component adapted to shared-head Data-IL",
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
        "optimizer_state_policy": (
            "Persistent per-client Adam state across rounds; global weights are broadcast "
            "before each local block. Optimizer state is not reset after stage 0. Frozen "
            "parameters remain in the optimizer but receive no gradients. This intentionally "
            "matches the controlled study optimizer rather than upstream FedProTIP SGD."
        ),
        "lr": 1e-4,
        "replay": False,
        "threshold": float(args.threshold),
        "gpm_max_batches": int(args.gpm_max_batches),
        "gpm_max_columns": 512,
        "gpm_activation_microbatch": int(args.gpm_activation_microbatch),
        "domain_order_name": args.domain_order,
        "domain_order": [int(x) for x in domain_order],
        "adapter_note": (
            "This is not full FedProTIP. It retains replay-free GPM gradient projection, "
            "upstream pretrained-ResNet early-block freezing, and later-task suppression "
            "of one-dimensional backbone gradients. Task-ID prediction/head masking is "
            "omitted because the controlled Data-IL protocol uses one shared 100-class head."
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
    post_stage0_steps = 0
    post_stage0_presentations = 0
    frozen_parameter_names = []

    round_id = 0
    for stage_position in range(7):
        domain_id = int(domain_order[stage_position])
        projection_active = stage_position > 0

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
                epochs = 1 if args.smoke else block_epoch_budget(args.local_epochs, 2, block_id)
                for epoch in range(epochs):
                    stats = train_epoch(
                        model,
                        optimizers[cid],
                        loader,
                        projectors[cid],
                        device,
                        projection_active=projection_active,
                    )
                    if projection_active:
                        if shared.bases and stats["projected_layers"] == 0:
                            raise RuntimeError(
                                f"projection active at stage {stage_position} but no layer was projected"
                            )
                        if stats["max_projection_residual_ratio"] > 1e-8:
                            raise RuntimeError(
                                "projected gradient retains excessive protected component: "
                                f"{stats['max_projection_residual_ratio']:.3e}"
                            )
                    total_steps += stats["optimizer_steps"]
                    total_presentations += stats["presentations"]
                    if stage_position > 0:
                        post_stage0_steps += stats["optimizer_steps"]
                        post_stage0_presentations += stats["presentations"]
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

        # Upstream freezes the pretrained ResNet stem/layer1/layer2 immediately
        # after task 0, before collecting the first protected representations.
        if stage_position == 0:
            per_client_frozen = []
            for projector in projectors:
                names = projector.freeze_early_blocks()
                per_client_frozen.append(names)
            if not per_client_frozen or any(names != per_client_frozen[0] for names in per_client_frozen):
                raise RuntimeError("inconsistent FedProTIP frozen parameter sets across clients")
            frozen_parameter_names = per_client_frozen[0]
            if not frozen_parameter_names:
                raise RuntimeError("FedProTIP early-block freezing selected no parameters")
            print(
                f"[FedProTIP] froze {len(frozen_parameter_names)} early-backbone parameters "
                "after stage 0; persistent Adam state retained",
                flush=True,
            )

        # Update the protected representation basis after each completed stage.
        # Frozen layers are excluded automatically because their weights no longer
        # require gradients. A shared basis is updated sequentially across clients,
        # matching the upstream global orthogonal-set semantics.
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
                activation_microbatch=args.gpm_activation_microbatch,
            )
            if record["max_orthogonality_error"] > 1e-5:
                raise RuntimeError(
                    "GPM basis failed orthonormality check: "
                    f"{record['max_orthogonality_error']:.3e}"
                )
            gpm_rows.append({
                "stage": record["stage"],
                "client": record["client"],
                "threshold": record["threshold"],
                "updated_layers": record["updated_layers"],
                "seconds": record["seconds"],
                "max_orthogonality_error": record["max_orthogonality_error"],
                "ranks_json": json.dumps(record["ranks"], sort_keys=True),
                "sampled_columns_json": json.dumps(record["sampled_columns"], sort_keys=True),
                "seen_columns_json": json.dumps(record["seen_columns"], sort_keys=True),
            })

        if stage_position == 0:
            nonzero_bases = {
                name: basis for name, basis in shared.bases.items() if basis.size > 0
            }
            if not nonzero_bases:
                raise RuntimeError("stage-0 GPM construction produced no protected bases")
            print(
                f"[FedProTIP] stage-0 basis ready for {len(nonzero_bases)} trainable conv layers",
                flush=True,
            )

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
        "optimizer_steps_total": int(total_steps),
        "example_presentations_total": int(total_presentations),
        "optimizer_steps_post_stage0": int(post_stage0_steps),
        "example_presentations_post_stage0": int(post_stage0_presentations),
        "gpm_update_seconds": float(shared.update_seconds),
        "gpm_basis_ranks": {key: int(value.shape[1]) for key, value in shared.bases.items()},
        "frozen_parameter_names": frozen_parameter_names,
        "max_gpm_orthogonality_error": float(
            max(
                (row["max_orthogonality_error"] for row in gpm_rows),
                default=0.0,
            )
        ),
        "max_projection_residual_ratio": float(
            max(
                (row["max_projection_residual_ratio"] for row in local_rows),
                default=0.0,
            )
        ),
    }
    (output_dir / "fedprotip_summary.json").write_text(json.dumps(summary, indent=2))
    print("[FedProTIP Data-IL] summary:", summary, flush=True)


if __name__ == "__main__":
    main()
