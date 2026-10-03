#!/usr/bin/env python3
"""Run the official GFedCL CIFAR implementation on our frozen Data-IL stream.

The script pins the upstream GFedCL source, injects our dataset adapter, and
keeps GFedCL's graph/discriminator/synthetic-replay mechanism intact.  The full
paper run uses seven held-out domains and two rounds per stage with a 3+2 local
epoch schedule for current-stage training.  GFedCL's synthetic previous-stage
pass remains additional method-specific compute and must be reported as such.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import types

import numpy as np
import torch

from src.external.gfedcl_datail import make_upstream_loader_factory

UPSTREAM_URL = "https://github.com/IntelliSys-Lab/GFedCL.git"
UPSTREAM_COMMIT = "daa9faa4af4e80f6bba79bc020c10a0842c3cfb7"


def _run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def ensure_upstream(root: Path, refresh: bool) -> Path:
    if refresh and root.exists():
        shutil.rmtree(root)
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", UPSTREAM_URL, root])
    _run(["git", "fetch", "origin", UPSTREAM_COMMIT], cwd=root)
    _run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=root)
    return root


def patch_upstream(gfedcl_py: Path, classifier_epochs: int) -> None:
    text = gfedcl_py.read_text()
    old_classifier = "self.dataloaders[i + j][task]['train'],\n                        20\n                    )"
    new_classifier = (
        "self.dataloaders[i + j][task]['train'],\n"
        f"                        {int(classifier_epochs)}\n"
        "                    )"
    )
    if old_classifier not in text:
        raise RuntimeError("upstream classifier-training call changed; refusing silent patch")
    text = text.replace(old_classifier, new_classifier, 1)

    old_epochs = "self.opt.num_local_epochs,"
    replacement = (
        "(self.opt.local_epoch_schedule[r] if hasattr(self.opt, "
        "'local_epoch_schedule') else self.opt.num_local_epochs),"
    )
    count = text.count(old_epochs)
    if count < 2:
        raise RuntimeError(f"expected >=2 local-epoch call sites, found {count}")
    text = text.replace(old_epochs, replacement)
    gfedcl_py.write_text(text)


def configure_opt(upstream_root: Path, args):
    cifar_dir = upstream_root / "CIFAR100"
    sys.path.insert(0, str(cifar_dir))
    from configs.CIFAR100 import parse_args

    opt = parse_args([])
    opt.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if opt.device == "auto":
        opt.device = "cpu"
    if opt.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    opt.seed = int(args.seed)
    opt.num_clients = 4
    opt.num_task = 7
    opt.class_per_task = 100  # shared-label Data-IL; only informational upstream
    opt.num_classes = 100
    opt.nc = 100
    opt.batch_size = int(args.batch_size)
    opt.num_rounds = 2
    opt.num_local_epochs = max(args.local_epoch_schedule)
    opt.local_epoch_schedule = list(args.local_epoch_schedule)
    opt.gat_epochs = int(args.gat_epochs)
    opt.num_workers = int(args.num_workers)
    opt.pin_memory = bool(torch.cuda.is_available())
    opt.data_dir = str(Path(args.data_dir).resolve())
    opt.output_dir = str(Path(args.output_dir).resolve())
    opt.log_path = str(Path(opt.output_dir) / "run.log")
    opt.use_graph = True
    opt.use_temporal = True
    # Preserve GFedCL method-specific optimizer/privacy defaults from upstream.
    return opt


def inject_dataset_adapter(opt, args):
    # gfedcl.py imports utils.evaluation_utils at module load, then imports
    # utils.dataset_utils lazily in ParallelServerGFedCL.__init__.  Install only
    # the dataset submodule so all other upstream utilities remain untouched.
    import utils  # noqa: F401
    module = types.ModuleType("utils.dataset_utils")
    module.setup_cifar100_loaders = make_upstream_loader_factory(
        val_size=int(args.val_size),
        subset_per_client=int(args.subset_per_client),
        domain_order_name=args.domain_order,
        metadata_path=str(Path(opt.output_dir) / "controlled_domain_shift_metadata.json"),
    )
    sys.modules["utils.dataset_utils"] = module


def write_manifest(opt, args):
    path = Path(opt.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "GFedCL shared-label Data-IL adaptation",
        "upstream_repository": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "clients": 4,
        "stages": 7,
        "rounds_per_stage": 2,
        "local_epoch_schedule": list(args.local_epoch_schedule),
        "batch_size": int(args.batch_size),
        "val_size": int(args.val_size),
        "subset_per_client": int(args.subset_per_client),
        "domain_order": args.domain_order,
        "graph_classifier_epochs": int(args.graph_classifier_epochs),
        "gat_epochs": int(args.gat_epochs),
        "gfedcl_method_specific_compute": (
            "GFedCL additionally trains a stage classifier for relational-graph construction "
            "and, after stage 0, performs its upstream synthetic previous-stage pass."
        ),
        "architecture": (
            "Upstream GFedCL CIFAR architecture at 32x32 is retained; this comparator matches "
            "the data stream and stage schedule but not the LMSS ImageNet-pretrained backbone."
        ),
    }
    (path / "gfedcl_datail_manifest.json").write_text(json.dumps(payload, indent=2))


def postprocess_primary(output_dir: Path):
    src = output_dir / "all_tasks_accuracy.csv"
    if not src.exists():
        return
    rows = []
    with src.open(newline="") as f:
        for row in csv.DictReader(f):
            current = int(row["Current Task"])
            values = []
            for task in range(1, current + 1):
                raw = row.get(f"Task {task} Accuracy", "")
                if raw != "":
                    values.append(float(raw))
            seen = float(np.mean(values)) if values else float("nan")
            rows.append({
                "round": row["Round"],
                "current_stage": current,
                "seen_domain_accuracy": seen,
            })
    with (output_dir / "datail_seen_domain_accuracy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    post_stage0 = [r["seen_domain_accuracy"] for r in rows if int(r["current_stage"]) > 1]
    summary = {
        "round_averaged_seen_domain_accuracy_post_stage0": float(np.mean(post_stage0)),
        "num_post_stage0_rounds": len(post_stage0),
    }
    (output_dir / "datail_primary_summary.json").write_text(json.dumps(summary, indent=2))
    print("[GFedCL Data-IL] primary summary:", summary, flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--upstream-root", default="/tmp/GFedCL-pinned")
    ap.add_argument("--refresh-upstream", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--domain-order", default="heldout", choices=["development", "heldout"])
    args = ap.parse_args()

    if args.smoke:
        args.val_size = 700
        args.subset_per_client = 140
        args.batch_size = 32
        args.local_epoch_schedule = [1, 1]
        args.graph_classifier_epochs = 1
        args.gat_epochs = 1
    else:
        args.val_size = 5000
        args.subset_per_client = -1
        args.batch_size = 256
        args.local_epoch_schedule = [3, 2]
        args.graph_classifier_epochs = 20
        args.gat_epochs = 20
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    upstream = ensure_upstream(Path(args.upstream_root), args.refresh_upstream)
    patch_upstream(upstream / "CIFAR100" / "gfedcl.py", args.graph_classifier_epochs)
    opt = configure_opt(upstream, args)
    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)
    write_manifest(opt, args)
    inject_dataset_adapter(opt, args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(opt.log_path)],
    )
    from gfedcl import ParallelServerGFedCL

    trainer = ParallelServerGFedCL(opt)
    trainer.train_GFedCL()
    postprocess_primary(Path(opt.output_dir))


if __name__ == "__main__":
    main()
