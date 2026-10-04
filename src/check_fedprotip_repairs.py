#!/usr/bin/env python3
"""Fast validation for the two final FedProTIP audit fixes.

Checks:
1. GPM activation collection processes every example in the selected loader
   batches while forwarding them in small chunks.
2. Later-stage 1-D backbone parameters remain bitwise unchanged across an Adam
   step after their gradients are suppressed with ``grad=None``.
"""
from __future__ import annotations

import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from src.external.fedprotip_projection import FedProTIPProjector, SharedGPMState
from src.model import build_resnet18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(123)
    model = build_resnet18(100).to(device)
    shared = SharedGPMState(threshold=0.70, max_columns=32)
    projector = FedProTIPProjector(model, shared, seed=123)
    frozen = projector.freeze_early_blocks()
    if len(frozen) != 30:
        raise RuntimeError(f"expected 30 frozen early-backbone parameters, got {len(frozen)}")

    # Check 1: 17 examples in two loader batches must all be processed even
    # though the forward microbatch is only 4 examples.
    x = torch.randn(17, 3, 32, 32)
    y = torch.randint(0, 100, (17,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10, shuffle=False)
    record = projector.update_from_loader(
        loader,
        stage=0,
        client_id=0,
        device=device,
        max_batches=2,
        activation_microbatch=4,
    )
    if record["processed_examples"] != 17:
        raise RuntimeError(
            "basis collection dropped examples: "
            f"processed={record['processed_examples']} expected=17"
        )
    if record["processed_loader_batches"] != 2:
        raise RuntimeError(
            "unexpected selected loader-batch count: "
            f"{record['processed_loader_batches']}"
        )

    # Check 2: create Adam momentum on trainable later-stage parameters, then
    # prove that grad=None makes protected 1-D backbone parameters unchanged.
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()

    warm_x = torch.randn(8, 3, 32, 32, device=device)
    warm_y = torch.randint(0, 100, (8,), device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(warm_x), warm_y)
    loss.backward()
    optimizer.step()

    protected = projector.protected_backbone_1d_parameters()
    if not protected:
        raise RuntimeError("no trainable protected 1-D backbone parameters found")

    before = {name: p.detach().clone() for name, p in protected.items()}
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(warm_x), warm_y)
    loss.backward()
    suppressed = projector.suppress_backbone_1d_gradients()
    if suppressed != len(protected):
        raise RuntimeError(
            f"suppressed {suppressed} protected gradients, expected {len(protected)}"
        )
    optimizer.step()

    max_change = 0.0
    changed = []
    for name, param in protected.items():
        delta = float((param.detach() - before[name]).abs().max().item())
        max_change = max(max_change, delta)
        if delta != 0.0:
            changed.append((name, delta))
    if changed:
        raise RuntimeError(f"protected 1-D parameters changed under Adam: {changed[:5]}")

    print(
        "[FedProTIP repair check] PASS | "
        f"basis_examples={record['processed_examples']}/17 | "
        f"loader_batches={record['processed_loader_batches']}/2 | "
        f"protected_1d={len(protected)} | "
        f"adam_max_parameter_change={max_change:.3e} | "
        f"max_basis_orthogonality_error={record['max_orthogonality_error']:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
