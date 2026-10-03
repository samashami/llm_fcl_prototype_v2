# GFedCL Data-IL Adaptation

This branch adapts the official GFedCL CIFAR-100 implementation to the paper's frozen controlled Data-IL stream.

## Upstream

- Repository: https://github.com/IntelliSys-Lab/GFedCL
- Pinned commit: `daa9faa4af4e80f6bba79bc020c10a0842c3cfb7`

## What is preserved

The adaptation retains GFedCL's defining mechanism:

- client relational graph construction from model updates;
- temporal/spatial graph attention;
- server discriminator;
- graph-conditioned client training;
- upstream synthetic previous-stage pass;
- upstream CIFAR model family and 32x32 inputs.

## What is adapted

Only the evaluation protocol is aligned to the paper:

- CIFAR-100 train split with 5,000 validation examples;
- four equal clients;
- seven class-stratified, non-overlapping Data-IL stages;
- the same controlled domain transforms and held-out order;
- two communication rounds per stage;
- current-stage local epoch schedule 3+2 for full runs;
- seeds 42, 43 and 44 for held-out evaluation.

Because the official GFedCL CIFAR implementation is not an ImageNet-pretrained ResNet-18 FedAvg pipeline, the architecture is intentionally not replaced. This is therefore a protocol-adapted external comparator, not a same-backbone ablation.

GFedCL also performs method-specific extra compute for relational-graph construction and synthetic previous-stage training. That overhead must be reported rather than treated as compute-matched to LMSS.

## Smoke

```bash
python -m src.run_gfedcl_datail \
  --smoke \
  --seed 42 \
  --device cuda \
  --output-dir experiments/paper2026/GFEDCL/smoke_s42
```

## Full held-out run

```bash
python -m src.run_gfedcl_datail \
  --seed 42 \
  --device cuda \
  --output-dir experiments/paper2026/GFEDCL/heldout_s42
```

Repeat for seeds 43 and 44 only after the smoke run is inspected.
