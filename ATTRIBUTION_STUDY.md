# Feedback-attribution study protocol

This file defines the preregistered feedback-attribution pilot. FedQCL-DPP's
code adaptation and smoke qualification are specified separately in
`FEDQCL_DPP_ADAPTATION.md`; its full comparator run remains downstream of the
pilot decision gate. BI adaptation is not part of this branch's implementation.

## Frozen design

- Seven shared-label CIFAR-100 domain stages.
- Two communication/control blocks per stage (14 rounds).
- Five local epochs per stage split as 3 + 2; early stopping is disabled.
- Replay reallocates each optimizer batch between current and historical
  samples. It does not increase batch size.
- Current-stage samples enter memory once, after the stage finishes.
- Development order: clean, brightness, contrast, blur, noise, saturation,
  brightness-plus-blur.
- Held-out order: clean, noise, contrast, blur, brightness, saturation,
  brightness-plus-blur.
- Controllers receive validation-only feedback. Development runs cannot select
  test evaluation through the CLI.
- The reported divergence input is mean client update distance divided by the
  pre-aggregation global parameter norm. Raw mean distance and relative
  update-magnitude dispersion are logged separately.

## Required Kaggle validation

```bash
python -m unittest tests.test_attribution_protocol tests.test_fixed_replay_budget tests.test_data_il_streams
```

Run a small smoke test before the full pilot:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m src.run_llm_fcl_controller \
  --attribution_protocol --attribution_smoke \
  --stream_mode controlled_domain_shift \
  --blocks_per_stage 2 --rounds 14 --cl_batches 7 \
  --domain_order development --evaluation_source validation \
  --controller fixed --control_mode joint \
  --clients 4 --split_mode equal --seed 41 \
  --epochs 2 --subset_per_client 140 --batch_size 32 \
  --lr 1e-4 --val_size 700 --num_workers 2 --optimizer adam \
  --tag attribution_smoke
```

## Six-trajectory development pilot

Use this frozen base for every evidence run:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m src.run_llm_fcl_controller \
--attribution_protocol --stream_mode controlled_domain_shift \
--blocks_per_stage 2 --rounds 14 --cl_batches 7 \
--clients 4 --split_mode equal --epochs 5 --batch_size 256 \
--lr 1e-4 --subset_per_client -1 --val_size 5000 \
--optimizer adam
```

The commands below show the treatment-specific arguments only. Every run
writes `run_protocol.json`; a frozen schedule is rejected unless that manifest
and every source action prove that it came from the preregistered, fallback-free
seed-40 live joint LMSS run using the same explicit model and policy code.

1. Live LMSS, seed 40: record `runs/<run_id>/action_round_*.json`.

```bash
--controller lmss_openrouter --control_mode joint --seed 40 \
--domain_order development --evaluation_source validation \
--lmss_model openai/gpt-4o-mini
```

2. Fixed, seed 41: create the immutable shared stage-0 checkpoint and continue
   the fixed trajectory.

```bash
--controller fixed --control_mode joint --seed 41 \
--domain_order development --evaluation_source validation \
--save_attribution_stage0 artifacts/dev_s41_stage0.pt
```

3. Frozen seed-40 LMSS schedule, seed 41:

```bash
--controller frozen_lmss --control_mode joint --seed 41 \
--domain_order development --evaluation_source validation \
--lmss_model openai/gpt-4o-mini \
--frozen_action_schedule <seed40-run-action-directory> \
--resume_attribution_stage0 artifacts/dev_s41_stage0.pt
```

4. Live LMSS joint, seed 41:

```bash
--controller lmss_openrouter --control_mode joint --seed 41 \
--domain_order development --evaluation_source validation \
--lmss_model openai/gpt-4o-mini \
--resume_attribution_stage0 artifacts/dev_s41_stage0.pt
```

5. Live LMSS with only learning rate active, seed 41: use the preceding command
   with `--control_mode eta_only`.

6. Live LMSS with only replay active, seed 41: use the preceding command with
   `--control_mode rho_only`.

## Pilot gate

Use validation metrics only. Proceed when at least one central contrast reaches
0.5 percentage points in `round_averaged_seen_domain_accuracy`, relevant action
traces differ, and mean historical-domain accuracy is not more than 0.5 points
worse. Otherwise stop the extension. Do not inspect held-out test results to
change this decision rule.
