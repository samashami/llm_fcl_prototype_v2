# Fixed-HighLR follow-up (Phase 1)

This branch adds a separate `fixed_highlr` controller to the frozen attribution study. Stage 0 retains `--lr 1e-4`; after stage 0, every round uses LR `1.5e-4`, replay fraction `0.5`, and no policy/API calls. The six original arms are unchanged.

The original stage-0 resume path validates hashes of the trainer source. **Do not bypass these checks** to load an old checkpoint. Produce a new shared stage-0 checkpoint using this branch, then run both Fixed-HighLR and an Eta-only reference from that same checkpoint if exact paired provenance is required. A new code version cannot be claimed byte-identical to the original held-out run without a verified compatibility audit.

The GitHub repository does not track `.pt` stage-0 checkpoints or the final held-out Eta-only action traces. Obtain those from the original run environment and audit their provenance before making exact-pairing claims.

Run the existing tests first:

```bash
python -m unittest tests.test_attribution_protocol tests.test_fixed_replay_budget tests.test_data_il_streams
```

Full-run base flags (GPU environment and CIFAR-100 dataset required):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m src.run_llm_fcl_controller \
  --attribution_protocol --stream_mode controlled_domain_shift \
  --blocks_per_stage 2 --rounds 14 --cl_batches 7 \
  --clients 4 --split_mode equal --epochs 5 --batch_size 256 \
  --lr 1e-4 --subset_per_client -1 --val_size 5000 \
  --optimizer adam --domain_order heldout --evaluation_source test \
  --controller fixed_highlr --control_mode joint --seed 42 \
  --tag fixed_highlr_s42
```

Repeat for seeds 43 and 44. For shared-start comparisons, pass `--resume_attribution_stage0 <compatible-seed-checkpoint.pt>` and use the same checkpoint for the reference Eta-only trajectory.

**Do not claim these runs are completed until the training outputs, action traces, and optimizer-presentation counts have been checked.** Evaluate paired seedwise primary accuracy and secondary outcomes. This follow-up was motivated by the observed Eta-only action distribution and is not an independently preregistered comparison.
