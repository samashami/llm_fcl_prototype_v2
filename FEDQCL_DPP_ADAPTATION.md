# FedQCL-DPP adaptation for shared-head Data-IL

This branch adds a comparator named `fedqcl_dpp` to the controlled-domain-shift attribution protocol. It adapts the published FedQCL drift-plus-penalty (DPP) mechanism to the repository's shared-head, domain-incremental stream. It is not an exact reproduction of the paper's original split-class architecture, optimizer, memory policy, or queue-update schedule.

## Adapted objective

For client `c`, historical domain `k`, and communication round `t`, the client keeps a nonnegative queue `Q[c,k]`. At the beginning of each new domain stage, the currently aggregated global model is the fixed reference for the duration of that stage. The reference loss is measured on the client's stored examples from each prior domain.

The local objective for a minibatch is

`V * CE(current examples) + sum_k Q[c,k] * (CE(replay domain k) - CE(reference model on domain k))`.

After local training and before FedAvg, each queue is updated as

`Q[c,k] = max(0, Q[c,k] + CE(local model on k) - CE(stage-start reference on k) - delta)`.

The stage reference term in the local objective is a fixed scalar, so it does not contribute a gradient. The queue-weighted replay CE does contribute a gradient to the next local optimization step. Queues persist across stages; newly available historical domains start with queue zero. Queue measurements use only each client's private replay memory, not validation or test labels.

The action remains fixed at the protocol's `lr` and replay ratio `0.50`; FedQCL-DPP acts through the local objective. The replay share is taken from the same fixed minibatch budget as the other attribution controllers. Memory is labelled by domain and balanced across seen domains under a total capacity of 2,000 examples per client.

## Frozen parameters and timing

- `V = 200`, the CIFAR-100 value used by the authors' published `run.sh` configuration.
- `delta = 1.0`, the paper's CIFAR-100 setting used in this comparison.
- Queue updates occur after every communication round, as specified in the paper's round-indexed queue equation. The authors' public implementation updates queues at task boundaries; this timing difference is an explicit adaptation for our two-block-per-domain schedule.
- The optimizer, stream, model, client count, replay budget, and evaluation split remain those of the common attribution protocol. No result-driven parameter search is part of this implementation.

## Qualification evidence

The unit tests verify that (1) the stage reference cannot change within a stage, (2) positive replay-loss violations increase queues while negative violations clamp at zero, (3) a positive queue adds a replay gradient and changes the resulting client optimizer update, and (4) domain-stratified memory stays within its configured capacity. Run a small end-to-end controlled-stream smoke before launching a full seed matrix.

Focused unit test:

```bash
python -m unittest tests.test_fedqcl_dpp
```

GPU smoke using the repository's reduced attribution configuration:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m src.run_llm_fcl_controller \
  --attribution_protocol --attribution_smoke \
  --stream_mode controlled_domain_shift \
  --blocks_per_stage 2 --rounds 14 --cl_batches 7 \
  --domain_order development --evaluation_source validation \
  --controller fedqcl_dpp --control_mode joint \
  --clients 4 --split_mode equal --seed 41 \
  --epochs 2 --subset_per_client 140 --batch_size 32 \
  --lr 1e-4 --val_size 700 --num_workers 2 --optimizer adam \
  --output_dir /kaggle/working/fedqcl_dpp_smoke \
  --tag fedqcl_dpp_smoke
```

The smoke qualifies only if it completes all 14 rounds, every logged round has
the expected fixed-compute counts, stage reference hashes stay unchanged within
each pair of blocks, queue updates are present when historical memory exists,
and the CSV/JSON artifacts report the local penalty and queue values. A queue
remaining zero is not a code failure; the unit tests separately establish that
a positive violation is accumulated and a positive queue changes gradients.

## Interpretation limits

Report this as “FedQCL-DPP adapted to shared-head Data-IL.” Do not describe it as an exact reproduction or as a faithful reproduction of the source code's task-boundary timing. The adaptation has one fixed shared-head validation protocol and should be compared as an empirical baseline, not used to claim that DPP control is new. A queue that remains zero in an experiment is a valid measured outcome and should be reported; thresholds must not be tuned after seeing evaluation results.

Reference paper and authors' code:

- Shah et al., *Federated Continual Learning as a Distributed Drift-Plus-Penalty Control Problem*, CoLLAs 2026 / arXiv:2608.21539.
- Official implementation: <https://github.com/Naveensomireddy4/FedQCL_Task>.
