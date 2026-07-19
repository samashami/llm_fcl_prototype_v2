# Canonical CIFAR-100 paper results

This directory contains the canonical experimental outputs used for the
CIFAR-100 paper results. Results are organized by method (`fixed`, `v4`,
`lmss`, and `openrouter`) and random seed (`seed42`, `seed43`, and `seed44`).

## CSV types

- `fcl_run_summary_*.csv` contains one row per communication round with the
  global accuracy, loss, forgetting, divergence, communication cost, AULC,
  and controller settings. These summary CSVs are the canonical inputs for
  paper-level aggregation, tables, and figures.
- `fcl_run_results_*.csv` contains detailed per-client and per-epoch training
  and validation measurements for each run.
- `fcl_run_cl_batches_*.csv` records the continual-learning batch assignment
  and size for every client.

Analysis CSVs are derived artifacts and should be regenerated from the
canonical summary CSVs. Paper figures and tables should likewise be
regenerated from the CSVs in this directory rather than treated as primary
experimental data.

The files in this directory are preserved experimental outputs. They should
not be edited in place.
