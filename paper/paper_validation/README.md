# Phase 0 CIFAR-100 forensics

`phase0 forensics.py` validates the CIFAR-100 values reported in Table 2 and
regenerates the associated diagnostic Figure 3 from the same dataframe. The
scientific and forensic checks are intentionally kept in that script.

## Expected repository structure

The default input is:

```text
experiments/results/cifar100/paper_results/
├── fixed/
├── v4/
├── lmss/
└── openrouter/
    └── seed42, seed43, seed44/
        └── fcl_run_summary_*.csv
```

Each of the four methods must contain one summary CSV for each of seeds 42,
43, and 44: 12 summary CSVs in total. Every summary CSV must contain `round`,
`divergence`, and `global_acc`. The method is read from the method directory,
the seed from the `seedNN` directory, and `global_acc` is used as accuracy.

The detailed `fcl_run_results_*.csv` and `fcl_run_cl_batches_*.csv` files are
part of the canonical artifact set but are not inputs to these Table 2 and
Figure 3 diagnostics.

Table 2 and Figure 3 use the same canonical runs but summarize divergence
differently:

- **Table 2 — Final-round divergence:** select the final round independently
  for each method and seed, then report the mean and sample standard deviation
  across seeds 42, 43, and 44.
- **Figure 3 — Divergence over rounds:** at every round, report the mean
  divergence across seeds 42, 43, and 44 to retain the complete trajectory.

Intermediate divergence spikes therefore appear in Figure 3 but do not enter
Table 2 except when they occur in a seed's final round.

## Running

From the repository root:

```bash
python "paper/paper_validation/phase0 forensics.py"
```

Outputs are written to `phase0_out/` by default. Use `--out PATH` to select a
different output directory. The original synthetic diagnostic remains
available with `--demo`.
