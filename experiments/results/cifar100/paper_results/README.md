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

The publication scripts in `plots_paper/Cifar100/` validate the complete set
of 36 CSVs before loading these summary files. Run the figure scripts and
`generate_table2.py` from any working directory to regenerate the CIFAR-100
paper artifacts using repository-relative paths.

From the repository root, run:

```bash
python plots_paper/Cifar100/plot_fig1_acc_aulc_bar.py
python plots_paper/Cifar100/plot_fig1_acc_vs_rounds.py
python plots_paper/Cifar100/plot_fig2_forgetting_vs_rounds.py
python plots_paper/Cifar100/plot_fig3_aulc_vs_rounds.py
python plots_paper/Cifar100/plot_fig4_comm_vs_rounds.py
python plots_paper/Cifar100/generate_table2.py
```

The scripts write the regenerated PNG figures and `table2.csv`/`table2.tex`
next to the scripts in `plots_paper/Cifar100/`.
