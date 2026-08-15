"""Generate Table 2 from the canonical CIFAR-100 per-seed summaries."""

from paper_data import METHODS, OUTPUT_ROOT, final_stats


stats = final_stats()
rows = []
for method, label in METHODS.items():
    rows.append(
        {
            "Method": label,
            "Final accuracy (%)": (
                f"{stats.loc[method, ('global_acc', 'mean')] * 100:.2f} ± "
                f"{stats.loc[method, ('global_acc', 'std')] * 100:.2f}"
            ),
            "Mean forgetting": (
                f"{stats.loc[method, ('forget_mean', 'mean')]:.4f} ± "
                f"{stats.loc[method, ('forget_mean', 'std')]:.4f}"
            ),
            "Max forgetting": (
                f"{stats.loc[method, ('forget_max', 'mean')]:.4f} ± "
                f"{stats.loc[method, ('forget_max', 'std')]:.4f}"
            ),
            "AULC": (
                f"{stats.loc[method, ('aulc_running', 'mean')]:.4f} ± "
                f"{stats.loc[method, ('aulc_running', 'std')]:.4f}"
            ),
            "Communication (GiB)": (
                f"{stats.loc[method, ('comm_bytes_cum', 'mean')] / (1024 ** 3):.3f} ± "
                f"{stats.loc[method, ('comm_bytes_cum', 'std')] / (1024 ** 3):.3f}"
            ),
        }
    )

import pandas as pd

table = pd.DataFrame(rows)
table.to_csv(OUTPUT_ROOT / "table2.csv", index=False)
table.to_latex(OUTPUT_ROOT / "table2.tex", index=False, escape=True)
