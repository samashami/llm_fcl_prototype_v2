import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/results/cifar100/aggregated/cifar100_per_round_stats.csv")

order = ["v4", "sft", "baseline"]  # baseline last
names = {
    "baseline": "Baseline",
    "v4": "Heuristic V4",
    "sft": "LLM-guided (SFT)"
}
styles = {
    "baseline": dict(color="black", linestyle="--", linewidth=2.5),
    "v4": dict(color="#d62728", linestyle="-", linewidth=2, marker="o"),
    "sft": dict(color="#1f77b4", linestyle="-", linewidth=2, marker="o"),
}

plt.figure(figsize=(6.5, 4.5))

for ctrl in order:
    d = df[df.controller == ctrl].sort_values("round")
    plt.plot(
        d["round"],
        d["comm_cum_mean"],
        label=names[ctrl],
        zorder=5,
        **styles[ctrl]
    )
    plt.fill_between(
        d["round"],
        d["comm_cum_mean"] - d["comm_cum_std"],
        d["comm_cum_mean"] + d["comm_cum_std"],
        alpha=0.12,
        zorder=1
    )

plt.xlabel("Communication Rounds")
plt.ylabel("Cumulative Communication (Bytes)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("plots_paper/Cifar100/fig4_comm_vs_rounds.png", dpi=300)
plt.show()