import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/results/cifar100/aggregated/cifar100_per_round_stats.csv")

order = ["v4", "sft", "baseline"]  # baseline last (on top)

styles = {
    "baseline": dict(linestyle="--", linewidth=3, marker=None),
    "v4": dict(linestyle="-", linewidth=2, marker="o"),
    "sft": dict(linestyle="-", linewidth=2, marker="o"),
}

names = {
    "baseline": "Baseline",
    "v4": "Heuristic V4",
    "sft": "LLM-guided (SFT)"
}

for ctrl in order:
    d = df[df.controller == ctrl].sort_values("round")

    plt.plot(
        d["round"],
        d["acc_mean"],
        label=names[ctrl],
        zorder=5,
        **styles[ctrl]
    )

    plt.fill_between(
        d["round"],
        d["acc_mean"] - d["acc_std"],
        d["acc_mean"] + d["acc_std"],
        alpha=0.12,
        zorder=1
    )

plt.xlabel("Communication Rounds")
plt.ylabel("Global Accuracy")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("plots_paper/Cifar100/fig1_acc_vs_rounds.png", dpi=300)
plt.show()