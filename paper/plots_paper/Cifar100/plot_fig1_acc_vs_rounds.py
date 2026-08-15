import matplotlib.pyplot as plt

from paper_data import METHODS, OUTPUT_ROOT, per_round_stats


stats = per_round_stats()
styles = {
    "fixed": dict(color="black", linestyle="--", linewidth=2.2, marker=None),
    "v4": dict(color="#d62728", linestyle="-", linewidth=2.0, marker="o", markersize=4),
    "lmss": dict(color="#1f77b4", linestyle="-", linewidth=2.0, marker="o", markersize=4),
    "openrouter": dict(color="#2ca02c", linestyle="-", linewidth=2.2, marker="s", markersize=4),
}

plt.figure(figsize=(7.2, 4.8), facecolor="white")
ax = plt.gca()
ax.set_facecolor("white")

for method, label in METHODS.items():
    data = stats[stats["method"] == method].sort_values("round")
    rounds = data["round"].to_numpy()
    mean = data[("global_acc", "mean")].to_numpy() * 100
    std = data[("global_acc", "std")].to_numpy() * 100
    ax.plot(rounds, mean, label=label, zorder=3, **styles[method])
    ax.fill_between(
        rounds,
        mean - std,
        mean + std,
        alpha=0.05,
        zorder=1,
        color=styles[method]["color"],
    )

ax.set_xlabel("Communication Rounds", fontsize=13)
ax.set_ylabel("Global Accuracy (%)", fontsize=13)
ax.set_xticks(sorted(stats["round"].unique()))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=11)
ax.legend(frameon=False, fontsize=11, loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "fig1_acc_vs_rounds.png", dpi=300, bbox_inches="tight")
plt.close()
