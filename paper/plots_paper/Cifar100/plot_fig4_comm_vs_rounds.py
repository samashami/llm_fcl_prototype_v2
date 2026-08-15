import matplotlib.pyplot as plt

from paper_data import METHODS, OUTPUT_ROOT, per_round_stats


stats = per_round_stats()
styles = {
    "fixed": dict(color="black", linestyle="--", linewidth=2.2),
    "v4": dict(color="#d62728", linestyle="-", linewidth=2.0, marker="o"),
    "lmss": dict(color="#1f77b4", linestyle="-", linewidth=2.0, marker="o"),
    "openrouter": dict(color="#2ca02c", linestyle="-", linewidth=2.2, marker="s"),
}

plt.figure(figsize=(7, 4.5))
ax = plt.gca()
for method, label in METHODS.items():
    data = stats[stats["method"] == method].sort_values("round")
    rounds = data["round"].to_numpy()
    mean = data[("comm_bytes_cum", "mean")].to_numpy() / (1024 ** 3)
    std = data[("comm_bytes_cum", "std")].to_numpy() / (1024 ** 3)
    ax.plot(rounds, mean, label=label, zorder=3, **styles[method])
    ax.fill_between(
        rounds,
        mean - std,
        mean + std,
        alpha=0.08,
        zorder=1,
        color=styles[method]["color"],
    )

ax.set_xlabel("Communication Rounds")
ax.set_ylabel("Cumulative Communication (GiB)")
ax.set_xticks(sorted(stats["round"].unique()))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "fig4_comm_vs_rounds.png", dpi=300, bbox_inches="tight")
plt.close()
