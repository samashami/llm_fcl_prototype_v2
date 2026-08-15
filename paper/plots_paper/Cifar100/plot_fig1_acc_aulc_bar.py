import numpy as np
import matplotlib.pyplot as plt

from paper_data import METHODS, OUTPUT_ROOT, final_stats


stats = final_stats()
method_keys = list(METHODS)
methods = [METHODS[key] for key in method_keys]
acc = stats.loc[method_keys, ("global_acc", "mean")].to_numpy() * 100
acc_std = stats.loc[method_keys, ("global_acc", "std")].to_numpy() * 100
aulc = stats.loc[method_keys, ("aulc_running", "mean")].to_numpy()
aulc_std = stats.loc[method_keys, ("aulc_running", "std")].to_numpy()

x = np.arange(len(methods))
width = 0.32

plt.figure(figsize=(7, 4.5))
ax1 = plt.gca()

# clearer error bars
error_style = dict(
    ecolor="black",
    elinewidth=1.5,
    capsize=4,
    capthick=1.5
)

# --- LEFT AXIS: ACCURACY ---
bars1 = ax1.bar(
    x - width/2,
    acc,
    width,
    yerr=acc_std,
    error_kw=error_style,
    color="#4C72B0",
    label="Accuracy (%)",
    zorder=3
)

ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_ylim(66.5, 72.7)

# --- RIGHT AXIS: AULC ---
ax2 = ax1.twinx()

bars2 = ax2.bar(
    x + width/2,
    aulc,
    width,
    yerr=aulc_std,
    error_kw=error_style,
    color="#DD8452",
    label="AULC",
    zorder=3
)

ax2.set_ylabel("AULC", fontsize=12)
ax2.set_ylim(0.63, 0.686)

# --- X axis ---
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=11)

# --- CLEAN STYLE ---
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(True)

# keep grid behind bars
ax1.set_axisbelow(True)
ax1.grid(axis="y", linestyle="--", alpha=0.2)

# --- LEGEND ---
ax1.legend(
    [bars1[0], bars2[0]],
    ["Accuracy (%)", "AULC"],
    frameon=False,
    fontsize=11,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98)
)

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "fig1_acc_aulc_bar_final.png", dpi=300, bbox_inches="tight")
plt.close()
