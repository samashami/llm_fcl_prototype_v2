import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load CSVs
# -----------------------------
BASE_DIR = "experiments/results"

base_dir = os.path.join(BASE_DIR, "cifar100_baseline_20251120")
ctrl_dir = os.path.join(BASE_DIR, "cifar100_controller_v4_20251120")

base_summary = pd.read_csv(os.path.join(base_dir, 
                                        "fcl_run_summary_20251120-080254_cifar100_baseline_s42.csv"))
ctrl_summary = pd.read_csv(os.path.join(ctrl_dir, 
                                        "fcl_run_summary_20251120-093528_cifar100_v4_s42.csv"))

# -----------------------------
# 2. Plot Global Accuracy
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(base_summary["round"], base_summary["global_acc"], 
         label="Baseline (Fixed Policy)", marker="o")
plt.plot(ctrl_summary["round"], ctrl_summary["global_acc"], 
         label="Controller V4 (Rule-based)", marker="s")

plt.title("CIFAR-100 – Global Accuracy Over Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cifar100_acc_baseline_vs_v4.png", dpi=300)
print("Saved: cifar100_acc_baseline_vs_v4.png")
plt.show()

# -----------------------------
# 3. Plot Forgetting (mean)
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(base_summary["round"], base_summary["forget_mean"], 
         label="Baseline – Forget Mean", marker="o")
plt.plot(ctrl_summary["round"], ctrl_summary["forget_mean"], 
         label="Controller V4 – Forget Mean", marker="s")

plt.title("CIFAR-100 – Forgetting (Mean) Over Rounds")
plt.xlabel("Round")
plt.ylabel("Forget Mean")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cifar100_forget_mean.png", dpi=300)
print("Saved: cifar100_forget_mean.png")
plt.show()

# -----------------------------
# 4. Plot AULC Running
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(base_summary["round"], base_summary["aulc_running"],
         label="Baseline – AULC", marker="o")
plt.plot(ctrl_summary["round"], ctrl_summary["aulc_running"],
         label="Controller V4 – AULC", marker="s")

plt.title("CIFAR-100 – AULC Running Over Rounds")
plt.xlabel("Round")
plt.ylabel("AULC Running")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cifar100_aulc_running.png", dpi=300)
print("Saved: cifar100_aulc_running.png")
plt.show()

# -----------------------------
# 5. Plot Communication Cost (Cumulative)
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(base_summary["round"], base_summary["comm_bytes_cum"] / 1e9,
         label="Baseline – Comm (GB)", marker="o")
plt.plot(ctrl_summary["round"], ctrl_summary["comm_bytes_cum"] / 1e9,
         label="Controller V4 – Comm (GB)", marker="s")

plt.title("CIFAR-100 – Communication Cost (Cumulative GB)")
plt.xlabel("Round")
plt.ylabel("Comm Cost (GB)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cifar100_comm_cost_gb.png", dpi=300)
print("Saved: cifar100_comm_cost_gb.png")
plt.show()