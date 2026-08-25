"""Presentation figure: F4-F1 per-object gap, seeds 42 and 43, fixed H=100.

All values are read from experiments/audit/seed_replication.json, which the
canonical analysis produced from raw per-episode records. Nothing is typed in.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open("experiments/audit/seed_replication.json"))
TASKS = [("3cube", "3 cubes"), ("4cube", "4 cubes"), ("5cube", "5 cubes")]
SEEDS = [("seed42", "Seed 42", "#1f77b4", "o"), ("seed43", "Seed 43", "#ff7f0e", "s")]
x = np.arange(len(TASKS))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8),
                              gridspec_kw={"width_ratios": [2.1, 1]})

for i, (key, label, c, m) in enumerate(SEEDS):
    ys = [D[key]["per_task"][t]["delta"] for t, _ in TASKS]
    ax.plot(x, ys, m + "-", color=c, lw=2.5, ms=11, label=label, zorder=3)
    for xi, y in zip(x, ys):
        ax.annotate(f"{y*100:+.1f}", (xi, y), textcoords="offset points",
                    xytext=(0, 12 if i == 0 else -20), ha="center",
                    fontsize=10, color=c, fontweight="bold")

ax.axhline(0, ls="--", c="k", lw=1.2, alpha=.7)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in TASKS], fontsize=12)
ax.set_ylabel("Flow@4 − Flow@1\nper-object success", fontsize=12)
ax.set_title("Per-task benefit of 4 steps vs 1 step  (fixed H=100)", fontsize=13)
ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=11, loc="lower left")
ax.set_ylim(-0.06, 0.11)
ax.annotate("sign flips\nbetween seeds", xy=(1, -0.029), xytext=(1.25, -0.05),
            fontsize=9.5, color="#b22222",
            arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.4))

# Right panel: the aggregate endpoint, which is what replicates.
means = [D[k]["primary"] for k, _, _, _ in SEEDS]
los = [D[k]["primary"] - D[k]["primary_ci95"][0] for k, _, _, _ in SEEDS]
his = [D[k]["primary_ci95"][1] - D[k]["primary"] for k, _, _, _ in SEEDS]
cols = [c for _, _, c, _ in SEEDS]
ax2.bar([0, 1], means, yerr=[los, his], color=cols, width=.55, capsize=7,
        edgecolor="k", linewidth=.8)
for xi, v in zip([0, 1], means):
    ax2.annotate(f"{v*100:+.1f} pts", (xi, v), textcoords="offset points",
                 xytext=(0, 26), ha="center", fontsize=12, fontweight="bold")
ax2.axhline(0, ls="--", c="k", lw=1.2, alpha=.7)
ax2.set_xticks([0, 1]); ax2.set_xticklabels([n for _, n, _, _ in SEEDS], fontsize=12)
ax2.set_ylabel("mean Δ across 3/4/5 cubes", fontsize=12)
ax2.set_title("Aggregate effect replicates", fontsize=13)
ax2.grid(alpha=.3, axis="y"); ax2.set_ylim(0, 0.105)

plt.suptitle("Four Flow steps vs one — aggregate benefit replicates, "
             "task localisation does not", fontsize=14)
plt.tight_layout()
os.makedirs("experiments/figures", exist_ok=True)
p = "experiments/figures/two_seed_nfe_replication.png"
plt.savefig(p, dpi=160); print("wrote", p)
