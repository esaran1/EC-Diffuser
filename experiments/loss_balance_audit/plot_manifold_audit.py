import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("experiments/loss_balance_audit/manifold_audit.json"))
K, gt = d["knn_distance"], d["ground_truth_error_within_run"]
SE = (42, 43, 44)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
pts = [("held-out real", 0.0, np.mean(K["heldout_real"]["5"]), "k", "*"),
       ("Gaussian@100", gt["gaussian100"], np.mean(K["gaussian100"]["5"]), "#7f7f7f", "s")]
for a, c, mk in [("euler16", "#1f77b4", "o"), ("euler512", "#d62728", "^"), ("copy", "#2ca02c", "D")]:
    pts.append((a, np.mean([gt[f"s{s}_{a}"] for s in SE]),
                np.mean([np.mean(K[f"s{s}_{a}"]["5"]) for s in SE]), c, mk))
for lab, x, y, c, mk in pts:
    ax.scatter(x, y, c=c, marker=mk, s=90, zorder=3, label=lab)
ax.axhline(np.mean(K["heldout_real"]["5"]), ls="--", c="k", lw=1, alpha=.6)
ax.set_xlabel("ground-truth imagination error"); ax.set_ylabel("manifold distance (k=5)")
ax.set_title("Prediction quality vs data-manifold proximity\n(dashed = held-out real baseline)")
ax.legend(fontsize=8)

ax = axes[1]
w, xs = 0.35, np.arange(3)
for i, (a, c) in enumerate([("euler16", "#1f77b4"), ("euler512", "#d62728")]):
    ax.bar(xs + i * w, [np.mean(K[f"s{s}_{a}"]["5"]) for s in SE], w, color=c, label=a)
ax.axhline(np.mean(K["heldout_real"]["5"]), ls="--", c="k", lw=1, label="held-out real")
ax.set_xticks(xs + w / 2); ax.set_xticklabels([f"seed {s}" for s in SE])
ax.set_ylim(0.060, 0.072); ax.set_ylabel("manifold distance (k=5)")
ax.set_title("E16 vs E512 manifold distance\n(difference is tiny vs the real-data baseline)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("experiments/figures/manifold_audit.png", dpi=160)
print("figure written")
