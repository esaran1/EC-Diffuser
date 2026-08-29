import json, numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "experiments/loss_balance_audit")
d = json.load(open("experiments/loss_balance_audit/direction_audit.json"))
SE = (42, 43, 44)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
ax = axes[0]
for anc, c, mk in [("e16", "#1f77b4", "o"), ("e512", "#d62728", "s"), ("independent", "#2ca02c", "^")]:
    v = [d[f"decomposition_{anc}_anchored"][f"s{s}"]["cos_future"]["mean"] for s in SE]
    lo = [d[f"decomposition_{anc}_anchored"][f"s{s}"]["cos_future"]["ci"][0] for s in SE]
    hi = [d[f"decomposition_{anc}_anchored"][f"s{s}"]["cos_future"]["ci"][1] for s in SE]
    ax.errorbar(np.arange(3), v, yerr=[np.array(v)-lo, np.array(hi)-v],
                marker=mk, color=c, capsize=4, ls="none", label=f"{anc}-anchored")
ax.axhline(0, c="k", lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels([f"seed {s}" for s in SE])
ax.set_ylabel("cos(delta, toward observed future)")
ax.set_title("Cosine depends on matching anchor\n=> direction is ASSIGNMENT-UNSTABLE")
ax.legend(fontsize=8)

ax = axes[1]
al = d["interpolation"]["alphas"]
for s, c in zip(SE, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    ax.plot(al, d["interpolation"][f"s{s}"], marker="o", color=c, label=f"seed {s}")
ax.set_xlabel("alpha  (0 = E16, 1 = E512)"); ax.set_ylabel("chamfer to observed future")
ax.set_title("Interpolation peaks mid-path,\nnot monotone to E512")
ax.legend(fontsize=8)

ax = axes[2]
w = 0.35; x = np.arange(3)
par = [d["decomposition_e16_anchored"][f"s{s}"]["parallel_future"]["mean"] for s in SE]
orth = [d["decomposition_e16_anchored"][f"s{s}"]["orthogonal_future_mean"] for s in SE]
ax.bar(x, par, w, color="#1f77b4", label="parallel to target axis")
ax.bar(x + w, orth, w, color="#999", label="orthogonal")
ax.set_xticks(x + w/2); ax.set_xticklabels([f"seed {s}" for s in SE])
ax.set_ylabel("component magnitude")
ax.set_title("The shift is overwhelmingly ORTHOGONAL\n(|para|/|delta| ~ 0.16)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("experiments/figures/solver_bias_direction.png", dpi=160)
print("figure written")
