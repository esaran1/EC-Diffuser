"""Figures for the solver/reference audit."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("experiments/loss_balance_audit/solver_reference_audit.json"))
C = {"euler": "#1f77b4", "midpoint": "#2ca02c", "rk4": "#9467bd"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
for meth in ["euler", "midpoint", "rk4"]:
    rs = d["raw_convergence"][meth]
    ax.loglog([r["nfe"] for r in rs], [r["all_free"]["mean_abs"] for r in rs],
              marker="o", color=C[meth], label=meth)
n = np.array([32, 512])
ax.loglog(n, 0.25 / n, "k--", lw=1, label="order 1 slope")
ax.loglog(n, 20.0 / n**2, "k:", lw=1, label="order 2 slope")
ax.set_xlabel("NFE"); ax.set_ylabel("mean |endpoint difference| vs next-coarser")
ax.set_title("Raw free-coordinate convergence\n(all schemes ~order 1)")
ax.legend(fontsize=8)

ax = axes[1]
for meth in ["euler", "midpoint", "rk4"]:
    rs = d["raw_convergence"][meth]
    ax.loglog([r["nfe"] for r in rs], [r["action"]["mean_abs"] for r in rs],
              marker="o", color=C[meth], label=f"{meth} action")
    ax.loglog([r["nfe"] for r in rs], [r["observation"]["mean_abs"] for r in rs],
              marker="s", ls="--", color=C[meth], alpha=0.5, label=f"{meth} obs")
ax.set_xlabel("NFE"); ax.set_ylabel("mean |endpoint difference|")
ax.set_title("Action vs observation coordinates\n(converge together)")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig("experiments/figures/audit_raw_convergence.png", dpi=160)

# OOD figure
fig, ax = plt.subplots(figsize=(7, 4.2))
o = d["rk_intermediate_ood"]
for i, (arm, recs) in enumerate(o.items()):
    ts = [r["t"] for r in recs]; ds = [r["dist_to_euler16_path"] for r in recs]
    ax.plot(ts, ds, marker="o", label=arm)
    for r in recs:
        if r["stage"] in ("k3", "k4") and r["dist_to_euler16_path"] > 3:
            ax.annotate(r["stage"], (r["t"], r["dist_to_euler16_path"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.set_xlabel("t at network evaluation"); ax.set_ylabel("distance from Euler@16 trajectory")
ax.set_title("RK intermediate stages leave the trajectory\n(RK4 k3/k4 furthest)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("experiments/figures/audit_rk_ood.png", dpi=160)
print("figures written")
