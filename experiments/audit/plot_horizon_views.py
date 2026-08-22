"""Native vs fixed-horizon scaling views. All values ingested from JSON."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("experiments/audit/fixed_horizon_analysis.json"))
ARMS = [("gaussian_nfe100","Gaussian @100","#d62728","s"),
        ("flow_nfe4","Flow @4","#2ca02c","o"),
        ("flow_nfe1","Flow @1","#1f77b4","^")]
NAT = {3:100,4:150,5:200}
OUT = "experiments/audit/figures"

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
for row, view in enumerate(("native", "fixed")):
    for col, (metric, ylab) in enumerate((("per_object_success","per-object success"),
                                          ("success_rate","full success"))):
        ax = axes[row][col]
        for lab, name, c, m in ARMS:
            xs = [n for n in (3,4,5) if f"{n}cube_{view}_{lab}" in D["tables"]]
            ys = [D["tables"][f"{n}cube_{view}_{lab}"][metric] for n in xs]
            ax.plot(xs, ys, m+"-", color=c, label=name, lw=2, ms=8)
        ax.set_xticks([3,4,5]); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_ylabel(ylab); ax.set_xlabel("number of cubes")
        if view == "native":
            ax.set_xticklabels([f"{n}\n(H={NAT[n]})" for n in (3,4,5)])
            ax.set_title(f"NATIVE horizon — {ylab}", fontsize=11)
        else:
            ax.set_xticklabels([f"{n}\n(H=100)" for n in (3,4,5)])
            ax.set_title(f"FIXED H=100 — {ylab}", fontsize=11)

    ax = axes[row][2]
    g = D["nfe_gap"].get(view, {})
    xs = sorted((int(k) for k in g), key=int)
    ys = [g[str(n)]["gap"] for n in xs]
    lo = [g[str(n)]["gap"]-g[str(n)]["ci"][0] for n in xs]
    hi = [g[str(n)]["ci"][1]-g[str(n)]["gap"] for n in xs]
    ax.errorbar(xs, ys, yerr=[lo,hi], fmt="D-", color="#6a3d9a", lw=2, ms=8, capsize=4)
    ax.axhline(0, ls="--", c="k", alpha=.6)
    ax.set_xticks(xs); ax.set_xlabel("number of cubes")
    ax.set_ylabel("Flow@4 − Flow@1 (per-object)")
    ax.grid(alpha=.3)
    ax.set_title(f"{'NATIVE' if view=='native' else 'FIXED H=100'} — low-NFE penalty", fontsize=11)
    if view == "fixed":
        ax.annotate("sign flips at 4 cubes:\npenalty is horizon-dependent",
                    xy=(0.04,0.06), xycoords="axes fraction", fontsize=8,
                    bbox=dict(boxstyle="round", fc="#f8d7da", ec="#721c24"))

plt.suptitle("Object count vs execution time — one training seed per method; "
             "zero-shot POLICY generalization (DLP saw up to 6 cubes)", fontsize=12)
plt.tight_layout()
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "horizon_views.png")
plt.savefig(p, dpi=150); print("wrote", p)
