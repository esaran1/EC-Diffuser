import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open("experiments/loss_balance_audit/multinoise_analysis.json"))
su, SE = r["summary"], (42, 43, 44)
C16, C512 = "#1f77b4", "#d62728"

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
ax = axes[0]
mets = ["mean_single_sample_error", "medoid_error", "best_of_8", "dispersion"]
lab = ["mean\nsingle-sample", "medoid\n(centre)", "best-of-8\n(coverage)", "dispersion\n(spread)"]
x = np.arange(len(mets)); w = 0.36
for i, (a, c) in enumerate([("euler16", C16), ("euler512", C512)]):
    ax.bar(x + i*w, [su[f"{m}_{a}"]["mean"] for m in mets], w,
           yerr=[su[f"{m}_{a}"]["sd"] for m in mets], color=c, capsize=3, label=a)
ax.set_xticks(x + w/2); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("chamfer"); ax.legend(fontsize=8)
ax.set_title("E512 is worse on every error measure\nAND more dispersed")

ax = axes[1]
ks = [1, 2, 4, 8]
ax.plot(ks, r["best_of_K"]["euler16"], marker="o", color=C16, label="euler16")
ax.plot(ks, r["best_of_K"]["euler512"], marker="s", color=C512, label="euler512")
ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
ax.set_xlabel("K draws"); ax.set_ylabel("expected best-of-K error")
ax.set_title("Best-of-K: E16 leads at every K\n(E512 never overtakes)")
ax.legend(fontsize=8)

ax = axes[2]
pn = r["paired_per_noise_E512_minus_E16"]
xs = np.arange(3)
ax.bar(xs, [pn[f"s{s}"]["mean"] for s in SE],
       yerr=[[pn[f"s{s}"]["mean"]-pn[f"s{s}"]["ci"][0] for s in SE],
             [pn[f"s{s}"]["ci"][1]-pn[f"s{s}"]["mean"] for s in SE]],
       color="#555", capsize=4)
for i, s in enumerate(SE):
    ax.text(i, pn[f"s{s}"]["mean"]*1.05, f"{pn[f's{s}']['frac_noises_favouring_E16']*100:.0f}%\nE16 wins",
            ha="center", fontsize=8)
ax.axhline(0, c="k", lw=1)
ax.set_xticks(xs); ax.set_xticklabels([f"seed {s}" for s in SE])
ax.set_ylabel("paired E512 - E16 error"); ax.set_ylim(0, 0.0026)
ax.set_title("Paired per-noise degradation\n(768 noises/seed, CI excludes 0)")
fig.tight_layout(); fig.savefig("experiments/figures/multinoise_solver_bias.png", dpi=160)
print("figure written")
