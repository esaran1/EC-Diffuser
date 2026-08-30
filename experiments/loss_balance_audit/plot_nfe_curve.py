import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("experiments/loss_balance_audit/nfe_curve_analysis.json"))
NF = [int(k) for k in d["curve"]]; NF.sort()
act = np.array([d["curve"][str(n)]["action_mean"] for n in NF])
st = np.array([d["curve"][str(n)]["state_mean"] for n in NF])
acts = np.array([d["curve"][str(n)]["action_sd"] for n in NF])
sts = np.array([d["curve"][str(n)]["state_sd"] for n in NF])

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
ax.errorbar(NF, act / act.min(), yerr=acts / act.min(), marker="o", color="#d62728",
            capsize=3, label="ACTION error (norm. to its best)")
ax.errorbar(NF, st / st.min(), yerr=sts / st.min(), marker="s", color="#1f77b4",
            capsize=3, label="STATE error (norm. to its best)")
ax.axvline(2, ls=":", c="#d62728", alpha=.6); ax.axvline(32, ls=":", c="#1f77b4", alpha=.6)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(NF); ax.set_xticklabels(NF, fontsize=8)
ax.set_xlabel("NFE"); ax.set_ylabel("error / best observed")
ax.set_title("Arm-neutral error vs NFE (3-seed mean)\nACTION best at 2, STATE best at 32 — opposite directions")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(NF, act, marker="o", color="#d62728", label="action (raw)")
ax.set_xscale("log", base=2); ax.set_xticks(NF); ax.set_xticklabels(NF, fontsize=8)
ax.set_xlabel("NFE"); ax.set_ylabel("action L2 error", color="#d62728")
ax.tick_params(axis="y", labelcolor="#d62728")
ax2 = ax.twinx()
ctrl = {1: 0.8056, 2: 0.8681, 4: 0.8889, 8: 0.8993, 16: 0.8854}
ax2.plot(list(ctrl), list(ctrl.values()), marker="^", color="#2ca02c", ls="--",
         label="control success (seed 42, n=288)")
ax2.set_ylabel("closed-loop success", color="#2ca02c")
ax2.tick_params(axis="y", labelcolor="#2ca02c")
ax.set_title("Offline action error vs measured control success\nthey do NOT align: action best at 2, control best at 8")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig("experiments/figures/arm_neutral_nfe_curve.png", dpi=160)
print("figure written")
