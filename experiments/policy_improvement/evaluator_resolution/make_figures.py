"""Figures for the evaluator-resolution study.

All axes in percentage points (pp). Every panel states its n and the exact
checkpoint group. Where an axis is truncated it is labelled.
"""
import itertools
import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = "experiments/policy_improvement/evaluator_resolution"
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]
ANCHORS = ["canonical", "cleanA42"]
COL = {"canonical": "#4c72b0", "cleanA42": "#dd8452"}


def load():
    p = os.path.join(HERE, "ANCHOR_ANALYSIS.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fig1(A):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for a in ANCHORS:
        rows = A["anchors"][a]["realizations"]
        x = [r["R"] for r in rows]
        y = [100 * r["macro"] for r in rows]
        ax.plot(x, y, "-o", color=COL[a], ms=6, label="{} (n={})".format(a, len(y)))
        ax.axhline(np.mean(y), color=COL[a], ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("evaluator realization index (same frozen checkpoint each time)")
    ax.set_ylabel("macro success (pp)")
    ax.set_title("Fig 1. Same-checkpoint macro success across realizations\n"
                 "5 fixed scenario sets x 96 episodes; NFE=4; EMA weights",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.text(0.02, -0.19, "dotted line = anchor mean. Axis not truncated to zero "
            "(note).", transform=ax.transAxes, fontsize=7, color="0.35")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_macro_realizations.png"), dpi=180)
    plt.close(fig)
    print("fig1 written")


def fig2(A):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(SETS) + 1)
    w = 0.36
    labels = SETS + ["MACRO"]
    for i, a in enumerate(ANCHORS):
        sds = [100 * (A["anchors"][a]["per_set"][s].get("sd") or 0) for s in SETS]
        sds.append(100 * (A["anchors"][a]["macro"].get("sd") or 0))
        ax.bar(x + (i - 0.5) * w, sds, w, color=COL[a], label=a)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("repeat SD across realizations (pp)")
    ax.set_xlabel("evaluation set (n=96 each); MACRO = mean of the five sets")
    ax.set_title("Fig 2. Per-set vs macro repeatability\n"
                 "same checkpoint, {} realizations per anchor".format(
                     A["n_realizations"][ANCHORS[0]]), fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_per_set_repeatability.png"), dpi=180)
    plt.close(fig)
    print("fig2 written")


def fig3(A):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    allv = []
    for a in ANCHORS:
        d = [100 * abs(v) for v in A["null_differences"][a]]
        allv.append(d)
    ax.hist(allv, bins=np.arange(0, max(max(v) for v in allv) + 0.6, 0.5),
            stacked=True, color=[COL[a] for a in ANCHORS], label=ANCHORS,
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("|difference| between two evaluations of the SAME checkpoint (pp)")
    ax.set_ylabel("count of pairs")
    ax.set_title("Fig 3. Same-policy pairwise repeat differences\n"
                 "pairs within an anchor share observations and are NOT "
                 "independent", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_null_differences.png"), dpi=180)
    plt.close(fig)
    print("fig3 written")


def fig4(A):
    if not A.get("pseudo_treatment"):
        print("fig4 skipped (no pseudo-treatment data)")
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for a in ANCHORS:
        eff = A["pseudo_treatment"].get(a)
        if not eff:
            continue
        ax.hist([100 * e for e in eff], bins=15, alpha=0.6, color=COL[a],
                label="{} (SD {:.2f} pp)".format(a, 100 * st.stdev(eff)),
                edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("apparent 'treatment effect' from a 3-vs-3 split (pp)")
    ax.set_ylabel("count of splits")
    ax.set_title("Fig 4. Pseudo-treatment null: splitting one checkpoint's own\n"
                 "realizations 3-vs-3, where the TRUE effect is exactly zero",
                 fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_pseudo_treatment.png"), dpi=180)
    plt.close(fig)
    print("fig4 written")


def fig5(A):
    v = A["uncertainty_vs_R"]
    Rs = sorted(int(k) for k in v)
    half = [100 * v[str(R)]["half_width"] if str(R) in v
            else 100 * v[R]["half_width"] for R in Rs]
    se = [100 * (v[str(R)]["se_delta"] if str(R) in v else v[R]["se_delta"])
          for R in Rs]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(Rs, half, "-o", color="#4c72b0", label="95% CI half-width on B−A")
    ax.plot(Rs, se, "-s", color="#55a868", label="SE of B−A")
    for target, style in ((1.0, "--"), (2.0, ":")):
        ax.axhline(target, color="0.5", ls=style, lw=1)
        ax.text(Rs[-1], target + 0.05, "{:.0f} pp".format(target),
                fontsize=7, color="0.4", ha="right")
    ax.set_xlabel("independent evaluator realizations per policy (R)")
    ax.set_ylabel("uncertainty on the A/B difference (pp)")
    ax.set_title("Fig 5. Expected B−A uncertainty vs repeats\n"
                 "conservative s_w = {:.2f} pp".format(
                     100 * A["s_w"]["conservative"]), fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_uncertainty_vs_R.png"), dpi=180)
    plt.close(fig)
    print("fig5 written")


def fig6(A):
    v = A["uncertainty_vs_R"]
    Rs = sorted(int(k) for k in v)
    cost = [(v[str(R)] if str(R) in v else v[R])["gpu_h_per_ab"] for R in Rs]
    half = [100 * (v[str(R)] if str(R) in v else v[R])["half_width"] for R in Rs]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(cost, half, "-o", color="#c44e52")
    for R, c, h in zip(Rs, cost, half):
        ax.annotate("R={}".format(R), (c, h), textcoords="offset points",
                    xytext=(6, 5), fontsize=7)
    ax.set_xlabel("GPU-hours for one A/B comparison (both policies)")
    ax.set_ylabel("95% CI half-width on B−A (pp)")
    ax.set_title("Fig 6. Cost vs precision for a two-policy comparison\n"
                 "measured evaluation cost {:.4f} GPU-h per realization".format(
                     (v[str(1)] if str(1) in v else v[1])["gpu_h_per_ab"] / 2),
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_cost_vs_precision.png"), dpi=180)
    plt.close(fig)
    print("fig6 written")


if __name__ == "__main__":
    A = load()
    if A is None or not A.get("anchors"):
        print("ANCHOR_ANALYSIS.json not available yet")
    else:
        fig1(A); fig2(A); fig3(A); fig4(A); fig5(A); fig6(A)
        print("figures in", FIG)
