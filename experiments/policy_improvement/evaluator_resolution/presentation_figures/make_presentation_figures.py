"""Presentation figures for the evaluator-resolution study.

Reads ONLY committed values from ANCHOR_ANALYSIS.json. Recomputes nothing
scientific: the SDs, pseudo-treatment effects and half-widths are all read back
from the committed JSON, and the two derived half-width curves use the formula
already documented in RESULT.md section 16 applied to the committed s_w values.

Zero GPU. Presentation styling only.
"""
import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
OUT = HERE

# Categorical slots 1 and 2 from the validated reference palette (light mode).
C_CANON = "#2a78d6"   # slot 1, blue
C_CLEAN = "#eb6834"   # slot 2, orange
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e3e3e0"
SURFACE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID,
    "axes.linewidth": 1.0,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.labelsize": 14,
})


def load():
    with open(os.path.join(SRC, "ANCHOR_ANALYSIS.json")) as fh:
        return json.load(fh)


def footnote(fig, text, y=0.045):
    """Figure-level footnote. Placed in figure coords with reserved bottom
    margin so it can never collide with the x-axis label or ticks."""
    fig.text(0.062, y, text, fontsize=12, color=MUTED, linespacing=1.55,
             va="bottom")


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def fig1(A):
    """Same-policy repeatability: six points per anchor + mean line."""
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    anchors = [("canonical", C_CANON, "canonical"),
               ("cleanA42", C_CLEAN, "clean-A42")]
    for i, (key, col, label) in enumerate(anchors):
        y = [100 * r["macro"] for r in A["anchors"][key]["realizations"]]
        # spread the six dots across the mean line's span so they read as a
        # distribution around it, not a column beside it
        x = [i + 0.085 * (j - 2.5) for j in range(len(y))]
        m = st.mean(y)
        # mean line first, so points sit on top of it
        ax.hlines(m, i - 0.34, i + 0.34, color=col, lw=2, zorder=2)
        ax.scatter(x, y, s=150, facecolor=col, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.text(i + 0.40, m, "mean\n{:.2f}%".format(m), va="center",
                ha="left", fontsize=12.5, color=INK2, linespacing=1.35)
        sd = st.stdev(y)
        ax.text(i, 82.35, label, ha="center", fontsize=16, color=INK)
        ax.text(i, 81.85, "SD = {:.2f} pp".format(sd), ha="center",
                fontsize=13.5, color=col)

    ax.set_xlim(-0.55, 1.92)
    ax.set_ylim(75.6, 83.0)
    ax.set_xticks([])
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylabel("macro success (%)")
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))

    ax.set_title("The same frozen checkpoint, evaluated six times",
                 fontsize=19, color=INK, pad=42, loc="left", x=-0.055)
    ax.text(-0.055, 1.055,
            "Nothing changed between repeats: identical checkpoint bytes, "
            "scenario sets, seeds and NFE.",
            transform=ax.transAxes, fontsize=12.5, color=INK2)
    fig.tight_layout(rect=(0.01, 0.105, 0.99, 0.99))
    footnote(fig, "pooled within-checkpoint SD = 1.40 pp        "
                  "each dot is one full 480-episode evaluation")
    p = os.path.join(OUT, "pres1_same_policy_repeatability.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print("wrote", os.path.basename(p))


def fig2(A):
    """Pseudo-treatment null: apparent B-A effects where truth is zero."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(9.6, 5.8))

    edges = np.arange(-3.0, 3.01, 0.75)
    centers = (edges[:-1] + edges[1:]) / 2
    w = (edges[1] - edges[0]) / 2 * 0.86     # side-by-side, 2px-style gap
    for k, (key, col, label) in enumerate((("canonical", C_CANON, "canonical"),
                                           ("cleanA42", C_CLEAN, "clean-A42"))):
        eff = [100 * e for e in A["pseudo_treatment"][key]]
        counts, _ = np.histogram(eff, bins=edges)
        ax.bar(centers + (k - 0.5) * w, counts, width=w * 0.92, color=col,
               label=label, zorder=2, linewidth=0)

    ax.axvline(0, color=INK, lw=2, zorder=4)
    top = max(ax.get_ylim()[1], 5.6)
    ax.set_ylim(0, top)
    ax.text(0.09, top * 0.97, "true effect = 0", fontsize=13.5, color=INK,
            va="top", ha="left", zorder=5)

    ax.annotate("apparent effects\nup to 2.64 pp",
                xy=(2.64, 1.05), xytext=(2.58, top * 0.60),
                fontsize=13, color=INK2, ha="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2,
                                connectionstyle="arc3,rad=0.22"))

    ax.set_xlim(-3.5, 3.9)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlabel("apparent B−A effect from a 3-vs-3 split (pp)")
    ax.set_ylabel("number of splits")
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left", "bottom"))
    ax.legend(frameon=False, fontsize=13.5, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), handlelength=1.1, handleheight=1.1)

    ax.set_title("Same exact checkpoint, split into a fake A and a fake B",
                 fontsize=19, color=INK, pad=42, loc="left", x=-0.055)
    ax.text(-0.055, 1.055,
            "Each anchor's own six evaluations split 3-vs-3, so the true "
            "difference is exactly zero.",
            transform=ax.transAxes, fontsize=12.5, color=INK2)

    fig.tight_layout(rect=(0.01, 0.155, 0.99, 0.99))
    footnote(fig, "All 10 unique splits per anchor, each shown with both sign "
                  "labelings — a complete\nenumeration of one six-run panel, "
                  "not independent statistical samples.")
    p = os.path.join(OUT, "pres2_pseudo_treatment_null.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print("wrote", os.path.basename(p))


def fig3(A):
    """Precision vs repeats, both variance estimates."""
    s_pool = 100 * A["s_w"]["pooled"]
    s_cons = 100 * A["s_w"]["conservative"]
    Rs = list(range(1, 7))
    hw = lambda s, R: 1.96 * s * math.sqrt(2.0 / R)
    pool = [hw(s_pool, R) for R in Rs]
    cons = [hw(s_cons, R) for R in Rs]

    fig, ax = plt.subplots(figsize=(9.9, 5.7))
    ax.plot(Rs, cons, "-", color=C_CLEAN, lw=2, zorder=3)
    ax.plot(Rs, pool, "-", color=C_CANON, lw=2, zorder=3)
    ax.scatter(Rs, cons, s=95, facecolor=C_CLEAN, edgecolor=SURFACE,
               linewidth=2, zorder=4)
    ax.scatter(Rs, pool, s=95, facecolor=C_CANON, edgecolor=SURFACE,
               linewidth=2, zorder=4)

    ax.text(6.16, cons[-1] + 0.20, "conservative SD  1.552 pp", fontsize=12.5,
            color=C_CLEAN, va="center", ha="left")
    ax.text(6.16, pool[-1] - 0.24, "pooled SD  1.399 pp", fontsize=12.5,
            color=C_CANON, va="center", ha="left")

    # Mark R=3.
    ax.axvline(3, color="#cfcfcb", lw=1.5, zorder=1)
    ax.scatter([3, 3], [cons[2], pool[2]], s=210, facecolor="none",
               edgecolor=[C_CLEAN, C_CANON], linewidth=2, zorder=5)
    ax.annotate("±{:.2f} pp".format(cons[2]), xy=(3, cons[2]),
                xytext=(2.62, cons[2] + 0.62), fontsize=14, color=C_CLEAN,
                va="center", ha="center")
    ax.annotate("±{:.2f} pp".format(pool[2]), xy=(3, pool[2]),
                xytext=(3.52, pool[2] - 0.52), fontsize=14, color=C_CANON,
                va="center", ha="center")
    ax.text(3, 0.60, "R = 3\nrecommended default", fontsize=13, color=INK2,
            ha="center", va="bottom", linespacing=1.4)

    ax.set_xlim(0.65, 8.5)
    ax.set_ylim(0.4, 5.1)
    ax.set_xticks(Rs)
    ax.set_xlabel("evaluator realizations per policy (R)")
    ax.set_ylabel("95% half-width on B−A (pp)")
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left", "bottom"))

    ax.set_title("How much repeating buys", fontsize=19, color=INK, pad=42,
                 loc="left", x=-0.055)
    ax.text(-0.055, 1.055,
            "Width of the 95% interval on a two-policy difference, "
            "under both variance estimates.",
            transform=ax.transAxes, fontsize=12.5, color=INK2)
    fig.tight_layout(rect=(0.01, 0.105, 0.99, 0.99))
    footnote(fig, "Planning uses the conservative (noisier-anchor) SD.        "
                  "half-width = 1.96 · s · √(2/R)")
    p = os.path.join(OUT, "pres3_precision_vs_repeats.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print("wrote", os.path.basename(p))


if __name__ == "__main__":
    A = load()
    fig1(A)
    fig2(A)
    fig3(A)
    print("presentation figures in", OUT)
