"""Figures for the tal_followup program (section 58).

Reads only committed JSON artifacts. Regenerates deterministically.
Repository .gitignore excludes *.png, so the script and its source data are
committed and the images regenerate in one command.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TF = os.path.dirname(HERE)
UWM = os.path.join(TF, "uwm")

C1, C2 = "#2a78d6", "#eb6834"      # validated categorical slots 1-2
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e3e3e0"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.labelsize": 12,
})


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fig_field_heatmaps():
    """FIGURE 2: 2D t_state x t_action diagnostic heatmaps, both arms."""
    du = load(os.path.join(UWM, "field_uwm.json"))
    dc = load(os.path.join(UWM, "field_ctl.json"))
    if not du or not dc:
        print("fig2 skipped (field json missing)")
        return
    grid = du["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    mats = []
    for d in (dc, du):
        mats.append(np.array([[d["cells"]["{:.2f}_{:.2f}".format(ts, ta)]["a_err"]
                               for ta in grid] for ts in grid]))
    vmin = min(m.min() for m in mats)
    vmax = max(m.max() for m in mats)
    for ax, M, title in zip(axes, mats,
                            ["matched diagonal control (p_diag=1.0)\nnever saw an off-diagonal time",
                             "faithful UWM (p_diag=0.0)\ntrained on independent times"]):
        im = ax.imshow(M, origin="lower", cmap="magma_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels(["{:.2f}".format(t) for t in grid])
        ax.set_yticks(range(len(grid)))
        ax.set_yticklabels(["{:.2f}".format(t) for t in grid])
        ax.set_xlabel("t_action   (0 = noise, 1 = clean)")
        ax.set_ylabel("t_state")
        ax.set_title(title, fontsize=11, color=INK)
        for i in range(len(grid)):
            for j in range(len(grid)):
                ax.text(j, i, "{:.3f}".format(M[i, j]), ha="center", va="center",
                        fontsize=7.5,
                        color="white" if M[i, j] > (vmin + vmax) / 2 else INK)
        # mark the diagonal
        ax.plot(range(len(grid)), range(len(grid)), color=C1, lw=1.5, ls=":",
                alpha=0.8)
    fig.colorbar(im, ax=axes, shrink=0.85, label="action velocity error")
    fig.suptitle("Fig 2. Action-prediction error across the 2D time field "
                 "(shared color scale; dotted line = diagonal)",
                 fontsize=12.5, y=1.02)
    p = os.path.join(HERE, "fig2_field_heatmap.png")
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(p))


def fig_paths():
    """FIGURE 3: the registered inference paths through the 2D time square."""
    sys.path.insert(0, UWM)
    from two_time_sampler import PATHS, schedule_points
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    styles = {"P0_diagonal": ("-", INK, 2.4),
              "P1_state_leads": ("-", C1, 2.0),
              "P2_action_leads": ("-", C2, 2.0),
              "P3_state_first": ("--", C1, 1.8),
              "P4_action_first": ("--", C2, 1.8)}
    for name in PATHS:
        pts = schedule_points(name, 64)
        ls, col, lw = styles[name]
        ax.plot([p[1] for p in pts], [p[0] for p in pts], ls, color=col, lw=lw,
                label=name.replace("_", " "))
    ax.scatter([0, 1], [0, 1], s=70, color=INK, zorder=5)
    ax.text(0.02, 0.06, "both noisy", fontsize=10, color=INK2)
    ax.text(0.98, 0.94, "both clean", fontsize=10, color=INK2, ha="right")
    ax.set_xlabel("t_action")
    ax.set_ylabel("t_state")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    despine(ax)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_title("Fig 3. Registered inference paths\n"
                 "all start (0,0) and end (1,1); above the diagonal = state leads",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    p = os.path.join(HERE, "fig3_inference_paths.png")
    fig.savefig(p, dpi=190)
    plt.close(fig)
    print("wrote", os.path.basename(p))


def fig_training():
    """FIGURE 1: training trajectories, with the comparability caveat on-figure."""
    import re

    def series(p):
        out = {}
        if not os.path.exists(p):
            return out
        for line in open(p):
            m = re.match(r"^(\d+):\s+([0-9.]+).*?action_loss:\s+([0-9.]+)", line)
            if m:
                out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        return out

    u = series("linux_logs/uwm_uwm.txt")
    c = series("linux_logs/uwm_ctl.txt")
    if not u or not c:
        print("fig1 skipped (training logs missing)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for ax, idx, lab in ((axes[0], 0, "flow loss"), (axes[1], 1, "action loss")):
        ax.plot(sorted(u), [u[k][idx] for k in sorted(u)], color=C1, lw=1.8,
                label="UWM (p_diag=0)")
        ax.plot(sorted(c), [c[k][idx] for k in sorted(c)], color=C2, lw=1.8,
                label="control (p_diag=1)")
        ax.set_xlabel("training step")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
        despine(ax)
        ax.legend(fontsize=9, frameon=False)
    axes[0].set_title("flow loss", fontsize=11)
    axes[1].set_title("action loss", fontsize=11)
    fig.suptitle("Fig 1. Training trajectories — NOT comparable as quality:\n"
                 "each arm computes its loss over its OWN time distribution",
                 fontsize=11.5, y=1.04)
    fig.text(0.5, -0.04, "At step 0 both arms hold byte-identical canonical "
             "weights; the UWM arm's higher loss is the off-diagonal "
             "out-of-distribution gap.", ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    p = os.path.join(HERE, "fig1_training.png")
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(p))


def fig_adaptation():
    """FIGURE 4: adaptation over training -- the late-adaptation finding."""
    steps, ratios, corrs = [], [], []
    for tag, st in (("field_uwm_500.json", 500), ("field_uwm_1000.json", 1000),
                    ("field_uwm_2000.json", 2000), ("field_uwm.json", 5000)):
        d = load(os.path.join(UWM, tag))
        if not d:
            continue
        grid, c = d["grid"], d["cells"]
        M = np.array([[c["{:.2f}_{:.2f}".format(a, b)]["a_err"] for b in grid]
                      for a in grid])
        dia = np.array([c["{:.2f}_{:.2f}".format(t, t)]["a_err"] for t in grid])
        off = M[~np.eye(len(grid), dtype=bool)]
        dd, vv = [], []
        for a in grid:
            for b in grid:
                if a == b:
                    continue
                dd.append(abs(a - b))
                vv.append(c["{:.2f}_{:.2f}".format(a, b)]["a_err"])
        steps.append(st)
        ratios.append(off.mean() / dia.mean())
        corrs.append(float(np.corrcoef(dd, vv)[0, 1]))
    if len(steps) < 2:
        print("fig4 skipped (need intermediate field json)")
        return
    dc = load(os.path.join(UWM, "field_ctl.json"))
    ctl_ratio = ctl_corr = None
    if dc:
        grid, c = dc["grid"], dc["cells"]
        M = np.array([[c["{:.2f}_{:.2f}".format(a, b)]["a_err"] for b in grid]
                      for a in grid])
        dia = np.array([c["{:.2f}_{:.2f}".format(t, t)]["a_err"] for t in grid])
        ctl_ratio = M[~np.eye(len(grid), dtype=bool)].mean() / dia.mean()
        dd, vv = [], []
        for a in grid:
            for b in grid:
                if a == b:
                    continue
                dd.append(abs(a - b))
                vv.append(c["{:.2f}_{:.2f}".format(a, b)]["a_err"])
        ctl_corr = float(np.corrcoef(dd, vv)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, ys, ctl, lab in ((axes[0], ratios, ctl_ratio, "off-diagonal / diagonal error"),
                             (axes[1], corrs, ctl_corr, "corr(|t_s - t_a|, error)")):
        ax.plot(steps, ys, "-o", color=C1, lw=2, ms=7, label="UWM (p_diag=0)")
        if ctl is not None:
            ax.axhline(ctl, color=C2, ls="--", lw=1.8,
                       label="control (never saw off-diagonal)")
        ax.set_xlabel("training step")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
        despine(ax)
        ax.legend(fontsize=9, frameon=False)
    axes[1].annotate("indistinguishable from\nthe control through 2000",
                     xy=(2000, corrs[2] if len(corrs) > 2 else corrs[-1]),
                     xytext=(2300, 0.55), fontsize=9, color=MUTED,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    fig.suptitle("Fig 4. Adaptation to off-diagonal times happens LATE\n"
                 "stopping at 2000 steps would have concluded it does not work",
                 fontsize=12, y=1.05)
    fig.tight_layout()
    p2 = os.path.join(HERE, "fig4_adaptation.png")
    fig.savefig(p2, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(p2))


if __name__ == "__main__":
    fig_training()
    fig_adaptation()
    fig_paths()
    fig_field_heatmaps()
    print("figures in", HERE)
