"""Figures for the terminal-mask mechanism report.

Every panel states its sample count and the exact checkpoint group it came from.
No truncated axis is used without an explicit note.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M = "experiments/policy_improvement/loss_finetune/mechanism"
FIG = os.path.join(M, "figures")
os.makedirs(FIG, exist_ok=True)
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]
SEEDS = [42, 43, 44, 45]

ORIG = {  # accepted four-seed result (macro success)
    42: {"A": 0.7750, "B": 0.8042},
    43: {"A": 0.7729, "B": 0.7729},
    44: {"A": 0.7500, "B": 0.7896},
    45: {"A": 0.7833, "B": 0.8042},
}


def _paired_panel(ax, data, title, note):
    for s in SEEDS:
        ax.plot([0, 1], [data[s]["A"], data[s]["B"]], "-o", ms=5,
                label="seed {}".format(s))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A\n(terminal supervised)", "B\n(terminal masked)"])
    ax.set_ylabel("closed-loop success rate\n(macro over 5 sets, n=96 each)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center", fontsize=7,
            color="0.35")


def fig1():
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    _paired_panel(ax, ORIG, "Fig 1. Paired A vs B, original evaluation",
                  "4 paired fine-tuning seeds; 1 simulator realization each.\n"
                  "Axis not truncated to zero (note).")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_paired_original.png"), dpi=180)
    plt.close(fig)
    print("fig1 written")


def fig2(avg):
    if not avg:
        print("fig2 skipped (no repeat evaluation yet)")
        return
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    _paired_panel(ax, avg, "Fig 2. Paired A vs B, physics-averaged (R=2)",
                  "4 paired seeds; mean of 2 independent simulator realizations.\n"
                  "Axis not truncated to zero (note).")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_paired_physics_averaged.png"), dpi=180)
    plt.close(fig)
    print("fig2 written")


def fig3(per_set):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    w = 0.2
    x = np.arange(len(SETS))
    for i, s in enumerate(SEEDS):
        ax.bar(x + (i - 1.5) * w, [100 * per_set[s][k] for k in SETS], w,
               label="seed {}".format(s))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SETS)
    ax.set_ylabel("B − A success (percentage points)")
    ax.set_xlabel("evaluation set (n=96 episodes each)")
    ax.set_title("Fig 3. Per-set treatment effect by fine-tuning seed", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_per_set_effects.png"), dpi=180)
    plt.close(fig)
    print("fig3 written")


def fig4():
    p = os.path.join(M, "BPRIME_PROBE_DIVERGENCE.json")
    if not os.path.exists(p):
        print("fig4 skipped (probe not complete)")
        return
    d = json.load(open(p))["steps"]
    steps = sorted(int(k) for k in d)
    bp = [d[str(s)]["abs_param_diff_Bprime_vs_B"] for s in steps]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(steps, bp, "-o", label="|B' − B|  (denominator only)")
    ba = [d[str(s)].get("abs_param_diff_B_vs_A") for s in steps]
    if any(v for v in ba):
        xs = [s for s, v in zip(steps, ba) if v]
        ys = [v for v in ba if v]
        ax.plot(xs, ys, "-s", label="|B − A|  (masking intervention)")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("L2 parameter distance\n(trainable params, buffers excluded)")
    ax.set_title("Fig 4. Denominator-only vs masking divergence", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_bprime_divergence.png"), dpi=180)
    plt.close(fig)
    print("fig4 written")


def fig5():
    files = {s: os.path.join("experiments/policy_improvement/loss_finetune/"
                             "replication/REPLAY_PT_seed{}.json".format(s))
             for s in (43, 44, 45)}
    files[42] = os.path.join("experiments/policy_improvement/loss_finetune/"
                             "rerun/REPLAY_PER_TIMESTEP.json")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    got = False
    for arm, style in (("A", "-o"), ("B", "--s")):
        curves = []
        for s, f in files.items():
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            if arm not in d:
                continue
            curves.append(d[arm]["EMA"]["per_timestep_action_error"])
        if curves:
            got = True
            m = np.mean(curves, axis=0)
            ax.plot(range(len(m)), m, style, label="arm {} (mean of {} seeds)".format(
                arm, len(curves)))
    if not got:
        print("fig5 skipped (no per-timestep data)")
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("horizon position t   (t=4 is the goal-row placeholder)")
    ax.set_ylabel("action L1 error (EMA, replay bank)")
    ax.set_title("Fig 5. Temporal action-error decomposition", fontsize=10)
    ax.axvspan(3.5, 4.5, color="0.9", zorder=0)
    ax.text(4.0, ax.get_ylim()[1] * 0.5, "constructed\nplaceholder",
            ha="center", fontsize=6, color="0.3")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_temporal_action_error.png"), dpi=180)
    plt.close(fig)
    print("fig5 written")


def fig6():
    p = os.path.join(M, "ACTION_DISTRIBUTION.json")
    if not os.path.exists(p):
        print("fig6 skipped (action distribution not run)")
        return
    d = json.load(open(p))
    seeds = sorted(int(k) for k in d)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, key, lab in ((axes[0], "var_trace", "executed-action variance\n(trace of sample cov.)"),
                         (axes[1], "bias", "executed-action bias\n(‖mean − target‖₂)")):
        A = [d[str(s)]["A"]["distribution"][key] for s in seeds]
        B = [d[str(s)]["B"]["distribution"][key] for s in seeds]
        x = np.arange(len(seeds))
        ax.bar(x - 0.18, A, 0.36, label="A")
        ax.bar(x + 0.18, B, 0.36, label="B")
        ax.set_xticks(x)
        ax.set_xticklabels(["seed {}".format(s) for s in seeds], fontsize=7)
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    n = d[str(seeds[0])]["A"]["n_conditions"]
    K = d[str(seeds[0])]["A"]["K"]
    fig.suptitle("Fig 6. Multi-noise executed-action distribution "
                 "({} conditions x K={} paired draws)".format(n, K), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_action_distribution.png"), dpi=180)
    plt.close(fig)
    print("fig6 written")


def fig7():
    p = os.path.join(M, "GRADIENT_INTERFERENCE_canonical.json")
    if not os.path.exists(p):
        print("fig7 skipped")
        return
    d = json.load(open(p))
    groups = ["full", "trunk", "action_head", "time_cond"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    means = [d["cosines"][g]["term_vs_exec"]["mean"] for g in groups]
    q25 = [d["cosines"][g]["term_vs_exec"]["q25"] for g in groups]
    q75 = [d["cosines"][g]["term_vs_exec"]["q75"] for g in groups]
    x = np.arange(len(groups))
    ax.bar(x, means, 0.55, yerr=[np.array(means) - np.array(q25),
                                 np.array(q75) - np.array(means)],
           capsize=4, color="#4c72b0")
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("cos(g_terminal, g_executed)")
    ax.set_title("Fig 7. Gradient alignment, terminal vs executed action\n"
                 "(canonical checkpoint, {} batches x {}; bars = mean, whiskers = IQR)"
                 .format(d["n_batches"], d["batch_size"]), fontsize=9)
    ax.set_ylim(0, max(q75) * 1.28)
    ax.text(0.5, 0.965, "positive = ALIGNED  ->  conflict hypothesis NOT supported",
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color="0.25")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_gradient_cosine.png"), dpi=180)
    plt.close(fig)
    print("fig7 written")


if __name__ == "__main__":
    per_set = {}
    P = "experiments/policy_improvement/phase2"
    tags = {42: ("lossft_A", "lossft_B"), 43: ("lossft_r1_A", "lossft_r1_B"),
            44: ("lossft_r2_A", "lossft_r2_B"), 45: ("lossft_r3_A", "lossft_r3_B")}
    for s, (ta, tb) in tags.items():
        A = json.load(open(os.path.join(P, ta + "_diagnostics.json")))
        B = json.load(open(os.path.join(P, tb + "_diagnostics.json")))
        per_set[s] = {k: B["control"][k]["success"] - A["control"][k]["success"]
                      for k in SETS}

    avg = None
    rep2 = {}
    ok = True
    for s in SEEDS:
        for arm in ("A", "B"):
            f = os.path.join(P, "lossft_rep2_{}{}_diagnostics.json".format(arm, s))
            if not os.path.exists(f):
                ok = False
            else:
                rep2.setdefault(s, {})[arm] = json.load(open(f))["control_mean"]
    if ok:
        avg = {s: {arm: 0.5 * (ORIG[s][arm] + rep2[s][arm]) for arm in ("A", "B")}
               for s in SEEDS}

    fig1(); fig2(avg); fig3(per_set); fig4(); fig5(); fig6(); fig7()
    print("figures in", FIG)
