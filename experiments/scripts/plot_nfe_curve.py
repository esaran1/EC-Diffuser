"""Plot the paired Isaac Gym NFE curve with per-replicate points."""

import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

RESULTS = "experiments/isaacgym_control/nfe_study"
OUT = "experiments/figures/nfe_curve_isaacgym.png"
FLOW = [("flow_nfe1", 1), ("flow_nfe2", 2), ("flow_nfe4", 4),
        ("flow_nfe8", 8), ("flow_nfe16", 16)]
REFERENCE = "gaussian_nfe100"


def load():
    runs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(RESULTS, "r*_*.json"))):
        with open(path) as handle:
            payload = json.load(handle)
        s = payload["summary"]
        runs[s["label"]][s["replicate"]] = s
    return runs


def pooled_ci(rows):
    successes = sum(r["successes"] for r in rows)
    n = sum(r["episodes"] for r in rows)
    lo = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(beta.ppf(0.975, successes + 1, n - successes))
    return successes / n, lo, hi, n


def main():
    runs = load()
    have = [(lab, nfe) for lab, nfe in FLOW if lab in runs]
    if not have:
        raise SystemExit("no flow results yet")

    xs, means, los, his = [], [], [], []
    for lab, nfe in have:
        rate, lo, hi, _ = pooled_ci(list(runs[lab].values()))
        xs.append(nfe); means.append(rate); los.append(lo); his.append(hi)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- left: success vs NFE ----
    left.errorbar(xs, means, yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                  fmt="o-", color="#1f77b4", capsize=4, lw=2, ms=7,
                  label="Flow (pooled, 95% CI)", zorder=3)
    for lab, nfe in have:  # individual replicates
        for rep, s in sorted(runs[lab].items()):
            left.plot(nfe, s["success_rate"], "o", color="#1f77b4",
                      alpha=0.32, ms=5, zorder=2)

    if REFERENCE in runs:
        g, glo, ghi, gn = pooled_ci(list(runs[REFERENCE].values()))
        left.axhline(g, color="#d62728", ls="--", lw=2,
                     label=f"Gaussian @100 NFE = {g:.3f}", zorder=1)
        left.axhspan(glo, ghi, color="#d62728", alpha=0.12, zorder=0)

    left.set_xscale("log", base=2)
    left.set_xticks(xs); left.set_xticklabels([str(x) for x in xs])
    left.set_xlabel("Flow NFE  (verified model calls per plan)")
    left.set_ylabel("success rate")
    left.set_title("Isaac Gym 3-cube: success vs inference cost")
    left.grid(alpha=0.3)
    left.legend(fontsize=8, loc="lower right")

    # ---- right: success vs measured latency ----
    lat = [np.mean([s["latency_mean_ms"] for s in runs[lab].values()]) / 16.0
           for lab, _ in have]
    right.errorbar(lat, means, yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                   fmt="o-", color="#1f77b4", capsize=4, lw=2, ms=7, label="Flow")
    for (lab, nfe), x in zip(have, lat):
        right.annotate(f"{nfe}", (x, np.mean([s['success_rate'] for s in runs[lab].values()])),
                       textcoords="offset points", xytext=(6, 6), fontsize=8)
    if REFERENCE in runs:
        glat = np.mean([s["latency_mean_ms"] for s in runs[REFERENCE].values()]) / 16.0
        right.errorbar([glat], [g], yerr=[[g - glo], [ghi - g]], fmt="s",
                       color="#d62728", capsize=4, ms=9, label="Gaussian @100")
        right.annotate("100", (glat, g), textcoords="offset points", xytext=(6, 6), fontsize=8)
    right.set_xscale("log")
    right.set_xlabel("planner latency per episode-step (ms, measured)")
    right.set_ylabel("success rate")
    right.set_title("Success vs measured inference latency")
    right.grid(alpha=0.3)
    right.legend(fontsize=8, loc="lower right")

    n_rep = len(runs[have[0][0]])
    plt.suptitle(
        f"Paired NFE study — {n_rep} evaluation replicates x 96 episodes, "
        "identical episodes within each replicate",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
