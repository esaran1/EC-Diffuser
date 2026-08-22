"""Canonical audit figures. All values are ingested from canonical_results.csv;
no number is typed into this script.
"""
import csv, json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

CSV = "experiments/audit/canonical_results.csv"
OUT = "experiments/audit/figures"
ARMS = [("gaussian_nfe100", "Gaussian @100", "#d62728", "s"),
        ("flow_nfe4", "Flow @4", "#2ca02c", "o"),
        ("flow_nfe1", "Flow @1", "#1f77b4", "^")]
HORIZON = {3: 100, 4: 150, 5: 200}


def load():
    pooled = defaultdict(lambda: {"k": 0, "n": 0, "placed": [], "lat": []})
    for r in csv.DictReader(open(CSV)):
        nc = int(r["num_cubes"])
        lab = r["label"]
        p = pooled[(nc, lab)]
        p["k"] += int(r["successes"])
        p["n"] += int(r["episodes"])
        # per-object success weighted by episodes
        p["placed"].append((float(r["per_object_success"]), int(r["episodes"])))
        if r["latency_mean_ms"]:
            p["lat"].append(float(r["latency_mean_ms"]))
    out = {}
    for (nc, lab), p in pooled.items():
        w = sum(n for _, n in p["placed"])
        out[(nc, lab)] = {
            "k": p["k"], "n": p["n"], "rate": p["k"] / p["n"],
            "ci": (0.0 if p["k"] == 0 else beta.ppf(.025, p["k"], p["n"] - p["k"] + 1),
                   1.0 if p["k"] == p["n"] else beta.ppf(.975, p["k"] + 1, p["n"] - p["k"])),
            "per_object": sum(v * n for v, n in p["placed"]) / w,
            "latency": float(np.mean(p["lat"])) if p["lat"] else None,
        }
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    D = load()
    levels = sorted({nc for nc, _ in D})

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # Fig 1: per-object success vs object count (primary)
    ax = axes[0]
    for lab, name, c, m in ARMS:
        xs = [n for n in levels if (n, lab) in D]
        ys = [D[(n, lab)]["per_object"] for n in xs]
        ax.plot(xs, ys, m + "-", color=c, label=name, lw=2, ms=8)
    ax.set_xticks(levels); ax.set_xlabel("number of cubes")
    ax.set_ylabel("per-object success"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Fig 1: per-object success (primary)", fontsize=11)

    # Fig 2: full success vs object count, horizon annotated
    ax = axes[1]
    for lab, name, c, m in ARMS:
        xs = [n for n in levels if (n, lab) in D]
        ys = [D[(n, lab)]["rate"] for n in xs]
        lo = [D[(n, lab)]["rate"] - D[(n, lab)]["ci"][0] for n in xs]
        hi = [D[(n, lab)]["ci"][1] - D[(n, lab)]["rate"] for n in xs]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=m + "-", color=c, label=name,
                    lw=2, ms=8, capsize=3)
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{n}\n(H={HORIZON[n]})" for n in levels])
    ax.set_xlabel("number of cubes (episode horizon differs)")
    ax.set_ylabel("full success (all N placed)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Fig 2: full success — criterion AND horizon change", fontsize=11)
    ax.annotate("criterion is N-of-N;\nhorizon rises with N",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8,
                bbox=dict(boxstyle="round", fc="#fff3cd", ec="#856404"))

    # Fig 3: Flow@4 - Flow@1 per-object gap vs object count, bootstrap CIs
    ax = axes[2]
    gaps = json.load(open("experiments/audit/nfe_gap_by_cubes.json"))
    xs = sorted(int(k) for k in gaps)
    ys = [gaps[str(n)]["gap"] for n in xs]
    lo = [gaps[str(n)]["gap"] - gaps[str(n)]["ci"][0] for n in xs]
    hi = [gaps[str(n)]["ci"][1] - gaps[str(n)]["gap"] for n in xs]
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="D-", color="#6a3d9a", lw=2, ms=8, capsize=4)
    ax.axhline(0, ls="--", c="k", alpha=.5)
    ax.set_xticks(xs); ax.set_xlabel("number of cubes")
    ax.set_ylabel("Flow@4 − Flow@1 (per-object success)")
    ax.grid(alpha=.3)
    ax.set_title("Fig 3: low-NFE penalty vs object count", fontsize=11)
    if len(xs) > 1 and gaps[str(xs[0])]["ci"][1] > gaps[str(xs[-1])]["ci"][0]:
        ax.annotate("CIs overlap:\nincrease NOT established",
                    xy=(0.03, 0.80), xycoords="axes fraction", fontsize=8,
                    bbox=dict(boxstyle="round", fc="#f8d7da", ec="#721c24"))

    plt.suptitle("Audited results — one training seed per method; "
                 "zero-shot POLICY generalization only (DLP saw up to 6 cubes)",
                 fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUT, "audit_canonical.png")
    plt.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    main()
