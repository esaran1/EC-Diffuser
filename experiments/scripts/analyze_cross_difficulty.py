"""Cross-difficulty scaling: 3 -> 4 -> 5 cubes for each method.

Combines the frozen 3-cube NFE study, the frozen 4-cube probe, and the 5-cube
probe. Nothing here rewrites any of those inputs.

Per-object success is treated as the primary curve. Full-task success becomes
mechanically stricter with object count (it demands N of N), so a drop there
conflates the criterion change with genuine loss of competence; per-object
success does not.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NFE_STUDY = "experiments/isaacgym_control/nfe_study"
PROBES = "experiments/isaacgym_control/fourcube"
OUT_FIG = "experiments/figures"

ARMS = [
    ("gaussian_nfe100", "Gaussian @100", "#d62728", "s"),
    ("flow_nfe4", "Flow @4", "#2ca02c", "o"),
    ("flow_nfe1", "Flow @1", "#1f77b4", "^"),
]


def three_cube():
    """Pooled over the three evaluation replicates of the frozen NFE study."""
    out = {}
    for label, *_ in ARMS:
        succ = total = 0
        placed = []
        for path in glob.glob(os.path.join(NFE_STUDY, f"r*_{label}.json")):
            with open(path) as handle:
                payload = json.load(handle)
            s = payload["summary"]
            succ += s["successes"]
            total += s["episodes"]
            placed.extend(e["cubes_placed"] for e in payload["episodes"])
        if total:
            out[label] = {
                "n_cubes": 3,
                "success_rate": succ / total,
                "per_object_success": float(np.mean(placed)) / 3.0,
                "cubes_placed": float(np.mean(placed)),
                "episodes": total,
            }
    return out


def probe(num_cubes):
    out = {}
    for label, *_ in ARMS:
        path = os.path.join(PROBES, f"{num_cubes}cube_{label}.json")
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            s = json.load(handle)["summary"]
        out[label] = {
            "n_cubes": num_cubes,
            "success_rate": s["success_rate"],
            "per_object_success": s["per_object_success"],
            "cubes_placed": s["cubes_placed"],
            "episodes": s["episodes"],
            "avg_obj_dist": s["avg_obj_dist"],
            "contact_rate": s["contact_rate"],
        }
    return out


def main():
    data = {3: three_cube(), 4: probe(4), 5: probe(5)}
    levels = [n for n in (3, 4, 5) if data[n]]

    print("=== 3 -> 4 -> 5 SCALING ===")
    print(f"{'arm':16s} {'metric':20s}" + "".join(f"{str(n)+' cubes':>12s}" for n in levels))
    for label, name, *_ in ARMS:
        for metric in ("success_rate", "per_object_success", "cubes_placed"):
            row = [data[n].get(label, {}).get(metric) for n in levels]
            cells = "".join(f"{v:12.4f}" if v is not None else f"{'-':>12s}" for v in row)
            print(f"{name:16s} {metric:20s}{cells}")
        print()

    # NFE penalty as a function of difficulty: the interaction the probe targets.
    print("=== NFE PENALTY vs DIFFICULTY (per-object success) ===")
    print(f"{'contrast':28s}" + "".join(f"{str(n)+' cubes':>12s}" for n in levels))
    for a, b, tag in (("flow_nfe4", "flow_nfe1", "Flow@4 - Flow@1"),
                      ("flow_nfe4", "gaussian_nfe100", "Flow@4 - Gaussian"),
                      ("gaussian_nfe100", "flow_nfe1", "Gaussian - Flow@1")):
        row = []
        for n in levels:
            x = data[n].get(a, {}).get("per_object_success")
            y = data[n].get(b, {}).get("per_object_success")
            row.append(None if x is None or y is None else x - y)
        cells = "".join(f"{v:+12.4f}" if v is not None else f"{'-':>12s}" for v in row)
        print(f"{tag:28s}{cells}")

    # ------------------------------- figures -------------------------------
    os.makedirs(OUT_FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for metric, ax, title, ylab in (
        ("success_rate", axes[0], "Full-task success vs object count",
         "full success (all N placed)"),
        ("per_object_success", axes[1], "Per-object success vs object count",
         "per-object success"),
    ):
        for label, name, colour, marker in ARMS:
            xs = [n for n in levels if label in data[n]]
            ys = [data[n][label][metric] for n in xs]
            ax.plot(xs, ys, marker + "-", color=colour, label=name, lw=2, ms=8)
        ax.set_xticks(levels)
        ax.set_xlabel("number of cubes")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].annotate("stricter criterion:\nrequires N of N",
                     xy=(0.05, 0.08), xycoords="axes fraction", fontsize=8,
                     bbox=dict(boxstyle="round", fc="#fff3cd", ec="#856404"))
    axes[1].annotate("primary curve:\ncriterion-independent",
                     xy=(0.05, 0.08), xycoords="axes fraction", fontsize=8,
                     bbox=dict(boxstyle="round", fc="#d4edda", ec="#155724"))

    # Panel 3: performance vs NFE at each difficulty.
    ax = axes[2]
    nfe_of = {"flow_nfe1": 1, "flow_nfe4": 4, "gaussian_nfe100": 100}
    shades = {3: "#a6cee3", 4: "#1f78b4", 5: "#08306b"}
    for n in levels:
        pts = sorted((nfe_of[l], data[n][l]["per_object_success"])
                     for l, *_ in ARMS if l in data[n])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                color=shades.get(n, "#333"), lw=2, ms=7, label=f"{n} cubes")
    ax.set_xscale("log")
    ax.set_xticks([1, 4, 100])
    ax.set_xticklabels(["1", "4", "100"])
    ax.set_xlabel("network calls per plan (log)")
    ax.set_ylabel("per-object success")
    ax.set_title("Performance vs NFE at each difficulty", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle(
        "Zero-shot policy generalization 3 -> 5 cubes "
        "(policy trained on 3 cubes only; DLP encoder saw up to 6)",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(OUT_FIG, "cross_difficulty_scaling.png")
    plt.savefig(path, dpi=150)
    print("\nwrote", path)

    with open(os.path.join(PROBES, "cross_difficulty.json"), "w") as handle:
        json.dump({str(k): v for k, v in data.items()}, handle, indent=2)
    print("wrote", os.path.join(PROBES, "cross_difficulty.json"))


if __name__ == "__main__":
    main()
