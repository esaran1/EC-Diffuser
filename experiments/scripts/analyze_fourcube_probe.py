"""Analyze the 4-cube probe: distributions, paired differences, regime call.

The paired analysis is what separates the two readings the probe exists to
distinguish:

  * ordinary degradation from needing one more cube -- per-object success holds
    up while full-success drops, because full success now requires 4 of 4;
  * genuine compositional failure -- per-object success itself drops, meaning
    the policy is worse at each cube, not merely asked for more of them.

Cube outcomes within an episode are correlated (one arm, one scene, shared
contacts), so no independence-based expectation is computed. The p^3 -> p^4
intuition is deliberately NOT used as a statistical null.
"""

import glob
import json
import os

import numpy as np
from scipy.stats import beta, binomtest, wilcoxon

RESULTS = "experiments/isaacgym_control/fourcube"
ORDER = ["gaussian_nfe100", "flow_nfe4", "flow_nfe1"]
REFERENCE = "gaussian_nfe100"
THREE_CUBE = "experiments/isaacgym_control/nfe_study"


def load(num_cubes=4):
    runs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, f"{num_cubes}cube_*.json"))):
        with open(path) as handle:
            payload = json.load(handle)
        runs[payload["summary"]["label"]] = payload
    return runs


def three_cube_reference():
    """Pooled 3-cube success for the same arms, for the 3->4 comparison."""
    out = {}
    for label in ORDER:
        successes = total = 0
        for path in glob.glob(os.path.join(THREE_CUBE, f"r*_{label}.json")):
            with open(path) as handle:
                s = json.load(handle)["summary"]
            successes += s["successes"]
            total += s["episodes"]
        if total:
            out[label] = successes / total
    return out


def main():
    runs = load()
    if not runs:
        raise SystemExit("no 4-cube results yet")

    n_cubes = runs[ORDER[0]]["summary"]["num_cubes"]

    print("=== INTEGRITY ===")
    hashes = {lab: p["summary"]["episode_set_sha256"] for lab, p in runs.items()}
    unique = set(hashes.values())
    print(f"  arms: {len(hashes)}   distinct episode-set hashes: {len(unique)} "
          f"-> {'OK' if len(unique) == 1 else 'MISMATCH'}  [{list(unique)[0][:16]}]")
    if len(unique) != 1:
        raise SystemExit(f"arms do not share one episode set: {hashes}")
    for lab, p in runs.items():
        s = p["summary"]
        flag = "  << MISMATCH" if s.get("CALL_COUNT_MISMATCH") else ""
        print(f"  {lab:18s} cubes={s['num_cubes']} requested_nfe={s['requested_nfe']:3d} "
              f"measured={s['measured_calls_per_plan']:.2f}{flag}")

    print("\n=== HEADLINE ===")
    print(f"{'arm':18s} {'full succ':>10s} {'95% CI':>18s} {'per-object':>11s} "
          f"{'cubes placed':>13s} {'goal frac':>10s}")
    for lab in ORDER:
        if lab not in runs:
            continue
        s = runs[lab]["summary"]
        ci = f"[{s['success_ci95'][0]:.3f},{s['success_ci95'][1]:.3f}]"
        print(f"{lab:18s} {s['success_rate']:10.4f} {ci:>18s} "
              f"{s['per_object_success']:11.4f} {s['cubes_placed']:13.3f} "
              f"{s['goal_success_frac']:10.4f}")

    print(f"\n=== CUBES-COMPLETED DISTRIBUTION (of {n_cubes}) ===")
    keys = [f"{k}_of_{n_cubes}" for k in range(n_cubes + 1)]
    print(f"{'arm':18s}" + "".join(f"{k:>10s}" for k in keys))
    for lab in ORDER:
        if lab not in runs:
            continue
        d = runs[lab]["summary"]["cubes_completed_distribution"]
        print(f"{lab:18s}" + "".join(f"{d.get(k, 0):10d}" for k in keys))
    print(f"{'(fraction)':18s}")
    for lab in ORDER:
        if lab not in runs:
            continue
        d = runs[lab]["summary"]["cubes_completed_distribution_frac"]
        print(f"{lab:18s}" + "".join(f"{d.get(k, 0):10.3f}" for k in keys))

    print("\n=== CONTROL / CONTACT METRICS ===")
    cols = ["avg_obj_dist", "max_obj_dist", "contact_rate", "n_contacted",
            "cubes_farther", "cubes_moved", "mean_progress"]
    print(f"{'arm':18s}" + "".join(f"{c[:12]:>14s}" for c in cols))
    for lab in ORDER:
        if lab not in runs:
            continue
        s = runs[lab]["summary"]
        print(f"{lab:18s}" + "".join(f"{s[c]:14.4f}" for c in cols))

    print("\n=== COST ===")
    print(f"{'arm':18s} {'calls/plan':>11s} {'ms/batch16':>12s} {'ms/ep-step':>12s} {'wall s':>9s}")
    for lab in ORDER:
        if lab not in runs:
            continue
        s = runs[lab]["summary"]
        print(f"{lab:18s} {s['measured_calls_per_plan']:11.2f} "
              f"{s['latency_mean_ms']:12.2f} {s['latency_per_episode_step_ms']:12.3f} "
              f"{s['wall_seconds']:9.1f}")

    # ---------------- paired differences ----------------
    print(f"\n=== PAIRED vs {REFERENCE} (same episodes) ===")
    ref = {e["episode"]: e for e in runs[REFERENCE]["episodes"]}
    paired = {}
    print(f"{'arm':18s} {'d(full)':>9s} {'b':>4s} {'c':>4s} {'McNemar p':>11s} "
          f"{'d(per-obj)':>11s} {'Wilcoxon p':>11s}")
    for lab in ORDER:
        if lab == REFERENCE or lab not in runs:
            continue
        xs, ys, px, py = [], [], [], []
        for e in runs[lab]["episodes"]:
            r = ref.get(e["episode"])
            if r is None:
                continue
            xs.append(e["success"]); ys.append(r["success"])
            px.append(e["per_object_success"]); py.append(r["per_object_success"])
        xs, ys = np.array(xs), np.array(ys)
        px, py = np.array(px), np.array(py)
        b = int(((xs == 1) & (ys == 0)).sum())
        c = int(((xs == 0) & (ys == 1)).sum())
        mp = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        # Per-object success is a bounded per-episode mean over correlated cubes,
        # so a paired signed-rank test is the right tool, not a proportion test.
        diff = px - py
        wp = wilcoxon(px, py).pvalue if np.any(diff != 0) else 1.0
        paired[lab] = {
            "delta_full_success": float(xs.mean() - ys.mean()),
            "b_arm_wins": b, "c_reference_wins": c, "mcnemar_p": float(mp),
            "delta_per_object": float(px.mean() - py.mean()),
            "wilcoxon_p": float(wp), "n_paired": len(xs),
        }
        print(f"{lab:18s} {xs.mean()-ys.mean():+9.4f} {b:4d} {c:4d} {mp:11.4f} "
              f"{px.mean()-py.mean():+11.4f} {wp:11.4f}")

    # ------------- 3-cube -> 4-cube degradation -------------
    print("\n=== 3-CUBE -> 4-CUBE (full success) ===")
    three = three_cube_reference()
    print(f"{'arm':18s} {'3-cube':>9s} {'4-cube':>9s} {'drop':>9s}")
    degradation = {}
    for lab in ORDER:
        if lab in three and lab in runs:
            a, b4 = three[lab], runs[lab]["summary"]["success_rate"]
            degradation[lab] = {"three_cube": a, "four_cube": b4, "drop": a - b4}
            print(f"{lab:18s} {a:9.4f} {b4:9.4f} {a-b4:+9.4f}")
    print("\n  Note: full success now requires 4 of 4 rather than 3 of 3, so some")
    print("  drop is expected from the stricter criterion alone. Per-object success")
    print("  above is the metric that separates that from genuine degradation.")
    print("  No independence-based expectation is quoted: cube outcomes within an")
    print("  episode are correlated, so p^3 -> p^4 is intuition, not a null.")

    out = {
        "num_cubes": n_cubes,
        "episode_set_sha256": list(unique)[0],
        "summaries": {lab: runs[lab]["summary"] for lab in runs},
        "paired_vs_reference": paired,
        "three_to_four_cube": degradation,
    }
    path = os.path.join(RESULTS, "fourcube_analysis.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
