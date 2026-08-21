"""Aggregate the paired Isaac Gym NFE study across evaluation replicates."""

import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import beta, binomtest

RESULTS = "experiments/isaacgym_control/nfe_study"
ORDER = ["flow_nfe1", "flow_nfe2", "flow_nfe4", "flow_nfe8", "flow_nfe16", "gaussian_nfe100"]
REFERENCE = "gaussian_nfe100"


def load():
    runs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(RESULTS, "r*_*.json"))):
        with open(path) as handle:
            payload = json.load(handle)
        summary = payload["summary"]
        runs[summary["replicate"]][summary["label"]] = payload
    return runs


def clopper_pearson(successes, n):
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(0.975, successes + 1, n - successes))
    return lower, upper


def main():
    runs = load()
    replicates = sorted(runs)
    if not replicates:
        raise SystemExit("no results found")

    # --- integrity: within a replicate, every arm must share one episode set ---
    print("=== EPISODE SET INTEGRITY ===")
    for r in replicates:
        hashes = {lab: p["summary"]["episode_set_sha256"] for lab, p in runs[r].items()}
        unique = set(hashes.values())
        status = "OK" if len(unique) == 1 else "MISMATCH"
        print(f"  replicate {r}: {len(hashes)} arms, {len(unique)} distinct hash -> {status}"
              f"  [{list(unique)[0][:16]}]")
        if len(unique) != 1:
            raise SystemExit(f"replicate {r} arms do not share one episode set: {hashes}")
    across = {runs[r][ORDER[0]]["summary"]["episode_set_sha256"] for r in replicates
              if ORDER[0] in runs[r]}
    print(f"  across replicates: {len(across)} distinct sets "
          f"({'independent' if len(across) == len(replicates) else 'NOT independent'})")

    # --- verified model-call counts ---
    # Latency is per planner call, and one call plans for all 16 parallel envs,
    # so the per-episode figure is that divided by the env count.
    print("\n=== MODEL CALLS PER PLAN (forward-hook counted) ===")
    print(f"{'arm':18s} {'requested':>10s} {'measured':>10s} {'ms/batch16':>12s} {'ms/episode':>12s}")
    for lab in ORDER:
        rows = [runs[r][lab]["summary"] for r in replicates if lab in runs[r]]
        if not rows:
            continue
        req = rows[0]["requested_nfe"]
        meas = np.mean([x["measured_calls_per_plan"] for x in rows])
        lat = np.mean([x["latency_mean_ms"] for x in rows])
        flag = "" if abs(meas - req) < 1e-6 else "  << MISMATCH"
        print(f"{lab:18s} {req:10d} {meas:10.2f} {lat:12.3f} {lat/16.0:12.3f}{flag}")

    # --- per-set results ---
    print("\n=== PER-SET SUCCESS ===")
    header = f"{'arm':18s}" + "".join(f"{'r'+str(r):>12s}" for r in replicates) + f"{'pooled':>14s}"
    print(header)
    pooled = {}
    for lab in ORDER:
        cells, succ, tot = [], 0, 0
        for r in replicates:
            if lab in runs[r]:
                s = runs[r][lab]["summary"]
                cells.append(f"{s['successes']:>4d}/{s['episodes']:<3d}")
                succ += s["successes"]
                tot += s["episodes"]
            else:
                cells.append("    -   ")
        lo, hi = clopper_pearson(succ, tot)
        pooled[lab] = (succ, tot, succ / tot, lo, hi)
        print(f"{lab:18s}" + "".join(f"{c:>12s}" for c in cells)
              + f"{succ/tot:>9.4f} n={tot}")

    # --- aggregate metrics ---
    print("\n=== AGGREGATE METRICS (mean over replicates) ===")
    metrics = ["success_rate", "goal_success_frac", "avg_obj_dist", "cubes_placed",
               "contact_rate", "cubes_farther", "latency_mean_ms"]
    print(f"{'arm':18s}" + "".join(f"{m[:11]:>13s}" for m in metrics))
    for lab in ORDER:
        rows = [runs[r][lab]["summary"] for r in replicates if lab in runs[r]]
        if not rows:
            continue
        vals = [np.mean([x[m] for x in rows]) for m in metrics]
        print(f"{lab:18s}" + "".join(f"{v:13.4f}" for v in vals))

    # --- between-replicate spread: the evaluation noise floor ---
    print("\n=== BETWEEN-REPLICATE SPREAD (evaluation noise, not training seeds) ===")
    print(f"{'arm':18s} {'min':>8s} {'max':>8s} {'range':>8s} {'std':>8s}")
    for lab in ORDER:
        rates = [runs[r][lab]["summary"]["success_rate"] for r in replicates if lab in runs[r]]
        if len(rates) > 1:
            print(f"{lab:18s} {min(rates):8.4f} {max(rates):8.4f} "
                  f"{max(rates)-min(rates):8.4f} {np.std(rates):8.4f}")

    # --- paired differences vs the Gaussian reference ---
    print(f"\n=== PAIRED vs {REFERENCE} (episode-level, pooled over replicates) ===")
    print(f"{'arm':18s} {'delta':>9s} {'b':>5s} {'c':>5s} {'McNemar p':>11s}")
    ref_by_ep = {}
    for r in replicates:
        if REFERENCE in runs[r]:
            for e in runs[r][REFERENCE]["episodes"]:
                ref_by_ep[(r, e["episode"])] = e["success"]

    paired = {}
    for lab in ORDER:
        if lab == REFERENCE:
            continue
        xs, ys = [], []
        for r in replicates:
            if lab not in runs[r]:
                continue
            for e in runs[r][lab]["episodes"]:
                key = (r, e["episode"])
                if key in ref_by_ep:
                    xs.append(e["success"])
                    ys.append(ref_by_ep[key])
        if not xs:
            continue
        xs, ys = np.array(xs), np.array(ys)
        b = int(((xs == 1) & (ys == 0)).sum())   # arm wins
        c = int(((xs == 0) & (ys == 1)).sum())   # gaussian wins
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        delta = float(xs.mean() - ys.mean())
        paired[lab] = {"delta_vs_gaussian": delta, "b_arm_wins": b,
                       "c_gaussian_wins": c, "mcnemar_p": float(p), "n_paired": len(xs)}
        print(f"{lab:18s} {delta:+9.4f} {b:5d} {c:5d} {p:11.4f}")

    out = {
        "replicates": replicates,
        "pooled": {k: {"successes": v[0], "episodes": v[1], "success_rate": v[2],
                       "ci95": [v[3], v[4]]} for k, v in pooled.items()},
        "paired_vs_gaussian": paired,
        "per_set": {str(r): {lab: runs[r][lab]["summary"] for lab in runs[r]} for r in replicates},
        "note": ("Three episode sets are EVALUATION REPLICATES, not training seeds. "
                 "One trained Flow checkpoint and one Gaussian checkpoint are used "
                 "throughout; only the episodes and the solver-step count vary."),
    }
    path = "experiments/isaacgym_control/nfe_study_aggregate.json"
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
