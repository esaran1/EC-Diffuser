"""Separate the effect of additional objects from the effect of execution time.

Two scaling views are built from raw episode records:

  native horizon : 3->H=100, 4->H=150, 5->H=200   (the benchmark's own budget)
  fixed horizon  : 3->H=100, 4->H=100, 5->H=100   (time budget held constant)

Pairing rules, applied strictly:

  * Within an object count, all arms share one hash-locked episode set, so
    method contrasts are episode-paired.
  * The fixed-H runs REPLAY the same frozen episode set as the native-H runs
    (same initial and goal states, verified by hash), so the horizon contrast
    is ALSO episode-paired. This is asserted, not assumed: if the hashes differ
    the comparison is downgraded to unpaired and no paired test is reported.
  * 4-cube and 5-cube episodes are never paired with each other.
"""

import json
import os
from collections import OrderedDict

import numpy as np
from scipy.stats import beta, binomtest, wilcoxon

NFE = "experiments/isaacgym_control/nfe_study"
PR = "experiments/isaacgym_control/fourcube"
ARMS = ["gaussian_nfe100", "flow_nfe4", "flow_nfe1"]
OUT = "experiments/audit/fixed_horizon_analysis.json"


def load(path):
    with open(path) as h:
        return json.load(h)


def episodes_of(path):
    return {e["episode"]: e for e in load(path)["episodes"]}


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))
    return lo, hi


def arm_stats(paths, ncubes):
    """Pool one arm over one or more raw files (3-cube has 3 replicates)."""
    eps = []
    for p in paths:
        eps.extend(load(p)["episodes"])
    n = len(eps)
    placed = np.array([e["cubes_placed"] for e in eps], dtype=float)
    succ = np.array([e["success"] for e in eps], dtype=float)
    k = int(succ.sum())
    lo, hi = cp(k, n)
    dist = OrderedDict(
        (f"{c}_of_{ncubes}", int((placed == c).sum())) for c in range(ncubes + 1))
    s0 = load(paths[0])["summary"]
    return {
        "episodes": n, "successes": k, "success_rate": k / n, "ci95": [lo, hi],
        "per_object_success": float((placed / ncubes).mean()),
        "cubes_placed": float(placed.mean()),
        "cubes_completed_distribution": dist,
        "avg_obj_dist": float(np.mean([e["avg_obj_dist"] for e in eps])),
        "max_obj_dist": float(np.mean([e["max_obj_dist"] for e in eps])) if "max_obj_dist" in eps[0] else None,
        "contact_rate": float(np.mean([e["n_contacted"] > 0 for e in eps])),
        "cubes_farther": float(np.mean([e["cubes_farther"] for e in eps])),
        "clip_fraction": float(np.mean([e["clip_fraction"] for e in eps])),
        "measured_calls_per_plan": s0.get("measured_calls_per_plan"),
        "latency_mean_ms": s0.get("latency_mean_ms"),
        "horizon": s0.get("horizon"),
        "episode_set_sha256": s0.get("episode_set_sha256"),
    }


def paired(a_paths, b_paths, ncubes, seed=0):
    """Episode-paired contrast. Returns None if the arms are not on one set."""
    ha = {load(p)["summary"].get("episode_set_sha256") for p in a_paths}
    hb = {load(p)["summary"].get("episode_set_sha256") for p in b_paths}
    if ha != hb or len(ha) != len(a_paths) and len(a_paths) == 1:
        pass  # 3-cube pools 3 sets; handled by zip below
    diffs, xs, ys = [], [], []
    for pa, pb in zip(a_paths, b_paths):
        if load(pa)["summary"].get("episode_set_sha256") != \
           load(pb)["summary"].get("episode_set_sha256"):
            return {"paired": False,
                    "reason": "episode-set hashes differ; paired tests suppressed"}
        A, B = episodes_of(pa), episodes_of(pb)
        for i in sorted(set(A) & set(B)):
            diffs.append(A[i]["cubes_placed"] / ncubes - B[i]["cubes_placed"] / ncubes)
            xs.append(A[i]["success"]); ys.append(B[i]["success"])
    d = np.array(diffs); xs = np.array(xs); ys = np.array(ys)
    b = int(((xs == 1) & (ys == 0)).sum()); c = int(((xs == 0) & (ys == 1)).sum())
    mp = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) else 1.0
    wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    rng = np.random.RandomState(seed)
    boot = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(20000)])
    return {
        "paired": True, "n": len(d),
        "delta_full_success": float(xs.mean() - ys.mean()),
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": mp,
        "delta_per_object": float(d.mean()), "wilcoxon_p": wp,
        "boot_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "_boot": boot,
    }


def main():
    files = {
        (3, "native"): {a: [f"{NFE}/r{r}_{a}.json" for r in (0, 1, 2)] for a in ARMS},
        (4, "native"): {a: [f"{PR}/4cube_{a}.json"] for a in ARMS},
        (5, "native"): {a: [f"{PR}/5cube_{a}.json"] for a in ARMS},
        (4, "fixed"): {a: [f"{PR}/4cube_H100_{a}.json"] for a in ARMS},
        (5, "fixed"): {a: [f"{PR}/5cube_H100_{a}.json"] for a in ARMS},
    }
    # 3 cubes: native H IS 100, so it serves both views.
    files[(3, "fixed")] = files[(3, "native")]

    have = {k: v for k, v in files.items()
            if all(os.path.exists(p) for ps in v.values() for p in ps)}
    missing = sorted(set(files) - set(have))
    if missing:
        print(f"MISSING (not yet run): {missing}\n")

    report = {"tables": {}, "method_contrasts": {}, "horizon_contrasts": {},
              "nfe_gap": {}, "missing": [list(m) for m in missing]}

    for (nc, view), spec in sorted(have.items()):
        print(f"=== {nc} CUBES, {view} horizon ===")
        print(f"{'arm':18s} {'H':>5s} {'calls':>7s} {'full':>16s} {'95% CI':>18s} "
              f"{'per-obj':>9s} {'dist':>8s} {'contact':>8s}")
        for a in ARMS:
            st = arm_stats(spec[a], nc)
            report["tables"][f"{nc}cube_{view}_{a}"] = st
            print(f"{a:18s} {str(st['horizon']):>5s} {st['measured_calls_per_plan']:7.1f} "
                  f"{st['successes']:3d}/{st['episodes']:<3d}={st['success_rate']:.4f} "
                  f"[{st['ci95'][0]:.3f},{st['ci95'][1]:.3f}] {st['per_object_success']:9.4f} "
                  f"{st['avg_obj_dist']:8.4f} {st['contact_rate']:8.4f}")
        print()

    # ---- method contrasts within each (cubes, view) ----
    print("=== PAIRED METHOD CONTRASTS ===")
    for (nc, view), spec in sorted(have.items()):
        for a, b in (("flow_nfe4", "flow_nfe1"), ("flow_nfe4", "gaussian_nfe100"),
                     ("flow_nfe1", "gaussian_nfe100")):
            r = paired(spec[a], spec[b], nc)
            if not r.get("paired"):
                continue
            key = f"{nc}cube_{view}__{a}_minus_{b}"
            report["method_contrasts"][key] = {k: v for k, v in r.items() if k != "_boot"}
            print(f"  {key}")
            print(f"    d_full={r['delta_full_success']:+.4f} McNemar_p={r['mcnemar_p']:.4f}  "
                  f"d_perobj={r['delta_per_object']:+.4f} Wilcoxon_p={r['wilcoxon_p']:.4f} "
                  f"boot95=[{r['boot_ci95'][0]:+.4f},{r['boot_ci95'][1]:+.4f}]")
            if a == "flow_nfe4" and b == "flow_nfe1":
                report["nfe_gap"].setdefault(view, {})[str(nc)] = {
                    "gap": r["delta_per_object"], "ci": r["boot_ci95"], "n": r["n"]}

    # ---- horizon contrasts: same states, different time budget ----
    print("\n=== HORIZON CONTRASTS (same frozen episode set, paired) ===")
    for nc, native_h in ((4, 150), (5, 200)):
        if (nc, "fixed") not in have or (nc, "native") not in have:
            continue
        for a in ARMS:
            r = paired(have[(nc, "native")][a], have[(nc, "fixed")][a], nc)
            key = f"{nc}cube__{a}__H{native_h}_minus_H100"
            if not r.get("paired"):
                print(f"  {key}: {r.get('reason')}")
                report["horizon_contrasts"][key] = r
                continue
            report["horizon_contrasts"][key] = {k: v for k, v in r.items() if k != "_boot"}
            print(f"  {key}")
            print(f"    d_full={r['delta_full_success']:+.4f} McNemar_p={r['mcnemar_p']:.4f}  "
                  f"d_perobj={r['delta_per_object']:+.4f} Wilcoxon_p={r['wilcoxon_p']:.4f} "
                  f"boot95=[{r['boot_ci95'][0]:+.4f},{r['boot_ci95'][1]:+.4f}]")

    # ---- does the NFE gap trend differ between views? ----
    print("\n=== Flow@4 - Flow@1 PER-OBJECT GAP: native vs fixed horizon ===")
    for view in ("native", "fixed"):
        if view not in report["nfe_gap"]:
            continue
        print(f"  {view} horizon:")
        for nc in sorted(report["nfe_gap"][view], key=int):
            g = report["nfe_gap"][view][nc]
            print(f"    {nc} cubes: {g['gap']:+.4f} [{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}] n={g['n']}")

    with open(OUT, "w") as h:
        json.dump(report, h, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
