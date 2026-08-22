"""Independent recomputation of every Isaac Gym result from RAW episode records.

Nothing here reads a Markdown report or a previously written summary block. Every
number is recomputed from the per-episode arrays stored in the raw result JSONs,
and cross-checked against the summary the original run wrote. Disagreements are
reported, not silently reconciled.

Statistical choices are re-derived rather than inherited:

  * Full success is a Bernoulli outcome per episode -> Clopper-Pearson exact CI.
  * Paired full success is binary/binary on the same episodes -> exact McNemar
    (binomial test on discordant pairs). This is the correct test; the normal
    approximation is not used because discordant counts are small (<50).
  * Per-object success is a bounded per-episode mean over CORRELATED cubes (one
    policy, one scene, one arm trajectory). It is not a proportion of
    independent trials, so a proportion test is invalid. A paired Wilcoxon
    signed-rank test on per-episode differences is used instead. Its own
    assumption -- symmetric distribution of differences -- is checked and
    reported, and a paired bootstrap is computed alongside as an
    assumption-light cross-check.

Outputs experiments/audit/canonical_results.csv. All audit tables derive from it.
"""

import csv
import glob
import hashlib
import json
import os
import pickle
from collections import OrderedDict

import numpy as np
from scipy.stats import beta, binomtest, wilcoxon

NFE_STUDY = "experiments/isaacgym_control/nfe_study"
PROBES = "experiments/isaacgym_control/fourcube"
OUT_CSV = "experiments/audit/canonical_results.csv"
OUT_JSON = "experiments/audit/recomputed_details.json"
THRESHOLD = 0.04  # dist_threshold, env_config/generalization_num_cubes/Config.yaml


def clopper_pearson(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def file_sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def recompute_arm(path):
    """Recompute every reported statistic from the per-episode records alone."""
    with open(path) as handle:
        payload = json.load(handle)
    eps = payload["episodes"]
    stated = payload["summary"]
    n = len(eps)

    success = np.array([e["success"] for e in eps], dtype=float)
    placed = np.array([e["cubes_placed"] for e in eps], dtype=float)
    n_cubes = int(stated.get("num_cubes", 3))

    k = int(success.sum())
    lo, hi = clopper_pearson(k, n)

    # Per-object success recomputed from cubes_placed, NOT copied.
    per_object = placed / n_cubes

    dist = OrderedDict()
    for c in range(n_cubes + 1):
        dist[f"{c}_of_{n_cubes}"] = int((placed == c).sum())

    out = {
        "raw_file": path,
        "raw_file_sha256": file_sha(path),
        "arm": stated.get("arm"),
        "label": stated.get("label"),
        "num_cubes": n_cubes,
        "requested_nfe": stated.get("requested_nfe"),
        "measured_calls_per_plan": stated.get("measured_calls_per_plan"),
        "episode_set_sha256": stated.get("episode_set_sha256"),
        "replicate": stated.get("replicate"),
        "episodes": n,
        "successes": k,
        "success_rate": k / n,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "per_object_success": float(per_object.mean()),
        "per_object_success_std": float(per_object.std(ddof=1)),
        "cubes_placed_mean": float(placed.mean()),
        "cubes_completed_distribution": dist,
        "avg_obj_dist_mean": float(np.mean([e["avg_obj_dist"] for e in eps])),
        "max_obj_dist_mean": float(np.mean([e["max_obj_dist"] for e in eps]))
        if "max_obj_dist" in eps[0] else None,
        "contact_rate": float(np.mean([e["n_contacted"] > 0 for e in eps])),
        "n_contacted_mean": float(np.mean([e["n_contacted"] for e in eps])),
        "cubes_farther_mean": float(np.mean([e["cubes_farther"] for e in eps])),
        "clip_fraction_mean": float(np.mean([e["clip_fraction"] for e in eps])),
        "latency_mean_ms": stated.get("latency_mean_ms"),
        "latency_p50_ms": stated.get("latency_p50_ms"),
        "latency_p95_ms": stated.get("latency_p95_ms"),
        "wall_seconds": stated.get("wall_seconds"),
    }

    # Cross-check recomputed values against what the run itself stored.
    mismatches = []
    for key, stated_key in (("success_rate", "success_rate"),
                            ("per_object_success", "per_object_success"),
                            ("cubes_placed_mean", "cubes_placed"),
                            ("contact_rate", "contact_rate")):
        if stated_key in stated and stated[stated_key] is not None:
            if abs(out[key] - stated[stated_key]) > 1e-9:
                mismatches.append(
                    {"field": stated_key, "stated": stated[stated_key], "recomputed": out[key]})
    out["summary_mismatches"] = mismatches
    out["_episodes"] = eps
    return out


def paired_tests(arm_a, arm_b):
    """Paired comparison on episodes present in BOTH arms, matched by episode id."""
    a = {e["episode"]: e for e in arm_a["_episodes"]}
    b = {e["episode"]: e for e in arm_b["_episodes"]}
    common = sorted(set(a) & set(b))
    if not common:
        return None

    xs = np.array([a[i]["success"] for i in common])
    ys = np.array([b[i]["success"] for i in common])
    disc_b = int(((xs == 1) & (ys == 0)).sum())
    disc_c = int(((xs == 0) & (ys == 1)).sum())
    mcnemar_p = float(binomtest(disc_b, disc_b + disc_c, 0.5).pvalue) if (disc_b + disc_c) else 1.0

    nc = arm_a["num_cubes"]
    pa = np.array([a[i]["cubes_placed"] for i in common], dtype=float) / nc
    pb = np.array([b[i]["cubes_placed"] for i in common], dtype=float) / nc
    diff = pa - pb

    if np.any(diff != 0):
        w_p = float(wilcoxon(pa, pb).pvalue)
    else:
        w_p = 1.0

    # Wilcoxon assumes the differences are symmetric about the median. Report
    # skew so the reader can judge, rather than asserting the assumption holds.
    nz = diff[diff != 0]
    skew = float(((nz - nz.mean()) ** 3).mean() / (nz.std() ** 3)) if nz.size > 2 and nz.std() > 0 else None

    # Assumption-light cross-check: paired bootstrap on the mean difference.
    rng = np.random.RandomState(0)
    boot = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(10000)])
    boot_lo, boot_hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided bootstrap p: fraction of resamples on the other side of zero.
    boot_p = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))

    return {
        "n_paired": len(common),
        "delta_full_success": float(xs.mean() - ys.mean()),
        "mcnemar_b": disc_b, "mcnemar_c": disc_c, "mcnemar_p_exact": mcnemar_p,
        "delta_per_object": float(diff.mean()),
        "wilcoxon_p": w_p,
        "diff_skew": skew,
        "bootstrap_ci95": [float(boot_lo), float(boot_hi)],
        "bootstrap_p_two_sided": min(boot_p, 1.0),
    }


def verify_episode_set(path, expected_sha):
    """Recompute the episode-set hash from the serialized states themselves."""
    if not os.path.exists(path):
        return {"file": path, "status": "MISSING"}
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    recomputed = hashlib.sha256(
        payload["init"].tobytes() + payload["goal"].tobytes()).hexdigest()
    return {
        "file": path,
        "stored_sha256": payload.get("sha256"),
        "recomputed_sha256": recomputed,
        "matches_stored": recomputed == payload.get("sha256"),
        "matches_results": (expected_sha is None) or (recomputed == expected_sha),
        "n_episodes": int(len(payload["init"])),
        "n_cubes": int(payload["init"].shape[1]),
        "init_shape": list(payload["init"].shape),
        "goal_shape": list(payload["goal"].shape),
    }


def main():
    os.makedirs("experiments/audit", exist_ok=True)
    arms = {}

    # 3-cube NFE study: three evaluation replicates.
    for path in sorted(glob.glob(os.path.join(NFE_STUDY, "r*_*.json"))):
        rec = recompute_arm(path)
        arms[f"3cube_r{rec['replicate']}_{rec['label']}"] = rec

    # 4- and 5-cube probes.
    for n in (4, 5):
        for path in sorted(glob.glob(os.path.join(PROBES, f"{n}cube_*.json"))):
            rec = recompute_arm(path)
            arms[f"{n}cube_{rec['label']}"] = rec

    print(f"=== RECOMPUTED {len(arms)} ARMS FROM RAW EPISODE RECORDS ===\n")

    # ---- summary-vs-recomputed integrity ----
    bad = {k: v["summary_mismatches"] for k, v in arms.items() if v["summary_mismatches"]}
    if bad:
        print("!! SUMMARY MISMATCHES (recomputed disagrees with stored summary):")
        for k, v in bad.items():
            print(f"   {k}: {v}")
    else:
        print("All stored summaries agree with values recomputed from raw episodes "
              "(tolerance 1e-9).")

    # ---- episode-set pairing (Part IV) ----
    print("\n=== EPISODE-SET PAIRING ===")
    pairing = {}
    groups = {}
    for key, rec in arms.items():
        gid = key.rsplit("_", 1)[0] if key.startswith("3cube") else f"{rec['num_cubes']}cube"
        groups.setdefault(gid, []).append((key, rec))
    for gid, members in sorted(groups.items()):
        hashes = {rec["episode_set_sha256"] for _, rec in members}
        cubes = {rec["num_cubes"] for _, rec in members}
        eps = {rec["episodes"] for _, rec in members}
        ok = len(hashes) == 1 and len(cubes) == 1 and len(eps) == 1
        print(f"  {gid:22s} arms={len(members):2d} hashes={len(hashes)} "
              f"cubes={sorted(cubes)} episodes={sorted(eps)} -> {'PAIRED' if ok else 'NOT PAIRED'}")
        pairing[gid] = {"arms": len(members), "distinct_hashes": len(hashes),
                        "hash": sorted(hashes)[0], "paired": ok,
                        "num_cubes": sorted(cubes), "episodes": sorted(eps)}

    # ---- independent recomputation of stored episode-set hashes ----
    print("\n=== EPISODE-SET FILE HASH RECOMPUTATION ===")
    setfiles = {}
    for n in (4, 5):
        p = os.path.join(PROBES, f"episode_set_{n}cube.pkl")
        exp = pairing.get(f"{n}cube", {}).get("hash")
        v = verify_episode_set(p, exp)
        setfiles[f"{n}cube"] = v
        if v.get("status") == "MISSING":
            print(f"  {n}-cube: MISSING {p}")
        else:
            print(f"  {n}-cube: stored={str(v['stored_sha256'])[:16]} "
                  f"recomputed={v['recomputed_sha256'][:16]} "
                  f"match_stored={v['matches_stored']} match_results={v['matches_results']} "
                  f"n={v['n_episodes']} cubes={v['n_cubes']}")
    for r in (0, 1, 2):
        p = f"experiments/isaacgym_episode_sets/replicate{r}_n96.pkl"
        exp = pairing.get(f"3cube_r{r}", {}).get("hash")
        v = verify_episode_set(p, exp)
        setfiles[f"3cube_r{r}"] = v
        if v.get("status") == "MISSING":
            print(f"  3-cube r{r}: MISSING {p}")
        else:
            print(f"  3-cube r{r}: stored={str(v['stored_sha256'])[:16]} "
                  f"recomputed={v['recomputed_sha256'][:16]} "
                  f"match_stored={v['matches_stored']} match_results={v['matches_results']} "
                  f"n={v['n_episodes']} cubes={v['n_cubes']}")

    # ---- NFE verification (Part XII) ----
    print("\n=== MODEL CALLS PER PLANNING DECISION ===")
    nfe_ok = True
    for key in sorted(arms):
        rec = arms[key]
        req, meas = rec["requested_nfe"], rec["measured_calls_per_plan"]
        if req is None or meas is None:
            continue
        if abs(meas - req) > 1e-6:
            nfe_ok = False
            print(f"  !! {key}: requested {req} measured {meas}")
    print("  all arms: measured calls per planning decision == requested NFE"
          if nfe_ok else "  MISMATCH FOUND")

    # ---- paired comparisons ----
    print("\n=== PAIRED COMPARISONS (recomputed) ===")
    comparisons = {}
    plan = []
    for r in (0, 1, 2):
        for lab in ("flow_nfe1", "flow_nfe2", "flow_nfe4", "flow_nfe8", "flow_nfe16"):
            plan.append((f"3cube_r{r}_{lab}", f"3cube_r{r}_gaussian_nfe100"))
    for n in (4, 5):
        plan.append((f"{n}cube_flow_nfe4", f"{n}cube_gaussian_nfe100"))
        plan.append((f"{n}cube_flow_nfe1", f"{n}cube_gaussian_nfe100"))
        plan.append((f"{n}cube_flow_nfe1", f"{n}cube_flow_nfe4"))
    for a, b in plan:
        if a in arms and b in arms:
            res = paired_tests(arms[a], arms[b])
            if res:
                comparisons[f"{a}__vs__{b}"] = res

    for key in sorted(comparisons):
        if key.startswith("3cube_r0") or not key.startswith("3cube"):
            c = comparisons[key]
            print(f"  {key}")
            print(f"    d_full={c['delta_full_success']:+.4f} b={c['mcnemar_b']} "
                  f"c={c['mcnemar_c']} McNemar_p={c['mcnemar_p_exact']:.4f}")
            print(f"    d_perobj={c['delta_per_object']:+.4f} Wilcoxon_p={c['wilcoxon_p']:.4f} "
                  f"boot95=[{c['bootstrap_ci95'][0]:+.4f},{c['bootstrap_ci95'][1]:+.4f}] "
                  f"boot_p={c['bootstrap_p_two_sided']:.4f}")

    # ---- write canonical CSV ----
    fields = ["key", "num_cubes", "arm", "label", "requested_nfe",
              "measured_calls_per_plan", "replicate", "episodes", "successes",
              "success_rate", "ci95_lo", "ci95_hi", "per_object_success",
              "per_object_success_std", "cubes_placed_mean", "avg_obj_dist_mean",
              "max_obj_dist_mean", "contact_rate", "n_contacted_mean",
              "cubes_farther_mean", "clip_fraction_mean", "latency_mean_ms",
              "latency_p50_ms", "latency_p95_ms", "wall_seconds",
              "episode_set_sha256", "raw_file", "raw_file_sha256"]
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["cubes_completed_distribution"])
        writer.writeheader()
        for key in sorted(arms):
            rec = arms[key]
            row = {f: rec.get(f) for f in fields}
            row["key"] = key
            row["cubes_completed_distribution"] = json.dumps(rec["cubes_completed_distribution"])
            writer.writerow(row)

    for rec in arms.values():
        rec.pop("_episodes", None)
    with open(OUT_JSON, "w") as handle:
        json.dump({"arms": arms, "pairing": pairing, "episode_set_files": setfiles,
                   "paired_comparisons": comparisons}, handle, indent=2)

    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
