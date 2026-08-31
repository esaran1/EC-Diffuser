"""Predeclared analysis for the 4-cube R=3 evaluation (sections 9-12, 15-16)."""
import glob, json, os
import numpy as np
from itertools import product

D = "experiments/topconf/fourcube_r3"
SEEDS = (42, 43, 44); REPS = (1, 2, 3); ARMS = (2, 4)
rng = np.random.default_rng(20260401)
out = {}


def raw(seed, nfe):
    S = []
    for r in REPS:
        d = json.load(open(f"{D}/4cube_H100_s{seed}_nfe{nfe}_rep{r}.json"))
        S.append({e["episode"]: float(e["success"]) for e in d["episodes"]})
    eps = sorted(S[0])
    return np.array([[S[i][e] for e in eps] for i in range(len(REPS))]), eps


def hier_boot(A, B, n=20000):
    """Episodes top-level, physics realizations nested. Returns delta = B - A."""
    ne = A.shape[1]; o = []
    for _ in range(n):
        ei = rng.integers(0, ne, ne)
        ra = rng.integers(0, A.shape[0], (ne, A.shape[0]))
        rb = rng.integers(0, B.shape[0], (ne, B.shape[0]))
        o.append((B[rb, ei[:, None]].mean(1) - A[ra, ei[:, None]].mean(1)).mean())
    return float(np.mean(o)), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


print("=== 6/7/8. PER-SEED (4-cube, fixed H=100, R=3) ===")
print(f"{'seed':>5s} {'p(NFE2)':>9s} {'p(NFE4)':>9s} {'delta(4-2)':>11s} {'95% CI':>22s}")
per, ds = {}, []
for s in SEEDS:
    A, eps = raw(s, 2); B, _ = raw(s, 4)
    m, lo, hi = hier_boot(A, B); ds.append(m)
    per[str(s)] = {"p_nfe2": float(A.mean()), "p_nfe4": float(B.mean()),
                   "delta": m, "ci": [lo, hi],
                   "per_realization_nfe2": [float(x) for x in A.mean(1)],
                   "per_realization_nfe4": [float(x) for x in B.mean(1)]}
    print(f"{s:5d} {A.mean():9.4f} {B.mean():9.4f} {m:+11.4f} [{lo:+.4f},{hi:+.4f}]")
out["per_seed"] = per
print(f"\n=== 9. THREE-SEED SUMMARY (N=3 checkpoints) ===")
print(f"  deltas {[round(100*x,2) for x in ds]} pp   mean {100*np.mean(ds):+.2f}  SD {100*np.std(ds,ddof=1):.2f}")
print(f"  signs: {['+' if x>0 else '-' for x in ds]}")
out["three_seed"] = {"deltas": [float(x) for x in ds], "mean": float(np.mean(ds)),
                     "sd": float(np.std(ds, ddof=1))}

print("\n=== 10/11. WITHIN-ARM PHYSICS VARIABILITY + SENSITIVE EPISODES ===")
var = {}
for s in SEEDS:
    for n in ARMS:
        A, _ = raw(s, n)
        p = A.mean(0)
        var[f"s{s}_nfe{n}"] = {
            "per_realization": [float(x) for x in A.mean(1)],
            "spread_pp": float(100*(A.mean(1).max()-A.mean(1).min())),
            "robust_success": int((p == 1).sum()), "robust_fail": int((p == 0).sum()),
            "sensitive": int(((p > 0) & (p < 1)).sum()), "n_ep": len(p)}
        v = var[f"s{s}_nfe{n}"]
        print(f"  s{s} nfe{n}: realizations {[round(100*x,1) for x in A.mean(1)]} "
              f"spread {v['spread_pp']:.1f}pp | robust+ {v['robust_success']:2d} "
              f"robust- {v['robust_fail']:2d} sensitive {v['sensitive']:2d}")
out["variability"] = var
sens = np.mean([var[f"s{s}_nfe{n}"]["sensitive"] for s in SEEDS for n in ARMS])
nep = var[f"s{SEEDS[0]}_nfe2"]["n_ep"]
print(f"\n  mean physics-sensitive fraction (4-cube): {sens:.1f}/{nep} = {100*sens/nep:.1f}%")
out["sensitive_fraction_4cube"] = float(sens/nep)

print("\n=== 12/13/14. RECONSTRUCTED SINGLE-REALIZATION VIEWS ===")
rec = {}
allf = []
for s in SEEDS:
    A, _ = raw(s, 2); B, _ = raw(s, 4)
    cal = B.mean() - A.mean()
    singles = [B[j].mean() - A[i].mean() for i, j in product(range(3), range(3))]
    flips = sum(1 for x in singles if np.sign(x) != np.sign(cal) and cal != 0)
    big = sum(1 for x in singles if abs(x - cal) > 0.05)
    rec[str(s)] = {"calibrated": float(cal), "singles": [float(x) for x in singles],
                   "sign_flips": int(flips), "n_views": len(singles),
                   "mag_gt_5pp": int(big),
                   "range_pp": float(100*(max(singles)-min(singles))),
                   "sd_pp": float(100*np.std(singles, ddof=1))}
    allf.append((flips, big, len(singles)))
    print(f"  s{s}: R=3 delta {100*cal:+.1f} pp | singles {[round(100*x,1) for x in singles]}")
    print(f"        sign flips {flips}/9, |diff|>5pp {big}/9, range {rec[str(s)]['range_pp']:.1f} pp")
tf = sum(x[0] for x in allf); tb = sum(x[1] for x in allf); tn = sum(x[2] for x in allf)
print(f"\n  4-cube overall: sign disagreement {tf}/{tn} = {100*tf/tn:.0f}%; "
      f"magnitude>5pp {tb}/{tn} = {100*tb/tn:.0f}%")
out["reconstructed"] = rec
out["sign_disagreement_rate"] = tf/tn
out["mag_instability_rate"] = tb/tn

print("\n=== 15. COMPARISON TO 3-CUBE EVALUATOR NOISE ===")
try:
    nf = json.load(open("experiments/evaluation_noise/noise_floor.json"))
    print(f"  3-cube same-arm spread (R=8): {nf['run_level']['range_pp']:.1f} pp, "
          f"SD {100*nf['run_level']['sd']:.2f} pp")
    print(f"  3-cube physics-sensitive: {nf['episode_frequency']['physics_sensitive']}/96 = "
          f"{100*nf['episode_frequency']['physics_sensitive']/96:.0f}%")
    print(f"  3-cube sign disagreement (81 reconstructed views): 25%")
    sp = np.mean([var[f"s{s}_nfe{n}"]["spread_pp"] for s in SEEDS for n in ARMS])
    print(f"  4-cube mean within-arm spread: {sp:.1f} pp")
    print(f"  4-cube physics-sensitive: {100*sens/nep:.0f}%")
    print(f"  4-cube sign disagreement: {100*tf/tn:.0f}%")
    out["comparison_3cube"] = {"3cube_spread_pp": nf['run_level']['range_pp'],
                               "3cube_sensitive_frac": nf['episode_frequency']['physics_sensitive']/96,
                               "4cube_mean_spread_pp": float(sp),
                               "4cube_sensitive_frac": float(sens/nep)}
except Exception as e:
    print("  [warn]", e)

print("\n=== 16. CONTACT LOCALIZATION (existing diagnostics) ===")
try:
    ct = {}
    for s in SEEDS:
        for n in ARMS:
            for r in REPS:
                d = json.load(open(f"{D}/4cube_H100_s{s}_nfe{n}_rep{r}.json"))
                for e in d["episodes"]:
                    ct.setdefault(e["episode"], []).append(
                        (e["success"], e["first_contact_step"], e["max_obj_dist"], e["n_contacted"]))
    A, _ = raw(SEEDS[0], 2)
    sens_ep = {e for e in ct if 0 < np.mean([x[0] for x in ct[e]]) < 1}
    rob_ep = {e for e in ct if np.mean([x[0] for x in ct[e]]) == 1}
    def stat(S, i): return float(np.mean([x[i] for e in S for x in ct[e]])) if S else float("nan")
    print(f"  physics-sensitive episodes: n={len(sens_ep)}  robust-success: n={len(rob_ep)}")
    print(f"    mean max_obj_dist  sensitive {stat(sens_ep,2):.4f} vs robust {stat(rob_ep,2):.4f}")
    print(f"    mean n_contacted   sensitive {stat(sens_ep,3):.3f} vs robust {stat(rob_ep,3):.3f}")
    print(f"    first_contact_step sensitive {stat(sens_ep,1):.3f} vs robust {stat(rob_ep,1):.3f}")
    out["contact"] = {"n_sensitive": len(sens_ep), "n_robust": len(rob_ep),
                      "maxdist_sensitive": stat(sens_ep,2), "maxdist_robust": stat(rob_ep,2),
                      "ncontact_sensitive": stat(sens_ep,3), "ncontact_robust": stat(rob_ep,3)}
except Exception as e:
    print("  [warn]", e)

json.dump(out, open("experiments/topconf/fourcube_r3_analysis.json", "w"), indent=2)
print("\nwrote fourcube_r3_analysis.json")
