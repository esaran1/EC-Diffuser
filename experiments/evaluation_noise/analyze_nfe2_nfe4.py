"""NFE1 vs NFE4 at R=3, calibrated hierarchical analysis.

Episodes are the top-level sampling unit; physics realizations are nested.
Physics repeats are NOT paired across arms (protocol section 8): the estimand is
per-episode success probability p_i under each arm.
"""
import glob, json, os
import numpy as np

R_DIR = "experiments/evaluation_noise/results"
SEEDS = (42, 43, 44); R = 3; MARGIN = 0.05
rng = np.random.default_rng(20260908)
out = {"protocol": {"R": R, "seeds": list(SEEDS), "arms": [2, 4],
                    "practical_margin_pp": MARGIN * 100,
                    "estimand": "p_i = per-episode success frequency over R physics realizations",
                    "pairing": "episodes paired across arms; physics realizations NOT paired"}}


def load(seed, nfe):
    S = []
    for r in (1, 2, 3):
        f = os.path.join(R_DIR, f"r0_s{seed}_nfe{nfe}_n24rep{r}.json")
        d = json.load(open(f))
        recs = {x["episode"]: x for x in d["episodes"]}
        S.append(recs)
    eps = sorted(S[0])
    return eps, S


print("=== 5/6/7. PER-SEED SUCCESS PROBABILITIES (R=3) ===")
print(f"{'seed':>5s} {'p(NFE2)':>9s} {'p(NFE4)':>9s} {'Delta':>9s} {'95% CI':>22s}")
per_seed, P = {}, {}
for s in SEEDS:
    eps, S1 = load(s, 2); _, S4 = load(s, 4)
    p1 = np.array([np.mean([float(r[e]["success"]) for r in S1]) for e in eps])
    p4 = np.array([np.mean([float(r[e]["success"]) for r in S4]) for e in eps])
    P[s] = (p1, p4, eps, S1, S4)
    d = p4 - p1
    # hierarchical bootstrap: resample EPISODES, then realizations within episode
    raw1 = np.array([[float(r[e]["success"]) for e in eps] for r in S1])   # (R, n)
    raw4 = np.array([[float(r[e]["success"]) for e in eps] for r in S4])
    bs = []
    for _ in range(20000):
        ei = rng.integers(0, len(eps), len(eps))
        r1 = rng.integers(0, R, (len(eps), R)); r4 = rng.integers(0, R, (len(eps), R))
        b1 = raw1[r1, ei[:, None]].mean(1); b4 = raw4[r4, ei[:, None]].mean(1)
        bs.append((b4 - b1).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    per_seed[s] = {"p_nfe2": float(p1.mean()), "p_nfe4": float(p4.mean()),
                   "delta": float(d.mean()), "ci": [float(lo), float(hi)],
                   "eff_successes_nfe2": float(p1.sum()), "eff_successes_nfe4": float(p4.sum()),
                   "n_episodes": len(eps)}
    print(f"{s:5d} {p1.mean():9.4f} {p4.mean():9.4f} {d.mean():+9.4f} [{lo:+.4f},{hi:+.4f}]")
    print(f"      effective counts: {p1.sum():.1f}/96 vs {p4.sum():.1f}/96")
out["per_seed"] = per_seed

deltas = [per_seed[s]["delta"] for s in SEEDS]
print(f"\n=== THREE-SEED SUMMARY (N=3 checkpoints) ===")
print(f"  deltas: {[round(x,4) for x in deltas]}   signs: {['+' if x>0 else '-' for x in deltas]}")
print(f"  mean={np.mean(deltas):+.4f}  SD={np.std(deltas, ddof=1):.4f}")
out["three_seed"] = {"deltas": [float(x) for x in deltas], "mean": float(np.mean(deltas)),
                     "sd": float(np.std(deltas, ddof=1))}

print(f"\n=== 9. RELATION TO THE PREDECLARED +/-{MARGIN*100:.0f} pp MARGIN ===")
for s in SEEDS:
    lo, hi = per_seed[s]["ci"]
    inside = (lo >= -MARGIN) and (hi <= MARGIN)
    excl0 = (lo > 0) or (hi < 0)
    print(f"  seed {s}: CI [{lo:+.4f},{hi:+.4f}]  within +/-5pp: {inside}  excludes 0: {excl0}")
out["margin"] = {str(s): {"within_margin": bool(per_seed[s]["ci"][0] >= -MARGIN and per_seed[s]["ci"][1] <= MARGIN),
                          "excludes_zero": bool(per_seed[s]["ci"][0] > 0 or per_seed[s]["ci"][1] < 0)}
                 for s in SEEDS}

print("\n=== 10. EPISODE-LEVEL Delta_i DISTRIBUTION ===")
epi = {}
for s in SEEDS:
    p1, p4, eps, _, _ = P[s]
    d = p4 - p1
    epi[str(s)] = {"positive": int((d > 0).sum()), "zero": int((d == 0).sum()),
                   "negative": int((d < 0).sum()), "mean_abs_nonzero": float(np.abs(d[d != 0]).mean()) if (d != 0).any() else 0.0}
    print(f"  seed {s}: Delta_i > 0 in {int((d>0).sum()):3d}, = 0 in {int((d==0).sum()):3d}, "
          f"< 0 in {int((d<0).sum()):3d} episodes | mean|Delta_i| when nonzero = "
          f"{np.abs(d[d!=0]).mean() if (d!=0).any() else 0:.3f}")
out["episode_delta"] = epi

print("\n=== 11. PHYSICS-SENSITIVE EPISODE ANALYSIS ===")
sens = {}
for s in SEEDS:
    p1, p4, eps, _, _ = P[s]
    def cls(p): return (int((p == 1).sum()), int((p == 0).sum()), int(((p > 0) & (p < 1)).sum()))
    a, b, c = cls(p1); a4, b4, c4 = cls(p4)
    sens[str(s)] = {"nfe2": {"robust_success": a, "robust_fail": b, "sensitive": c},
                    "nfe4": {"robust_success": a4, "robust_fail": b4, "sensitive": c4}}
    print(f"  seed {s}: NFE2 robust+={a:3d} robust-={b:3d} sensitive={c:3d} | "
          f"NFE4 robust+={a4:3d} robust-={b4:3d} sensitive={c4:3d}")
    # where does the change happen?
    either_sens = ((p1 > 0) & (p1 < 1)) | ((p4 > 0) & (p4 < 1))
    d = p4 - p1
    print(f"           Delta on sensitive-either ({int(either_sens.sum())} eps): {d[either_sens].mean():+.4f}; "
          f"on both-robust ({int((~either_sens).sum())} eps): {d[~either_sens].mean() if (~either_sens).any() else 0:+.4f}")
out["sensitivity"] = sens

print("\n=== 12. SECONDARY CANONICAL METRICS (episode-mean over R, then across episodes) ===")
keys = ["goal_success_frac", "cubes_placed", "avg_obj_dist", "max_obj_dist", "n_contacted", "cubes_farther"]
sec = {}
print(f"{'metric':20s} " + " ".join(f"{'s'+str(s)+' d':>10s}" for s in SEEDS) + f" {'mean d':>10s}")
for k in keys:
    ds = []
    for s in SEEDS:
        _, _, eps, S1, S4 = P[s]
        v1 = np.array([np.mean([r[e][k] for r in S1]) for e in eps])
        v4 = np.array([np.mean([r[e][k] for r in S4]) for e in eps])
        ds.append(float((v4 - v1).mean()))
    sec[k] = ds
    print(f"{k:20s} " + " ".join(f"{v:10.4f}" for v in ds) + f" {np.mean(ds):10.4f}")
out["secondary"] = sec
json.dump(out, open("experiments/evaluation_noise/nfe2_nfe4_analysis.json", "w"), indent=2)
print("\nwrote nfe2_nfe4_analysis.json")
