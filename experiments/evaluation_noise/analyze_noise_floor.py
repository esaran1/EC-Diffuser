"""Evaluator noise-floor analysis. Episodes are the primary unit; physics
repeats are nested within episode."""
import glob, json, os
import numpy as np

R_DIR = "experiments/evaluation_noise/results"
rng = np.random.default_rng(20260907)
out = {}

files = sorted(glob.glob(os.path.join(R_DIR, "r0_s42_nfe4_crnrep*.json")))
runs = []
for f in files:
    d = json.load(open(f))
    recs = {r["episode"]: r for r in d["episodes"]}
    runs.append(recs)
print(f"=== 6. RUN-LEVEL SUCCESS ACROSS {len(runs)} PHYSICS REPEATS (same arm, same CRN bank) ===")
eps = sorted(runs[0])
S = np.array([[float(r[e]["success"]) for e in eps] for r in runs])   # (R, n_ep)
for i, f in enumerate(files):
    print(f"  repeat {i+1}: {S[i].mean():.4f}  ({int(S[i].sum())}/{len(eps)})")
print(f"  mean={S.mean(1).mean():.4f}  SD={S.mean(1).std(ddof=1):.4f}  "
      f"range=[{S.mean(1).min():.4f}, {S.mean(1).max():.4f}]  "
      f"spread={100*(S.mean(1).max()-S.mean(1).min()):.1f} pp")
out["run_level"] = {"files": files, "rates": S.mean(1).tolist(),
                    "mean": float(S.mean(1).mean()), "sd": float(S.mean(1).std(ddof=1)),
                    "range_pp": float(100*(S.mean(1).max()-S.mean(1).min()))}

R = len(runs)
p = S.mean(0)   # per-episode success frequency
print(f"\n=== 7/8. EPISODE SUCCESS-FREQUENCY DISTRIBUTION (p_i over R={R}) ===")
robust_s = int((p == 1).sum()); robust_f = int((p == 0).sum()); border = int(((p > 0) & (p < 1)).sum())
print(f"  robust success (p=1):   {robust_s}/{len(eps)} ({100*robust_s/len(eps):.1f}%)")
print(f"  robust failure (p=0):   {robust_f}/{len(eps)} ({100*robust_f/len(eps):.1f}%)")
print(f"  PHYSICS-SENSITIVE:      {border}/{len(eps)} ({100*border/len(eps):.1f}%)")
print(f"  histogram of p_i: {np.bincount((p*R).astype(int), minlength=R+1).tolist()} (bins 0..{R})")
out["episode_frequency"] = {"robust_success": robust_s, "robust_failure": robust_f,
                            "physics_sensitive": border, "n_episodes": len(eps),
                            "hist": np.bincount((p*R).astype(int), minlength=R+1).tolist()}

print("\n=== 9. RELATIONSHIP TO THE 0.04 THRESHOLD ===")
dist = np.array([[r[e]["max_obj_dist"] for e in eps] for r in runs])
md = dist.mean(0)
for lab, m in [("robust success", p == 1), ("physics-sensitive", (p > 0) & (p < 1)), ("robust failure", p == 0)]:
    if m.sum():
        print(f"  {lab:20s} n={int(m.sum()):3d}  mean max_obj_dist={md[m].mean():.4f}  "
              f"|margin to 0.04|={np.abs(md[m]-0.04).mean():.4f}")
sens = (p > 0) & (p < 1)
out["threshold"] = {"mean_maxdist_by_class": {
    "robust_success": float(md[p == 1].mean()) if (p == 1).any() else None,
    "physics_sensitive": float(md[sens].mean()) if sens.any() else None,
    "robust_failure": float(md[p == 0].mean()) if (p == 0).any() else None}}

print("\n=== 10/11. HIERARCHICAL BOOTSTRAP: CI WIDTH vs R ===")
print("  estimator: resample EPISODES with replacement (primary unit), then")
print("  resample physics repeats within each chosen episode (nested).")
def boot_ci(Rsub, nboot=4000):
    w = []
    for _ in range(nboot):
        ei = rng.integers(0, len(eps), len(eps))
        ri = rng.integers(0, R, (len(eps), Rsub))
        w.append(S[ri, ei[:, None]].mean())
    return float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))
ciw = {}
for Rsub in (1, 2, 3, 5, 8):
    if Rsub > R: continue
    lo, hi = boot_ci(Rsub)
    ciw[Rsub] = 100*(hi-lo)
    print(f"  R={Rsub}: 95% CI = [{lo:.4f}, {hi:.4f}]  width = {100*(hi-lo):.2f} pp")
out["ci_width_pp"] = ciw

print("\n=== 12/14. RESOLUTION FOR A PAIRED TWO-ARM COMPARISON ===")
print("  Per-episode expected success p_i^A vs p_i^B; SE of the paired difference.")
print("  Physics variance within episode: Var[p_hat_i] = p_i(1-p_i)/R")
var_within = (p*(1-p)).mean()
n = len(eps)
for Rsub in (1, 3, 5, 8):
    # SE of mean paired difference: two independent arms, each with within-episode
    # sampling error; episode-level true differences add their own variance which
    # we cannot observe from a single arm, so this is the NOISE-ONLY floor.
    se = np.sqrt(2*var_within/Rsub/n)
    print(f"  R={Rsub}: noise-only SE of Delta = {100*se:.2f} pp; "
          f"95% half-width = {100*1.96*se:.2f} pp; "
          f"detectable |Delta| >~ {100*1.96*se:.1f} pp")
out["paired_resolution"] = {str(Rsub): float(100*1.96*np.sqrt(2*var_within/Rsub/n))
                            for Rsub in (1, 3, 5, 8)}
out["var_within_episode"] = float(var_within)
json.dump(out, open("experiments/evaluation_noise/noise_floor.json", "w"), indent=2)
print("\nwrote noise_floor.json")
