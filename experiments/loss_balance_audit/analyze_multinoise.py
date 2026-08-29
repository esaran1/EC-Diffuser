"""Analysis for the multi-noise E16 vs E512 dispersion study. CPU only."""
import json, itertools, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "experiments/loss_balance_audit")
from latent_metric import chamfer_position

SEEDS = (42, 43, 44); ARMS = ("euler16", "euler512"); N_NOISE = 8
rng = np.random.default_rng(99)


def boot(d, n=20000):
    d = np.asarray(d); i = rng.integers(0, len(d), (n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


Z = np.load("experiments/loss_balance_audit/multinoise_endpoints.npz")
proto = json.load(open("experiments/loss_balance_audit/multinoise_protocol.json"))
res = {"protocol": proto["protocol"], "determinism_checks": proto["determinism_checks"]}

err, disp, medoid, bestof, displ = {}, {}, {}, {}, {}
for s in SEEDS:
    real = Z[f"s{s}_real"]                      # (M,24,10)
    M = len(real)
    for a in ARMS:
        G = Z[f"s{s}_{a}"]                      # (M,8,24,10)
        e = np.array([[chamfer_position(G[m, i], real[m]) for i in range(N_NOISE)]
                      for m in range(M)])        # (M,8)
        err[(s, a)] = e
        bestof[(s, a)] = e.min(1)
        # pairwise dispersion + medoid, within condition
        dsp, med = np.zeros(M), np.zeros(M)
        for m in range(M):
            D = np.zeros((N_NOISE, N_NOISE))
            for i, j in itertools.combinations(range(N_NOISE), 2):
                D[i, j] = D[j, i] = chamfer_position(G[m, i], G[m, j])
            iu = np.triu_indices(N_NOISE, 1)
            dsp[m] = D[iu].mean()
            med[m] = e[m, int(np.argmin(D.sum(1)))]
        disp[(s, a)] = dsp; medoid[(s, a)] = med
    # E16 <-> E512 endpoint displacement, same z
    A, B = Z[f"s{s}_euler16"], Z[f"s{s}_euler512"]
    displ[s] = np.array([[chamfer_position(A[m, i], B[m, i]) for i in range(N_NOISE)]
                         for m in range(M)])

print("=== 4/5/6/7 PER SEED ===")
print(f"{'seed':5s} {'arm':9s} {'mean err':>10s} {'best-of-8':>10s} {'dispersion':>11s} {'medoid err':>11s}")
for s in SEEDS:
    for a in ARMS:
        print(f"{s:5d} {a:9s} {err[(s,a)].mean():10.5f} {bestof[(s,a)].mean():10.5f} "
              f"{disp[(s,a)].mean():11.5f} {medoid[(s,a)].mean():11.5f}")
res["per_seed"] = {f"s{s}_{a}": {"mean_single_sample_error": float(err[(s,a)].mean()),
                                 "best_of_8": float(bestof[(s,a)].mean()),
                                 "dispersion_mean_pairwise": float(disp[(s,a)].mean()),
                                 "dispersion_median_pairwise": float(np.median(disp[(s,a)])),
                                 "medoid_error": float(medoid[(s,a)].mean())}
                   for s in SEEDS for a in ARMS}

print("\n=== 8/9 PAIRED PER-NOISE  E512 - E16 ===")
pn = {}
for s in SEEDS:
    d = (err[(s,"euler512")] - err[(s,"euler16")]).ravel()
    m, lo, hi = boot(d)
    frac = float((d > 0).mean())
    pn[f"s{s}"] = {"mean": m, "median": float(np.median(d)), "ci": [lo, hi],
                   "frac_noises_favouring_E16": frac, "n": int(d.size)}
    print(f"  s{s}: mean={m:+.5f} median={np.median(d):+.5f} CI=[{lo:+.5f},{hi:+.5f}] "
          f"E16 wins {frac*100:.1f}% of {d.size} noises")
dall = np.concatenate([(err[(s,'euler512')] - err[(s,'euler16')]).ravel() for s in SEEDS])
m, lo, hi = boot(dall)
pn["pooled_descriptive"] = {"mean": m, "median": float(np.median(dall)), "ci": [lo, hi],
                            "frac_noises_favouring_E16": float((dall > 0).mean()), "n": int(dall.size)}
print(f"  pooled(descriptive): mean={m:+.5f} CI=[{lo:+.5f},{hi:+.5f}] "
      f"E16 wins {(dall>0).mean()*100:.1f}%")
res["paired_per_noise_E512_minus_E16"] = pn

print("\n=== 10/11 DISPLACEMENT AND ITS CORRELATION WITH DEGRADATION ===")
dc = {}
for s in SEEDS:
    dp = displ[s].ravel(); dg = (err[(s,"euler512")] - err[(s,"euler16")]).ravel()
    rho, p = spearmanr(dp, dg)
    dc[f"s{s}"] = {"mean_displacement": float(dp.mean()),
                   "spearman_displacement_vs_delta_gt": {"rho": float(rho), "p": float(p)}}
    print(f"  s{s}: mean |E16-E512| = {dp.mean():.5f}   rho(displacement, dGT) = {rho:+.4f} (p={p:.3g})")
res["displacement"] = dc

print("\n=== 12 BEST-OF-K (expected over random subsets of the 8) ===")
kc = {}
for a in ARMS:
    row = []
    for K in (1, 2, 4, 8):
        vals = []
        for s in SEEDS:
            e = err[(s, a)]
            if K == 8:
                vals.append(e.min(1).mean())
            else:
                sub = np.stack([e[np.arange(len(e))[:, None],
                                  rng.permuted(np.tile(np.arange(N_NOISE), (len(e), 1)),
                                               axis=1)[:, :K]].min(1) for _ in range(200)])
                vals.append(sub.mean())
        row.append(float(np.mean(vals)))
    kc[a] = row
    print(f"  {a:9s} K=1:{row[0]:.5f}  K=2:{row[1]:.5f}  K=4:{row[2]:.5f}  K=8:{row[3]:.5f}")
res["best_of_K"] = kc

print("\n=== 3-SEED SUMMARY (mean of per-seed values, sd across seeds) ===")
summ = {}
for metric, store in [("mean_single_sample_error", err), ("best_of_8", bestof),
                      ("dispersion", disp), ("medoid_error", medoid)]:
    for a in ARMS:
        v = [ (store[(s,a)].mean() if metric!="best_of_8" else store[(s,a)].mean()) for s in SEEDS]
        summ[f"{metric}_{a}"] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                 "per_seed": [float(x) for x in v]}
    d = summ[f"{metric}_euler512"]["mean"] - summ[f"{metric}_euler16"]["mean"]
    print(f"  {metric:26s} E16={summ[f'{metric}_euler16']['mean']:.5f} "
          f"E512={summ[f'{metric}_euler512']['mean']:.5f}  diff={d:+.5f}")
res["summary"] = summ
json.dump(res, open("experiments/loss_balance_audit/multinoise_analysis.json", "w"), indent=2)
print("\nwrote multinoise_analysis.json")
