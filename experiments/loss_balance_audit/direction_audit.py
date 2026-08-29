"""CPU-only directional decomposition of the E16->E512 shift.

Operates on POSITION features only (latent_metric.POS), on the within-run
multi-noise endpoints. No GPU, no regeneration.

Correspondence discipline:
  * E16 and E512 share x0/model/noise, so their particle SLOTS correspond
    computationally; this is VERIFIED, not assumed (slot_correspondence).
  * The observed future has no stable particle order, so it is Hungarian-matched
    to E16 and that assignment is FROZEN for E512 (E16-anchored). Sensitivity
    analyses use E512-anchored and independently-optimal matching.
"""
import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
sys.path.insert(0, "experiments/loss_balance_audit")
from latent_metric import POS, chamfer_position

SEEDS = (42, 43, 44); N_NOISE = 8
rng = np.random.default_rng(4242)
Z = np.load("experiments/loss_balance_audit/multinoise_endpoints.npz")


def match(a, b):
    """Hungarian assignment of b's particles onto a, by position."""
    c = np.linalg.norm(a[:, None, POS] - b[None, :, POS], axis=-1)
    r, col = linear_sum_assignment(c)
    return col


def boot(d, n=20000):
    d = np.asarray(d); i = rng.integers(0, len(d), (n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


out = {}

# ---------- 2. particle SLOT correspondence between E16 and E512 ----------
sc = {}
for s in SEEDS:
    A, B = Z[f"s{s}_euler16"], Z[f"s{s}_euler512"]
    ident, opt = [], []
    for m in range(len(A)):
        for i in range(N_NOISE):
            a, b = A[m, i], B[m, i]
            d_ident = np.linalg.norm(a[:, POS] - b[:, POS], axis=-1).mean()
            col = match(a, b)
            d_opt = np.linalg.norm(a[:, POS] - b[col][:, POS], axis=-1).mean()
            ident.append(d_ident); opt.append(d_opt)
            # is the optimal assignment the identity?
    ident, opt = np.array(ident), np.array(opt)
    frac_id = float(np.mean(np.isclose(ident, opt, rtol=1e-6, atol=1e-9)))
    sc[f"s{s}"] = {"mean_slotwise_distance": float(ident.mean()),
                   "mean_optimally_matched_distance": float(opt.mean()),
                   "ratio_slot_over_optimal": float(ident.mean() / opt.mean()),
                   "fraction_where_identity_is_optimal": frac_id}
out["slot_correspondence"] = sc
print("=== 2. E16<->E512 particle slot correspondence ===")
for k, v in sc.items():
    print(f"  {k}: slotwise={v['mean_slotwise_distance']:.5f} optimal={v['mean_optimally_matched_distance']:.5f} "
          f"ratio={v['ratio_slot_over_optimal']:.3f} identity-optimal in {v['fraction_where_identity_is_optimal']*100:.1f}%")

# ---------- 3-5. displacement + decomposition vs observed future ----------
def decompose(anchor):
    """anchor in {'e16','e512','independent'} - which endpoint the GT is matched to."""
    per_seed, rows = {}, []
    for s in SEEDS:
        A, B, R = Z[f"s{s}_euler16"], Z[f"s{s}_euler512"], Z[f"s{s}_real"]
        cosv, par, orth, frac, dmag, derr = [], [], [], [], [], []
        for m in range(len(A)):
            for i in range(N_NOISE):
                a, b, r = A[m, i], B[m, i], R[m]
                if anchor == "e16":
                    col = match(a, r)
                    gt_a = r[col]; gt_b = r[col]
                elif anchor == "e512":
                    col = match(b, r)
                    gt_a = r[col]; gt_b = r[col]
                else:
                    gt_a = r[match(a, r)]; gt_b = r[match(b, r)]
                p16, p512 = a[:, POS], b[:, POS]
                d = (p512 - p16).ravel()
                g = (gt_a[:, POS] - p16).ravel()
                ng, nd = np.linalg.norm(g), np.linalg.norm(d)
                if ng < 1e-12 or nd < 1e-12:
                    continue
                gh = g / ng
                p = float(d @ gh)
                o = float(np.linalg.norm(d - p * gh))
                cosv.append(p / nd); par.append(p); orth.append(o)
                frac.append(abs(p) / nd); dmag.append(nd)
                derr.append(chamfer_position(b, r) - chamfer_position(a, r))
        cosv = np.array(cosv); par = np.array(par); orth = np.array(orth)
        frac = np.array(frac); dmag = np.array(dmag); derr = np.array(derr)
        mc, lc, hc = boot(cosv); mp, lp, hp = boot(par)
        per_seed[f"s{s}"] = {
            "n": int(len(cosv)),
            "cos_future": {"mean": mc, "median": float(np.median(cosv)), "ci": [lc, hc],
                           "frac_negative": float((cosv < 0).mean())},
            "parallel_future": {"mean": mp, "median": float(np.median(par)), "ci": [lp, hp]},
            "orthogonal_future_mean": float(orth.mean()),
            "frac_of_shift_along_target_axis": float(frac.mean()),
            "displacement_magnitude_mean": float(dmag.mean()),
            "spearman_vs_delta_err": {
                "cos_future": [float(x) for x in spearmanr(cosv, derr)],
                "parallel_future": [float(x) for x in spearmanr(par, derr)],
                "orthogonal_future": [float(x) for x in spearmanr(orth, derr)],
                "displacement_magnitude": [float(x) for x in spearmanr(dmag, derr)]}}
        rows.append((cosv, par, orth, derr))
    return per_seed, rows


for anchor in ["e16", "e512", "independent"]:
    ps, rows = decompose(anchor)
    out[f"decomposition_{anchor}_anchored"] = ps
    print(f"\n=== 4/5. decomposition ({anchor}-anchored matching) ===")
    for k, v in ps.items():
        c = v["cos_future"]
        print(f"  {k}: cos_future={c['mean']:+.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] "
              f"neg {c['frac_negative']*100:.1f}% | parallel={v['parallel_future']['mean']:+.5f} "
              f"| orth={v['orthogonal_future_mean']:.5f} | |para|/|d|={v['frac_of_shift_along_target_axis']:.3f} "
              f"| |d|={v['displacement_magnitude_mean']:.5f}")
    if anchor == "e16":
        print("  Spearman vs delta-error (s42):",
              {k: [round(x, 4) for x in val] for k, val in ps["s42"]["spearman_vs_delta_err"].items()})

# ---------- 6. interpolation curve, permutation-invariant metric ----------
print("\n=== 6. alpha interpolation E16 -> E512 (validated chamfer metric) ===")
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
interp = {}
for s in SEEDS:
    A, B, R = Z[f"s{s}_euler16"], Z[f"s{s}_euler512"], Z[f"s{s}_real"]
    curve = []
    for al in alphas:
        v = [chamfer_position(A[m, i] + al * (B[m, i] - A[m, i]), R[m])
             for m in range(len(A)) for i in range(N_NOISE)]
        curve.append(float(np.mean(v)))
    interp[f"s{s}"] = curve
    print(f"  s{s}: " + "  ".join(f"a={a}:{c:.5f}" for a, c in zip(alphas, curve)))
mono = all(all(interp[f"s{s}"][i] <= interp[f"s{s}"][i+1] + 1e-9 for i in range(4)) for s in SEEDS)
interp["alphas"] = alphas; interp["monotone_increasing_all_seeds"] = bool(mono)
out["interpolation"] = interp
print(f"  monotone increasing on all three seeds: {mono}")

# ---------- 13. mean signed displacement field ----------
print("\n=== 13. magnitude of the average solver-induced bias ===")
md = {}
for s in SEEDS:
    A, B, R = Z[f"s{s}_euler16"], Z[f"s{s}_euler512"], Z[f"s{s}_real"]
    d = (B[:, :, :, POS] - A[:, :, :, POS])
    per_particle_mean = d.mean(axis=(0, 1))          # (24,2) slot-wise mean over samples/noises
    mean_norm = float(np.linalg.norm(per_particle_mean, axis=-1).mean())
    typ_disp = float(np.linalg.norm(d, axis=-1).mean())
    e16gt = float(np.mean([chamfer_position(A[m, i], R[m])
                           for m in range(len(A)) for i in range(N_NOISE)]))
    md[f"s{s}"] = {"mean_signed_displacement_norm": mean_norm,
                   "mean_unsigned_displacement": typ_disp,
                   "consistency_ratio_signed_over_unsigned": mean_norm / typ_disp,
                   "typical_E16_to_GT": e16gt}
    print(f"  s{s}: |E[delta]|={mean_norm:.5f}  E|delta|={typ_disp:.5f}  "
          f"ratio={mean_norm/typ_disp:.3f}  (E16->GT = {e16gt:.5f})")
out["mean_shift"] = md

json.dump(out, open("experiments/loss_balance_audit/direction_audit.json", "w"), indent=2)
print("\nwrote direction_audit.json")
