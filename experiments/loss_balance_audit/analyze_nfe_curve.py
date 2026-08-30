"""CPU analysis of the arm-neutral NFE curve."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "experiments/loss_balance_audit")
from latent_metric import chamfer_position

SEEDS = (42, 43, 44); NPV = 24
rng = np.random.default_rng(20260903)
Z = np.load("experiments/loss_balance_audit/arm_neutral_nfe_curve.npz", allow_pickle=True)
P = json.loads(str(Z["_protocol"])); NFES = P["nfes"]; NN = P["n_noise"]
out = {"protocol": P}


def boot(d, n=20000):
    d = np.asarray(d).ravel(); i = rng.integers(0, len(d), (n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


raw = {}
print("=== 4/5. ARM-NEUTRAL ERROR vs NFE (per seed) ===")
print(f"{'NFE':>5s} | " + " ".join(f"{'act s'+str(s):>10s}" for s in SEEDS) + f" {'act mean':>10s} | "
      + " ".join(f"{'st s'+str(s):>10s}" for s in SEEDS) + f" {'st mean':>10s}")
for n in NFES:
    arow, srow = [], []
    for s in SEEDS:
        T = Z[f"s{s}_target_t1"][:, :NPV]; ta = Z[f"s{s}_target_action"]
        g = Z[f"s{s}_nfe{n}_state"]; a = Z[f"s{s}_nfe{n}_action"]
        se = np.array([[chamfer_position(g[m, i], T[m]) for i in range(NN)] for m in range(len(T))])
        ae = np.linalg.norm(a - ta[:, None, :], axis=-1)
        raw[(n, s, "state")] = se; raw[(n, s, "action")] = ae
        arow.append(ae.mean()); srow.append(se.mean())
    print(f"{n:5d} | " + " ".join(f"{v:10.5f}" for v in arow) + f" {np.mean(arow):10.5f} | "
          + " ".join(f"{v:10.5f}" for v in srow) + f" {np.mean(srow):10.5f}")
out["curve"] = {str(n): {"action_per_seed": [float(raw[(n, s, "action")].mean()) for s in SEEDS],
                         "state_per_seed": [float(raw[(n, s, "state")].mean()) for s in SEEDS],
                         "action_mean": float(np.mean([raw[(n, s, "action")].mean() for s in SEEDS])),
                         "state_mean": float(np.mean([raw[(n, s, "state")].mean() for s in SEEDS])),
                         "action_sd": float(np.std([raw[(n, s, "action")].mean() for s in SEEDS], ddof=1)),
                         "state_sd": float(np.std([raw[(n, s, "state")].mean() for s in SEEDS], ddof=1))}
                for n in NFES}

print("\n=== 6. ACTION magnitude / direction decomposition (3-seed mean) ===")
print(f"{'NFE':>5s} {'L2':>9s} {'L1':>9s} {'|mag|err':>9s} {'cos':>8s}")
dec = {}
for n in NFES:
    l1, l2, mg, cs = [], [], [], []
    for s in SEEDS:
        ta = Z[f"s{s}_target_action"]; a = Z[f"s{s}_nfe{n}_action"]
        d = a - ta[:, None, :]
        l1.append(np.abs(d).sum(-1).mean()); l2.append(np.linalg.norm(d, axis=-1).mean())
        mg.append(np.abs(np.linalg.norm(a, axis=-1) - np.linalg.norm(ta, axis=-1)[:, None]).mean())
        nz = np.linalg.norm(ta, axis=-1) > 0.1
        c = (a @ ta[:, :, None]).squeeze(-1) / (np.linalg.norm(a, axis=-1)
                                                * np.linalg.norm(ta, axis=-1)[:, None] + 1e-9)
        cs.append(c[nz].mean())
    dec[str(n)] = {"l1": float(np.mean(l1)), "l2": float(np.mean(l2)),
                   "mag_err": float(np.mean(mg)), "cos": float(np.mean(cs))}
    print(f"{n:5d} {np.mean(l2):9.5f} {np.mean(l1):9.5f} {np.mean(mg):9.5f} {np.mean(cs):8.4f}")
out["action_decomposition"] = dec

am = {n: np.mean([raw[(n, s, "action")].mean() for s in SEEDS]) for n in NFES}
sm = {n: np.mean([raw[(n, s, "state")].mean() for s in SEEDS]) for n in NFES}
best_a = min(am, key=am.get); best_s = min(sm, key=sm.get)
print(f"\n=== 8. NFE minimising ACTION error: {best_a} (err {am[best_a]:.5f})")
print(f"       NFE minimising STATE  error: {best_s} (err {sm[best_s]:.5f})")
per_seed_best = {s: min(NFES, key=lambda n: raw[(n, s, "action")].mean()) for s in SEEDS}
print(f"       per-seed action optimum: {per_seed_best}")
out["optima"] = {"action_best_nfe": best_a, "state_best_nfe": best_s,
                 "per_seed_action_best": {str(k): int(v) for k, v in per_seed_best.items()}}

print("\n=== 9. is STATE effectively NFE-invariant? ===")
sv = np.array([sm[n] for n in NFES])
print(f"  state range over all NFE: {sv.min():.5f} - {sv.max():.5f}  "
      f"(spread {100*(sv.max()-sv.min())/sv.min():.2f}% of min)")
av = np.array([am[n] for n in NFES])
print(f"  action range over all NFE: {av.min():.5f} - {av.max():.5f}  "
      f"(spread {100*(av.max()-av.min())/av.min():.2f}% of min)")
out["invariance"] = {"state_rel_spread": float((sv.max()-sv.min())/sv.min()),
                     "action_rel_spread": float((av.max()-av.min())/av.min())}

print("\n=== 10/12/13. degradation from the action optimum, and fraction of the E512 effect ===")
den = am[512] - am[best_a]
frac = {}
print(f"{'NFE':>5s} {'act err':>9s} {'rel degr':>9s} {'frac of E512 effect':>20s}")
for n in NFES:
    rel = (am[n] - am[best_a]) / am[best_a]
    f = (am[n] - am[best_a]) / den if den != 0 else float("nan")
    frac[str(n)] = {"rel_degradation": float(rel), "fraction_of_E512_effect": float(f)}
    print(f"{n:5d} {am[n]:9.5f} {rel*100:8.2f}% {f*100:19.1f}%")
k80 = [n for n in NFES if n > best_a and frac[str(n)]["fraction_of_E512_effect"] >= 0.80]
print(f"  smallest NFE reaching >=80% of the E512 degradation: {k80[0] if k80 else 'none'}")
out["degradation"] = frac
out["smallest_nfe_80pct"] = int(k80[0]) if k80 else None
# state equivalent
print("  state relative degradation from its own optimum:")
for n in NFES:
    print(f"    NFE {n:>3d}: {100*(sm[n]-sm[best_s])/sm[best_s]:+.2f}%")

print("\n=== 11. PAIRED changes from the action optimum (per noise, same x0) ===")
pa = {}
for n in NFES:
    if n == best_a: continue
    per = {}
    for s in SEEDS:
        d = (raw[(n, s, "action")] - raw[(best_a, s, "action")]).ravel()
        m, lo, hi = boot(d)
        per[f"s{s}"] = {"mean": m, "median": float(np.median(d)), "ci": [lo, hi],
                        "frac_worse": float((d > 0).mean())}
    pa[str(n)] = per
    v = [per[f"s{s}"]["mean"] for s in SEEDS]
    w = [per[f"s{s}"]["frac_worse"] for s in SEEDS]
    print(f"  NFE {best_a}->{n:<4d}: mean d = {np.mean(v):+.5f}  worse in {100*np.mean(w):.1f}% of noises "
          f"(per-seed {[round(x,5) for x in v]})")
out["paired_from_action_optimum"] = pa

print("\n=== 12. state/action coupling across NFE steps ===")
cp = {}
for i in range(len(NFES) - 1):
    a, b = NFES[i], NFES[i + 1]
    rr = []
    for s in SEEDS:
        da = (raw[(b, s, "action")] - raw[(a, s, "action")]).ravel()
        ds = (raw[(b, s, "state")] - raw[(a, s, "state")]).ravel()
        rr.append(spearmanr(da, ds)[0])
    cp[f"{a}->{b}"] = float(np.mean(rr))
    print(f"  {a:>3d} -> {b:<4d}: mean rho(dAction, dState) = {np.mean(rr):+.4f}")
out["coupling"] = cp

json.dump(out, open("experiments/loss_balance_audit/nfe_curve_analysis.json", "w"), indent=2)
print("\nwrote nfe_curve_analysis.json")
