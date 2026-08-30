"""CPU analysis of the arm-neutral evaluation."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "experiments/loss_balance_audit")
from latent_metric import chamfer_position

SEEDS = (42, 43, 44); N_NOISE = 8; NPV = 24
rng = np.random.default_rng(31337)
Z = np.load("experiments/loss_balance_audit/arm_neutral_endpoints.npz", allow_pickle=True)
out = {"preflight": json.loads(str(Z["_preflight"]))}


def boot(d, n=20000):
    d = np.asarray(d).ravel(); i = rng.integers(0, len(d), (n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def gen_t1(s, arm):
    o = Z[f"s{s}_{arm}_obs_unnorm"]                     # (96,8,5,480)
    return o[:, :, 1].reshape(o.shape[0], N_NOISE, 2, NPV, 10)[:, :, 0]


st, ac = {}, {}
print("=== 8/9. STATE: recorded t1 target (ARM-NEUTRAL) ===")
for s in SEEDS:
    T = Z[f"s{s}_target_t1_latent"][:, :NPV]            # view 0 of the RECORDED next state
    E = {}
    for a in ("euler16", "euler512"):
        g = gen_t1(s, a)
        E[a] = np.array([[chamfer_position(g[m, i], T[m]) for i in range(N_NOISE)]
                         for m in range(len(T))])
    d = (E["euler512"] - E["euler16"]).ravel()
    m, lo, hi = boot(d)
    st[f"s{s}"] = {"e16": float(E["euler16"].mean()), "e512": float(E["euler512"].mean()),
                   "delta": {"mean": m, "median": float(np.median(d)), "ci": [lo, hi]},
                   "frac_e16_wins": float((d > 0).mean()),
                   "rel_degradation": float((E["euler512"].mean() - E["euler16"].mean())
                                            / E["euler16"].mean())}
    st[f"s{s}"]["_raw"] = d
    r = st[f"s{s}"]
    print(f"  s{s}: E16={r['e16']:.5f} E512={r['e512']:.5f} d={m:+.5f} [{lo:+.5f},{hi:+.5f}] "
          f"median={r['delta']['median']:+.5f} E16 wins {r['frac_e16_wins']*100:.1f}% "
          f"rel={r['rel_degradation']*100:+.2f}%")

print("\n=== 10/11/12. ACTION: recorded dataset action target (ARM-NEUTRAL) ===")
for s in SEEDS:
    ta = Z[f"s{s}_target_action"]                      # (96,3) recorded act[s]
    A = {a: Z[f"s{s}_{a}_act_unnorm"][:, :, 0] for a in ("euler16", "euler512")}
    res = {}
    for a in ("euler16", "euler512"):
        diff = A[a] - ta[:, None, :]
        res[a] = {"l1": np.abs(diff).sum(-1), "l2": np.linalg.norm(diff, axis=-1),
                  "mag": np.abs(np.linalg.norm(A[a], axis=-1) - np.linalg.norm(ta, axis=-1)[:, None])}
        nz = np.linalg.norm(ta, axis=-1) > 0.1
        cs = (A[a] @ ta[:, :, None]).squeeze(-1) / (
            np.linalg.norm(A[a], axis=-1) * np.linalg.norm(ta, axis=-1)[:, None] + 1e-9)
        res[a]["cos"] = cs[nz]
    d2 = (res["euler512"]["l2"] - res["euler16"]["l2"]).ravel()
    m, lo, hi = boot(d2)
    ac[f"s{s}"] = {
        "l2_e16": float(res["euler16"]["l2"].mean()), "l2_e512": float(res["euler512"]["l2"].mean()),
        "l1_e16": float(res["euler16"]["l1"].mean()), "l1_e512": float(res["euler512"]["l1"].mean()),
        "magerr_e16": float(res["euler16"]["mag"].mean()), "magerr_e512": float(res["euler512"]["mag"].mean()),
        "cos_e16": float(res["euler16"]["cos"].mean()), "cos_e512": float(res["euler512"]["cos"].mean()),
        "delta_l2": {"mean": m, "median": float(np.median(d2)), "ci": [lo, hi]},
        "frac_e16_wins": float((d2 > 0).mean()),
        "rel_degradation": float((res["euler512"]["l2"].mean() - res["euler16"]["l2"].mean())
                                 / res["euler16"]["l2"].mean())}
    ac[f"s{s}"]["_raw"] = d2
    r = ac[f"s{s}"]
    print(f"  s{s}: L2 E16={r['l2_e16']:.5f} E512={r['l2_e512']:.5f} d={m:+.5f} [{lo:+.5f},{hi:+.5f}] "
          f"E16 wins {r['frac_e16_wins']*100:.1f}% rel={r['rel_degradation']*100:+.2f}%")
    print(f"        L1 {r['l1_e16']:.5f}/{r['l1_e512']:.5f} | |mag|err {r['magerr_e16']:.5f}/{r['magerr_e512']:.5f} "
          f"| cos {r['cos_e16']:.4f}/{r['cos_e512']:.4f}")

print("\n=== 13. ARM-NEUTRAL ACTION/STATE SENSITIVITY RATIO ===")
for s in SEEDS:
    fs, fa = st[f"s{s}"]["rel_degradation"], ac[f"s{s}"]["rel_degradation"]
    ratio = fa / fs if abs(fs) > 1e-9 else float("nan")
    print(f"  s{s}: state {fs*100:+.2f}%  action {fa*100:+.2f}%  ratio = {ratio:.2f}x")

print("\n=== 14. JOINT STATE/ACTION CO-OCCURRENCE ===")
joint = {}
for s in SEEDS:
    ds, da = st[f"s{s}"]["_raw"], ac[f"s{s}"]["_raw"]
    rho, p = spearmanr(ds, da)
    joint[f"s{s}"] = {"spearman": [float(rho), float(p)],
                      "both_improve": float(((ds < 0) & (da < 0)).mean()),
                      "state_improves_action_worsens": float(((ds < 0) & (da > 0)).mean()),
                      "state_worsens_action_improves": float(((ds > 0) & (da < 0)).mean()),
                      "both_worsen": float(((ds > 0) & (da > 0)).mean())}
    j = joint[f"s{s}"]
    print(f"  s{s}: rho={rho:+.4f} (p={p:.2g}) | both improve {j['both_improve']*100:.1f}% "
          f"| both worsen {j['both_worsen']*100:.1f}% | S+A- {j['state_improves_action_worsens']*100:.1f}% "
          f"| S-A+ {j['state_worsens_action_improves']*100:.1f}%")

for k in (st, ac):
    for s in SEEDS:
        k[f"s{s}"].pop("_raw")
out["state"] = st; out["action"] = ac; out["joint"] = joint
json.dump(out, open("experiments/loss_balance_audit/arm_neutral_analysis.json", "w"), indent=2)
print("\nwrote arm_neutral_analysis.json")
