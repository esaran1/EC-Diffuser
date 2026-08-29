"""CPU analysis of the enriched multi-noise cache. No GPU."""
import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
sys.path.insert(0, "experiments/loss_balance_audit")
from latent_metric import POS, chamfer_position

SEEDS = (42, 43, 44); N_NOISE = 8; EPS = 1e-9
rng = np.random.default_rng(777)
Z = np.load("experiments/loss_balance_audit/enriched_endpoints.npz", allow_pickle=True)
out = {}


def boot(d, n=20000):
    d = np.asarray(d); i = rng.integers(0, len(d), (n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def view0(x):
    return x[..., 0, :, :] if x.ndim == 4 else x


def gen_t1(seed, arm):
    """generated timestep-1, view 0 particles: (96,8,24,10)"""
    o = Z[f"s{seed}_{arm}_obs_unnorm"]          # (96,8,5,480)
    return o[:, :, 1].reshape(o.shape[0], N_NOISE, 2, 24, 10)[:, :, 0]


def cham(a, b):
    return chamfer_position(a, b)


# ---------- 1. aggregate replication ----------
print("=== 1. AGGREGATE REPLICATION ===")
rep = {}
for s in SEEDS:
    R = view0(Z[f"s{s}_real_t1_raw"])
    e16, e512 = gen_t1(s, "euler16"), gen_t1(s, "euler512")
    E = {a: np.array([[cham(g[m, i], R[m]) for i in range(N_NOISE)] for m in range(len(R))])
         for a, g in [("euler16", e16), ("euler512", e512)]}
    d = (E["euler512"] - E["euler16"]).ravel()
    m, lo, hi = boot(d)
    disp, med = {}, {}
    for a, g in [("euler16", e16), ("euler512", e512)]:
        ds, mm = [], []
        for mi in range(len(R)):
            D = np.zeros((N_NOISE, N_NOISE))
            for i in range(N_NOISE):
                for j in range(i + 1, N_NOISE):
                    D[i, j] = D[j, i] = cham(g[mi, i], g[mi, j])
            ds.append(D[np.triu_indices(N_NOISE, 1)].mean())
            mm.append(E[a][mi, int(np.argmin(D.sum(1)))])
        disp[a] = float(np.mean(ds)); med[a] = float(np.mean(mm))
    rep[f"s{s}"] = {"mean_e16": float(E["euler16"].mean()), "mean_e512": float(E["euler512"].mean()),
                    "paired_delta": {"mean": m, "ci": [lo, hi]},
                    "best_of_8_e16": float(E["euler16"].min(1).mean()),
                    "best_of_8_e512": float(E["euler512"].min(1).mean()),
                    "dispersion_e16": disp["euler16"], "dispersion_e512": disp["euler512"],
                    "medoid_e16": med["euler16"], "medoid_e512": med["euler512"],
                    "frac_favouring_e16": float((d > 0).mean())}
    r = rep[f"s{s}"]
    print(f"  s{s}: E16={r['mean_e16']:.5f} E512={r['mean_e512']:.5f} d={m:+.5f}[{lo:+.5f},{hi:+.5f}] "
          f"| best8 {r['best_of_8_e16']:.5f}/{r['best_of_8_e512']:.5f} "
          f"| disp {r['dispersion_e16']:.5f}/{r['dispersion_e512']:.5f} "
          f"| medoid {r['medoid_e16']:.5f}/{r['medoid_e512']:.5f} | E16 wins {r['frac_favouring_e16']*100:.1f}%")
out["replication"] = rep

# ---------- 4-6, 11. distance table ----------
print("\n=== 4/5/6/11. DISTANCE TABLE (permutation-invariant position chamfer) ===")
dt = {}
for s in SEEDS:
    cur = view0(Z[f"s{s}_cur_raw"]); gl = view0(Z[f"s{s}_goal_raw"])
    R = view0(Z[f"s{s}_real_t1_raw"])
    row = {}
    for a in ("euler16", "euler512"):
        g = gen_t1(s, a)
        row[a] = {ref: float(np.mean([[cham(g[m, i], T[m]) for i in range(N_NOISE)]
                                      for m in range(len(R))]))
                  for ref, T in [("current", cur), ("observed_t1", R), ("goal", gl)]}
    row["observed_t1_state"] = {"current": float(np.mean([cham(R[m], cur[m]) for m in range(len(R))])),
                                "observed_t1": 0.0,
                                "goal": float(np.mean([cham(R[m], gl[m]) for m in range(len(R))]))}
    row["current_state"] = {"current": 0.0,
                            "observed_t1": row["observed_t1_state"]["current"],
                            "goal": float(np.mean([cham(cur[m], gl[m]) for m in range(len(R))]))}
    # paired deltas
    row["paired_E512_minus_E16"] = {}
    for ref, T in [("current", cur), ("observed_t1", R), ("goal", gl)]:
        d = np.array([[cham(gen_t1(s, "euler512")[m, i], T[m]) - cham(gen_t1(s, "euler16")[m, i], T[m])
                       for i in range(N_NOISE)] for m in range(len(R))]).ravel()
        mm, lo, hi = boot(d)
        row["paired_E512_minus_E16"][ref] = {"mean": mm, "ci": [lo, hi],
                                             "frac_E512_closer": float((d < 0).mean())}
    dt[f"s{s}"] = row
    print(f"  s{s}:")
    for a in ("current_state", "observed_t1_state", "euler16", "euler512"):
        r = row[a]
        print(f"    {a:18s} ->cur {r['current']:.5f}  ->obs_t1 {r['observed_t1']:.5f}  ->goal {r['goal']:.5f}")
    for ref, v in row["paired_E512_minus_E16"].items():
        print(f"    paired d(E512)-d(E16) ->{ref:12s}: {v['mean']:+.5f} [{v['ci'][0]:+.5f},{v['ci'][1]:+.5f}]"
              f"  E512 closer in {v['frac_E512_closer']*100:.1f}%")
out["distance_table"] = dt

# ---------- 7/12/13. normalized goal progress ----------
print("\n=== 7/12/13. NORMALIZED GOAL PROGRESS  1 - d(x,goal)/d(current,goal) ===")
gp = {}
for s in SEEDS:
    cur = view0(Z[f"s{s}_cur_raw"]); gl = view0(Z[f"s{s}_goal_raw"])
    R = view0(Z[f"s{s}_real_t1_raw"])
    base = np.array([cham(cur[m], gl[m]) for m in range(len(R))])
    prog = {}
    prog["observed_t1"] = float(np.mean([1 - cham(R[m], gl[m]) / max(base[m], EPS)
                                         for m in range(len(R))]))
    for a in ("euler16", "euler512"):
        g = gen_t1(s, a)
        prog[a] = float(np.mean([[1 - cham(g[m, i], gl[m]) / max(base[m], EPS)
                                  for i in range(N_NOISE)] for m in range(len(R))]))
    gp[f"s{s}"] = prog
    print(f"  s{s}: observed_t1={prog['observed_t1']:+.4f}  E16={prog['euler16']:+.4f}  "
          f"E512={prog['euler512']:+.4f}   overshoot(E512-obs)={prog['euler512']-prog['observed_t1']:+.4f}"
          f"  (E16-obs)={prog['euler16']-prog['observed_t1']:+.4f}")
out["goal_progress"] = gp

# ---------- 9/10/11. direction cosines with anchor sensitivity ----------
def match(a, b):
    c = np.linalg.norm(a[:, None, POS] - b[None, :, POS], axis=-1)
    return linear_sum_assignment(c)[1]


print("\n=== 9/10. DIRECTION COSINES (E16-anchored primary) ===")
dirs = {}
for anchor in ("e16", "e512"):
    dirs[anchor] = {}
    for s in SEEDS:
        cur = view0(Z[f"s{s}_cur_raw"]); gl = view0(Z[f"s{s}_goal_raw"])
        R = view0(Z[f"s{s}_real_t1_raw"])
        A, B = gen_t1(s, "euler16"), gen_t1(s, "euler512")
        acc = {k: [] for k in ("future", "goal", "current")}
        fr = {k: [] for k in acc}
        for m in range(len(R)):
            for i in range(N_NOISE):
                a, b = A[m, i], B[m, i]
                anc = a if anchor == "e16" else b
                d = (b[:, POS] - a[:, POS]).ravel()
                nd = np.linalg.norm(d)
                if nd < 1e-8:
                    continue
                for k, T in [("future", R[m]), ("goal", gl[m]), ("current", cur[m])]:
                    tg = T[match(anc, T)]
                    g = (tg[:, POS] - a[:, POS]).ravel()
                    ng = np.linalg.norm(g)
                    if ng < 1e-8:
                        continue
                    gh = g / ng; p = float(d @ gh)
                    acc[k].append(p / nd); fr[k].append(abs(p) / nd)
        dirs[anchor][f"s{s}"] = {k: {"cos_mean": float(np.mean(v)),
                                     "cos_median": float(np.median(v)),
                                     "frac_positive": float((np.array(v) > 0).mean()),
                                     "frac_of_shift_on_axis": float(np.mean(fr[k]))}
                                 for k, v in acc.items()}
        if anchor == "e16":
            r = dirs[anchor][f"s{s}"]
            print(f"  s{s}: cos_future={r['future']['cos_mean']:+.4f}  "
                  f"cos_goal={r['goal']['cos_mean']:+.4f}  cos_current={r['current']['cos_mean']:+.4f}"
                  f"   |para|/|d|: fut {r['future']['frac_of_shift_on_axis']:.3f} "
                  f"goal {r['goal']['frac_of_shift_on_axis']:.3f} cur {r['current']['frac_of_shift_on_axis']:.3f}")
print("  anchor sensitivity (E512-anchored):")
for s in SEEDS:
    r = dirs["e512"][f"s{s}"]
    print(f"  s{s}: cos_future={r['future']['cos_mean']:+.4f}  cos_goal={r['goal']['cos_mean']:+.4f}  "
          f"cos_current={r['current']['cos_mean']:+.4f}")
out["direction_cosines"] = dirs

# ---------- 12/13/14. ACTION analysis ----------
print("\n=== 12/13/14. ACTION CHANNEL ===")
act = {}
for s in SEEDS:
    a16 = Z[f"s{s}_euler16_act_unnorm"][:, :, 0]     # (96,8,3) action at generated t=0
    a512 = Z[f"s{s}_euler512_act_unnorm"][:, :, 0]
    demo = Z[f"s{s}_real_action"]                     # (96,3) the action actually executed
    e16 = np.linalg.norm(a16 - demo[:, None, :], axis=-1)
    e512 = np.linalg.norm(a512 - demo[:, None, :], axis=-1)
    d = (e512 - e16).ravel(); m, lo, hi = boot(d)
    disp_a = np.linalg.norm(a512 - a16, axis=-1)
    # normalized state vs action displacement (per-coordinate RMS / data scale)
    o16 = Z[f"s{s}_euler16_obs_unnorm"][:, :, 1]; o512 = Z[f"s{s}_euler512_obs_unnorm"][:, :, 1]
    st_rms = np.sqrt(((o512 - o16) ** 2).mean(-1))
    st_scale = Z[f"s{s}_euler16_obs_unnorm"][:, :, 1].std()
    ac_rms = np.sqrt(((a512 - a16) ** 2).mean(-1))
    ac_scale = a16.std()
    act[f"s{s}"] = {"action_err_e16": float(e16.mean()), "action_err_e512": float(e512.mean()),
                    "paired_delta": {"mean": m, "ci": [lo, hi]},
                    "action_displacement_mean": float(disp_a.mean()),
                    "action_magnitude_e16": float(np.linalg.norm(a16, axis=-1).mean()),
                    "action_magnitude_e512": float(np.linalg.norm(a512, axis=-1).mean()),
                    "demo_action_magnitude": float(np.linalg.norm(demo, axis=-1).mean()),
                    "normalized_state_displacement": float((st_rms / st_scale).mean()),
                    "normalized_action_displacement": float((ac_rms / ac_scale).mean())}
    r = act[f"s{s}"]
    print(f"  s{s}: |a-demo| E16={r['action_err_e16']:.5f} E512={r['action_err_e512']:.5f} "
          f"d={m:+.5f}[{lo:+.5f},{hi:+.5f}]")
    print(f"        normalized displacement: state={r['normalized_state_displacement']:.4f}  "
          f"action={r['normalized_action_displacement']:.4f}  "
          f"ratio state/action = {r['normalized_state_displacement']/max(r['normalized_action_displacement'],EPS):.2f}x")
out["action"] = act

json.dump(out, open("experiments/loss_balance_audit/enriched_analysis.json", "w"), indent=2)
print("\nwrote enriched_analysis.json")
