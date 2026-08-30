"""Paired analysis of the NFE4 vs NFE32 closed-loop experiment."""
import json, os, sys
import numpy as np
from scipy.stats import binomtest

D = "experiments/isaacgym_control/nfe4_vs_nfe32"
SEEDS = (42, 43, 44)
rng = np.random.default_rng(20260904)
out = {}


def load(s, n):
    with open(os.path.join(D, f"r0_s{s}_nfe{n}.json")) as h:
        return json.load(h)


def boot(d, n=20000):
    d = np.asarray(d, float); i = rng.integers(0, len(d), (n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def succ(recs):
    k = "full_success" if "full_success" in recs[0] else "success"
    return np.array([float(r[k]) for r in recs]), k


print("=== 5/6. PER-SEED FULL EPISODE SUCCESS (n=96, paired) ===")
print(f"{'seed':>5s} {'NFE4':>14s} {'NFE32':>14s} {'delta':>9s} {'95% CI':>22s}")
rows, deltas = {}, []
for s in SEEDS:
    a, b = load(s, 4), load(s, 32)
    sa, key = succ(a["episodes"]); sb, _ = succ(b["episodes"])
    assert len(sa) == len(sb) == 96
    d = sb - sa
    m, lo, hi = boot(d)
    nb = int(((sa == 0) & (sb == 1)).sum())   # NFE4 fail -> NFE32 success
    nc = int(((sa == 1) & (sb == 0)).sum())   # NFE4 success -> NFE32 fail
    p = binomtest(nb, nb + nc, 0.5).pvalue if (nb + nc) else 1.0
    rows[s] = {"nfe4": float(sa.mean()), "nfe32": float(sb.mean()),
               "n4": int(sa.sum()), "n32": int(sb.sum()),
               "delta": m, "ci": [lo, hi], "mcnemar_b": nb, "mcnemar_c": nc,
               "mcnemar_p_exact": float(p), "success_key": key,
               "A_both_succeed": int(((sa == 1) & (sb == 1)).sum()),
               "B_4fail_32succeed": nb, "C_4succeed_32fail": nc,
               "D_both_fail": int(((sa == 0) & (sb == 0)).sum())}
    deltas.append(m)
    print(f"{s:5d} {int(sa.sum()):3d}/96={sa.mean():.4f} {int(sb.sum()):3d}/96={sb.mean():.4f} "
          f"{m:+9.4f} [{lo:+.4f},{hi:+.4f}]")
out["per_seed"] = rows

print(f"\n=== 12/16. THREE-SEED SUMMARY (N=3 checkpoints) ===")
print(f"  per-seed deltas: {[round(x,4) for x in deltas]}")
print(f"  signs: {['+' if x>0 else ('-' if x<0 else '0') for x in deltas]}")
print(f"  mean seed-level delta = {np.mean(deltas):+.4f}   SD = {np.std(deltas, ddof=1):.4f}")
print("  (N=3; no t-test reported - insufficient power)")
out["three_seed"] = {"deltas": [float(x) for x in deltas], "mean": float(np.mean(deltas)),
                     "sd": float(np.std(deltas, ddof=1))}

print("\n=== 11. PAIRED EPISODE TRANSITIONS ===")
print(f"{'seed':>5s} {'A both ok':>10s} {'B 4x->32ok':>11s} {'C 4ok->32x':>11s} {'D both x':>9s} {'McNemar p':>10s}")
for s in SEEDS:
    r = rows[s]
    print(f"{s:5d} {r['A_both_succeed']:10d} {r['B_4fail_32succeed']:11d} "
          f"{r['C_4succeed_32fail']:11d} {r['D_both_fail']:9d} {r['mcnemar_p_exact']:10.4f}")

print("\n=== 7/8/9/10. SECONDARY CANONICAL DIAGNOSTICS ===")
sec = {}
probe = load(SEEDS[0], 4)["episodes"][0]
cand = [k for k in probe if isinstance(probe[k], (int, float, bool)) and k not in ("full_success",)]
print(f"  available per-episode fields: {cand}")
for k in cand:
    per = {}
    for s in SEEDS:
        a = np.array([float(r[k]) for r in load(s, 4)["episodes"]])
        b = np.array([float(r[k]) for r in load(s, 32)["episodes"]])
        m, lo, hi = boot(b - a)
        per[f"s{s}"] = {"nfe4": float(a.mean()), "nfe32": float(b.mean()),
                        "delta": m, "ci": [lo, hi]}
    sec[k] = per
    v4 = np.mean([per[f"s{s}"]["nfe4"] for s in SEEDS])
    v32 = np.mean([per[f"s{s}"]["nfe32"] for s in SEEDS])
    print(f"  {k:26s} NFE4={v4:9.4f}  NFE32={v32:9.4f}  d={v32-v4:+9.4f}")
out["secondary"] = sec

print("\n=== 13. EXTENDED SEED-42 CONTROL CURVE ===")
canon = {1: 0.8056, 2: 0.8681, 4: 0.8889, 8: 0.8993, 16: 0.8854}
print("  NFE : success   (source)")
for n, v in canon.items():
    print(f"  {n:4d} : {v:.4f}   canonical n=288 (3 replicates)")
print(f"  {32:4d} : {rows[42]['nfe32']:.4f}   this experiment, n=96 (replicate0 only)")
print(f"  [{4:3d} : {rows[42]['nfe4']:.4f}   this experiment, n=96 - compare to 0.8889 at n=288]")
out["seed42_curve"] = {**{str(k): v for k, v in canon.items()},
                       "32": rows[42]["nfe32"], "4_thisrun_n96": rows[42]["nfe4"]}

print("\n=== 14. OFFLINE vs ONLINE CONTRAST (NFE4 -> NFE32) ===")
off = json.load(open("experiments/loss_balance_audit/nfe_curve_analysis.json"))["curve"]
a4, a32 = off["4"]["action_mean"], off["32"]["action_mean"]
s4, s32 = off["4"]["state_mean"], off["32"]["state_mean"]
print(f"  offline ACTION  {a4:.5f} -> {a32:.5f}  ({100*(a32-a4)/a4:+.1f}%)  WORSE")
print(f"  offline STATE   {s4:.5f} -> {s32:.5f}  ({100*(s32-s4)/s4:+.1f}%)  BETTER")
print(f"  closed-loop     {np.mean([rows[s]['nfe4'] for s in SEEDS]):.4f} -> "
      f"{np.mean([rows[s]['nfe32'] for s in SEEDS]):.4f}  "
      f"({np.mean(deltas):+.4f} absolute)")
out["offline_contrast"] = {"action_rel": float((a32-a4)/a4), "state_rel": float((s32-s4)/s4),
                           "control_delta_mean": float(np.mean(deltas))}

json.dump(out, open("experiments/loss_balance_audit/control_analysis.json", "w"), indent=2)
print("\nwrote control_analysis.json")
