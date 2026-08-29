"""Paired analysis + figures for the matched-NFE solver study."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("experiments/loss_balance_audit/solver_study.json"))
R, RD, DG = D["results"], D["reference_distance"], D["vector_field_diagnostics"]
SEEDS = [42, 43, 44]
ARMS = [("euler",2,2),("euler",4,4),("euler",8,8),("euler",16,16),
        ("midpoint",1,2),("midpoint",2,4),("midpoint",4,8),
        ("heun",1,2),("heun",2,4),("heun",4,8)]
rng = np.random.default_rng(0)

def boot(d, n=10000):
    d = np.asarray(d); idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1); return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

print("="*78); print("REFERENCE CONVERGENCE (RK4@64 vs independent midpoint@128, both NFE 256)")
rc = np.array([r["d"] for r in D["reference_convergence"]])
print(f"  n={len(rc)}  mean={rc.mean():.5f}  median={np.median(rc):.5f}  max={rc.max():.5f}")

print("\n" + "="*78); print("IMPLEMENTATION VALIDATION")
for k, v in D["implementation_validation"].items():
    print(f"  {k}: bit-identity={v['euler_bit_identity_max_abs_diff']:.3e} "
          f"determinism={v['euler_determinism_max_abs_diff']:.3e} nfe(euler@4)={v['euler4_nfe_used']}")

def table(metric, store, title):
    print("\n" + "="*78); print(title)
    print(f"{'arm':18s} {'NFE':>4s} " + " ".join(f"{'s'+str(s):>9s}" for s in SEEDS) + f" {'mean':>9s} {'sd':>8s}")
    out = {}
    for meth, nst, nfe in ARMS:
        vals = [np.mean(store[f"flow_s{s}_{meth}{nst}"] if metric=="ref"
                        else R[f"flow_s{s}_{meth}{nst}"]["chamfer"]) for s in SEEDS]
        out[(meth,nfe)] = vals
        print(f"{meth+'@'+str(nst):18s} {nfe:4d} " + " ".join(f"{v:9.5f}" for v in vals)
              + f" {np.mean(vals):9.5f} {np.std(vals,ddof=1):8.5f}")
    return out

gt  = table("gt",  R,  "A. GROUND-TRUTH IMAGINATION ERROR (chamfer position, lower=better)")
ref = table("ref", RD, "B. NUMERICAL ODE ERROR (latent distance to RK4@64 reference)")

print("\n" + "="*78); print("REFERENCE ENDPOINT GROUND-TRUTH ERROR (Phase 12)")
rv = [np.mean(D["reference_chamfer"][f"flow_s{s}"]) for s in SEEDS]
print(f"  RK4@64 (256 NFE) per seed: " + " ".join(f"{v:.5f}" for v in rv))
print(f"  three-seed mean = {np.mean(rv):.5f}  sd = {np.std(rv,ddof=1):.5f}")
print(f"  Gaussian@100 = 0.04004   Euler@16 = {np.mean(gt[('euler',16)]):.5f}   copy-current = 0.07484")
print(f"  residual (Flow accurate-reference - Gaussian) = {np.mean(rv)-0.04004:+.5f}")

print("\n" + "="*78); print("PAIRED MATCHED-NFE EFFECTS vs EULER (per seed, 95% bootstrap CI)")
print("negative = solver BEATS Euler at the same NFE")
for nfe, e_st, m_st, h_st in [(2,2,1,1),(4,4,2,2),(8,8,4,4)]:
    print(f"\n  --- NFE {nfe} ---")
    for name, st in [("midpoint", m_st), ("heun", h_st)]:
        for s in SEEDS:
            for lab, store, key in [("gt", R, "chamfer"), ("ref", RD, None)]:
                a = store[f"flow_s{s}_{name}{st}"]; b = store[f"flow_s{s}_euler{e_st}"]
                a = a[key] if key else a; b = b[key] if key else b
                d = np.array(a) - np.array(b); m, lo, hi = boot(d)
                print(f"    {name}@{st} - euler@{e_st}  s{s} {lab:3s}: {m:+.5f} [{lo:+.5f},{hi:+.5f}]")

print("\n" + "="*78); print("VECTOR-FIELD DIAGNOSTICS (seed 42)")
print(f"{'arm':18s} {'|update|':>10s} {'cos(v_k,v_k-1)':>15s} {'rel |v2-v1|/|v1|':>18s}")
for meth, nst, nfe in ARMS:
    d = DG[f"flow_s42_{meth}{nst}"]
    f = lambda v: f"{v:10.4f}" if v is not None else f"{'--':>10s}"
    print(f"{meth+'@'+str(nst):18s} {f(d['mean_update_norm'])} "
          f"{d['mean_cos_consecutive'] if d['mean_cos_consecutive'] is not None else float('nan'):15.4f} "
          f"{d['mean_rel_v_change'] if d['mean_rel_v_change'] is not None else float('nan'):18.4f}")

# ---------------- figures ----------------
C = {"euler":"#1f77b4","midpoint":"#2ca02c","heun":"#d62728"}
fig, ax = plt.subplots(figsize=(6.4,4.4))
for meth in ["euler","midpoint","heun"]:
    xs = sorted({nfe for m,n,nfe in ARMS if m==meth})
    ys = [np.mean(gt[(meth,x)]) for x in xs]
    er = [np.std(gt[(meth,x)],ddof=1) for x in xs]
    ax.errorbar(xs, ys, yerr=er, marker="o", color=C[meth], label=meth, capsize=3)
ax.axhline(0.04004, ls="--", c="k", label="Gaussian@100")
ax.axhline(np.mean(rv), ls=":", c="purple", label="Flow RK4@64 ref (256 NFE)")
ax.axhline(0.07484, ls="-.", c="gray", label="copy-current")
ax.set_xscale("log", base=2); ax.set_xticks([2,4,8,16]); ax.set_xticklabels([2,4,8,16])
ax.set_xlabel("NFE (velocity-network calls)"); ax.set_ylabel("imagination error (chamfer)")
ax.set_title("Fig 1: ground-truth imagination error vs NFE\n(mean of 3 Flow seeds, bars = seed sd)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("experiments/figures/solver_gt_error.png", dpi=160)

fig, ax = plt.subplots(figsize=(6.4,4.4))
for meth in ["euler","midpoint","heun"]:
    xs = sorted({nfe for m,n,nfe in ARMS if m==meth})
    ys = [np.mean(ref[(meth,x)]) for x in xs]
    ax.plot(xs, ys, marker="o", color=C[meth], label=meth)
ax.axhline(rc.mean(), ls="--", c="gray", label="reference uncertainty floor")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([2,4,8,16]); ax.set_xticklabels([2,4,8,16])
ax.set_xlabel("NFE"); ax.set_ylabel("distance to RK4@64 reference")
ax.set_title("Fig 2: numerical ODE error vs NFE")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("experiments/figures/solver_ref_error.png", dpi=160)

fig, axes = plt.subplots(1, 2, figsize=(9.5,4.2))
labs = ["euler@4","midpoint@2","heun@2"]
for ax_, store, ttl in [(axes[0], gt, "ground-truth imagination"), (axes[1], ref, "numerical ODE error")]:
    vals = [store[("euler",4)], store[("midpoint",4)], store[("heun",4)]]
    ax_.bar(labs, [np.mean(v) for v in vals],
            yerr=[np.std(v,ddof=1) for v in vals],
            color=[C["euler"],C["midpoint"],C["heun"]], capsize=4)
    for i,v in enumerate(vals):
        for s_i,s in enumerate(SEEDS):
            ax_.plot(i, v[s_i], "k.", ms=5)
    ax_.set_title(f"{ttl}\nall arms = exactly 4 NFE"); ax_.set_ylabel("error")
    if ax_ is axes[0]: ax_.axhline(0.04004, ls="--", c="k", lw=1)
fig.suptitle("Fig 3: matched-compute comparison at NFE 4", fontsize=11)
fig.tight_layout(); fig.savefig("experiments/figures/solver_matched_nfe4.png", dpi=160)
print("\nfigures written to experiments/figures/solver_*.png")
