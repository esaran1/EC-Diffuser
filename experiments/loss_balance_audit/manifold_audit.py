"""Coarse-Euler manifold audit. CPU-only analysis on cached endpoints.

Primary metric: permutation-invariant Chamfer nearest-neighbour distance on the
particle POSITION block (the validated metric), from a query latent to a
reference database of REAL training-distribution DLP latents.

Reference/held-out split is DISJOINT by episode so held-out real queries are
never present in the reference database.
"""
import hashlib, json, pickle, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "experiments/loss_balance_audit")
from latent_metric import POS, SCALE, DEPTH, VIS, TRANSP, BLOCKS

DATA = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
REF_EPISODES = (0, 1500)      # reference database
HELDOUT_EPISODES = (1500, 2000)  # held-out real calibration queries
N_REF = 4000
N_PER_VIEW = 24
N_HELDOUT = 288
KS = (1, 5)                   # predeclared, not tuned
SEEDS = (42, 43, 44)
RNG_SEED = 20260829


def chamfer_pos_batch(q, R):
    """Symmetric position Chamfer from one query (48,10) to every ref in R (M,48,10)."""
    a = q[None, :, None, POS]          # (1,48,1,2)
    b = R[:, None, :, POS]             # (M,1,48,2)
    d = np.linalg.norm(a - b, axis=-1)  # (M,48,48)
    return 0.5 * (d.min(axis=2).mean(axis=1) + d.min(axis=1).mean(axis=1))


def knn(queries, R, ks, chunk=500):
    out = {k: [] for k in ks}
    for i, q in enumerate(queries):
        ds = np.concatenate([chamfer_pos_batch(q, R[j:j + chunk])
                             for j in range(0, len(R), chunk)])
        srt = np.sort(ds)
        for k in ks:
            out[k].append(float(srt[:k].mean()))
    return {k: np.array(v) for k, v in out.items()}


def boot(d, rng, n=20000):
    d = np.asarray(d); i = rng.integers(0, len(d), (n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rng = np.random.default_rng(RNG_SEED)
    obs = pickle.load(open(DATA, "rb"))["observations"]
    out = {"reference_dataset": {
        "path": DATA, "full_shape": list(obs.shape),
        "reference_episodes": list(REF_EPISODES), "heldout_episodes": list(HELDOUT_EPISODES),
        "n_reference": N_REF, "n_heldout_queries": N_HELDOUT,
        "sampling": "uniform without replacement over (episode, timestep) within the split",
        "rng_seed": RNG_SEED,
        "sha256_observations": hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest(),
        "ks": list(KS)}}

    # The cached query endpoints are VIEW 0 only (24 particles), because the arms
    # extract [:, horizon=1, view=0]. The reference database must be the same view,
    # or a 24-particle query would be matched against a 48-particle two-view cloud.
    ref_pool = obs[REF_EPISODES[0]:REF_EPISODES[1], :, :N_PER_VIEW].reshape(-1, N_PER_VIEW, 10)
    ho_pool = obs[HELDOUT_EPISODES[0]:HELDOUT_EPISODES[1], :, :N_PER_VIEW].reshape(-1, N_PER_VIEW, 10)
    R = ref_pool[rng.choice(len(ref_pool), N_REF, replace=False)]
    HO = ho_pool[rng.choice(len(ho_pool), N_HELDOUT, replace=False)]
    print(f"[ref] {R.shape} from episodes {REF_EPISODES}; heldout {HO.shape} from {HELDOUT_EPISODES}",
          flush=True)

    # ---------- PHASE 8: is flattened Mahalanobis valid? ----------
    F = ref_pool[rng.choice(len(ref_pool), 6000, replace=False)].reshape(-1, N_PER_VIEW * 10)
    C = np.cov(F, rowvar=False)
    ev = np.linalg.eigvalsh(C); ev = np.clip(ev, 0, None)
    cond = float(ev.max() / max(ev[ev > 0].min(), 1e-30))
    p = ev / ev.sum(); eff_rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    rank = int(np.linalg.matrix_rank(C))
    Cinv = np.linalg.pinv(C)
    mu = F.mean(0)
    def maha(v):
        d = v - mu
        return float(np.sqrt(max(d @ Cinv @ d, 0)))
    # permutation sensitivity: permute particles WITHIN each view (24 each)
    samp = ref_pool[rng.choice(len(ref_pool), 40, replace=False)]
    base, perm = [], []
    for s in samp:
        base.append(maha(s.reshape(-1)))
        perm.append(maha(s[rng.permutation(N_PER_VIEW)].reshape(-1)))
    base, perm = np.array(base), np.array(perm)
    rel = float(np.mean(np.abs(perm - base) / np.maximum(base, 1e-9)))
    out["mahalanobis_validation"] = {
        "condition_number": cond, "matrix_rank": rank, "dim": N_PER_VIEW * 10,
        "effective_rank_entropy": eff_rank,
        "mean_relative_score_change_under_particle_permutation": rel,
        "mean_abs_change": float(np.mean(np.abs(perm - base))),
        "mean_base_score": float(base.mean()),
        "verdict": ("INVALID - particle permutation changes the score materially"
                    if rel > 0.05 else "permutation-insensitive")}
    print("[maha]", json.dumps(out["mahalanobis_validation"], indent=1), flush=True)

    # ---------- load cached endpoints ----------
    Z = np.load("experiments/loss_balance_audit/cached_endpoints.npz")
    arms = {}
    for s in SEEDS:
        for a in ["euler8", "euler16", "euler512"]:
            arms[f"s{s}_{a}"] = Z[f"s{s}_{a}"]
        arms[f"s{s}_copy"] = Z[f"s{s}_current"]
        arms[f"s{s}_realfut"] = Z[f"s{s}_real_future"]
    arms["gaussian100"] = Z["gauss_gaussian100"]

    # ---------- PHASE 5/6: kNN manifold distance ----------
    res = {}
    print("[knn] held-out real ...", flush=True)
    res["heldout_real"] = {str(k): v.tolist() for k, v in knn(HO, R, KS).items()}
    for name, Q in arms.items():
        if name.endswith("realfut"):
            continue
        print(f"[knn] {name} ...", flush=True)
        res[name] = {str(k): v.tolist() for k, v in knn(Q, R, KS).items()}
    out["knn_distance"] = res

    # ---------- PHASE 9: robust range violations, per block ----------
    lo = np.percentile(ref_pool.reshape(-1, 10), 0.5, axis=0)
    hi = np.percentile(ref_pool.reshape(-1, 10), 99.5, axis=0)
    rv = {}
    for name, Q in list(arms.items()) + [("heldout_real", HO)]:
        f = Q.reshape(-1, 10)
        out_of = (f < lo) | (f > hi)
        rv[name] = {b: float(out_of[:, sl].mean()) for b, sl in BLOCKS.items()}
    out["range_violation_p0.5_p99.5"] = rv

    # ---------- PHASE 10/12: paired E512 - E16 ----------
    from latent_metric import chamfer_position
    # Ground-truth errors are recomputed WITHIN THIS RUN. Isaac Gym's DLP
    # observations are not bit-reproducible across processes (max |dchamfer| ~0.05
    # vs the earlier run), so cross-run per-sample pairing would be invalid.
    # Within one run every arm shares one x0 and one env trajectory, so pairing
    # here is exact.
    gt = {}
    for s_ in SEEDS:
        rf = arms[f"s{s_}_realfut"]
        for a in ["euler8", "euler16", "euler512"]:
            gt[f"s{s_}_{a}"] = [chamfer_position(arms[f"s{s_}_{a}"][i], rf[i])
                                for i in range(len(rf))]
    gt["gaussian100"] = [chamfer_position(arms["gaussian100"][i], Z["gauss_real_future"][i])
                         for i in range(len(arms["gaussian100"]))]
    for s_ in SEEDS:
        rf = arms[f"s{s_}_realfut"]
        gt[f"s{s_}_copy"] = [chamfer_position(arms[f"s{s_}_copy"][i], rf[i])
                             for i in range(len(rf))]
    out["ground_truth_error_within_run"] = {k: float(np.mean(v)) for k, v in gt.items()}
    paired = {}
    for k in KS:
        per_seed, pooled_dm, pooled_gt = {}, [], []
        for s in SEEDS:
            dm = (np.array(res[f"s{s}_euler512"][str(k)])
                  - np.array(res[f"s{s}_euler16"][str(k)]))
            dg = np.array(gt[f"s{s}_euler512"]) - np.array(gt[f"s{s}_euler16"])
            m, l, h = boot(dm, rng)
            per_seed[f"s{s}"] = {"delta_manifold_mean": m, "ci": [l, h],
                                 "delta_gt_mean": float(dg.mean())}
            pooled_dm.append(dm); pooled_gt.append(dg)
        dm = np.concatenate(pooled_dm); dg = np.concatenate(pooled_gt)
        m, l, h = boot(dm, rng)
        rho, pv = spearmanr(dm, dg)
        paired[f"k{k}"] = {"per_seed": per_seed,
                           "pooled_delta_manifold": {"mean": m, "ci": [l, h], "n": len(dm)},
                           "pooled_delta_gt_mean": float(dg.mean()),
                           "spearman_dManifold_vs_dGT": {"rho": float(rho), "p": float(pv)}}
    out["paired_E512_minus_E16"] = paired

    json.dump(out, open("experiments/loss_balance_audit/manifold_audit.json", "w"), indent=2)
    print("wrote manifold_audit.json")


if __name__ == "__main__":
    main()
