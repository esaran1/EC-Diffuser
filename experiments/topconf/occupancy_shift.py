"""Phase I: does NFE1 occupy a different closed-loop state distribution?

PRIMARY measure uses RAW PHYSICAL STATE (cube xy positions relative to goal,
plus EEF position), not DLP latents. Distance to the replay support is a kNN
distance in a standardized physical feature space (no ill-conditioned Mahalanobis).
"""
import json, os, sys
import numpy as np
import pickle

OUT = "experiments/topconf/selfstate"
DATA = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
K = 5
RNG = np.random.default_rng(20260911)


def phys_features(Z):
    """(n, d) physical features: cube xy (3 cubes) + eef xyz, goal-relative."""
    root = Z["root"]           # (n, n_actors, 13)
    eef = Z["eef"][:, :3]      # (n, 3)
    cubes = root[:, 1:4, :3]   # (n, 3 cubes, xyz)
    return np.concatenate([cubes.reshape(len(root), -1), eef], axis=1)


def knn_dist(Q, Rref, k=K, chunk=512):
    out = []
    for i in range(0, len(Q), chunk):
        d = np.linalg.norm(Q[i:i+chunk, None, :] - Rref[None, :, :], axis=-1)
        out.append(np.sort(d, axis=1)[:, :k].mean(1))
    return np.concatenate(out)


def main():
    seed = 42
    res = {"protocol": {"feature": "raw physical: 3 cube xyz + eef xyz (9+3=12 dims)",
                        "metric": f"mean distance to {K} nearest reference states",
                        "reference": "states visited by NFE4 policy (in-support proxy)"}}
    dists = {}
    for n in (1, 2, 4):
        f = os.path.join(OUT, f"s{seed}_nfe{n}_states.npz")
        if not os.path.exists(f):
            print(f"[warn] missing {f}"); return
        Z = np.load(f, allow_pickle=True)
        dists[n] = {"F": phys_features(Z), "step": Z["step"], "ep": Z["episode"],
                    "succ": json.loads(str(Z["_success"]))}
        print(f"  nfe{n}: {dists[n]['F'].shape} physical states")

    # Reference support = NFE4-visited states (the best-performing arm), subsampled
    ref = dists[4]["F"]
    sub = ref[RNG.choice(len(ref), min(3000, len(ref)), replace=False)]

    print(f"\n{'arm':>6s} {'n':>7s} {'kNN dist to NFE4 support':>26s}")
    for n in (1, 2, 4):
        F = dists[n]["F"]
        s = F[RNG.choice(len(F), min(3000, len(F)), replace=False)]
        d = knn_dist(s, sub)
        res[f"nfe{n}"] = {"knn_mean": float(d.mean()), "knn_median": float(np.median(d)),
                          "knn_p90": float(np.percentile(d, 90))}
        print(f"{'nfe'+str(n):>6s} {len(s):7d} {d.mean():26.5f}")
        # stratify by phase
        st = dists[n]["step"][RNG.choice(len(F), min(3000, len(F)), replace=False)] \
             if False else None
    # phase-stratified, computed consistently
    print(f"\n{'arm':>6s} {'early<20':>10s} {'mid20-60':>10s} {'late>=60':>10s}")
    for n in (1, 2, 4):
        F = dists[n]["F"]; stp = dists[n]["step"]
        row = []
        for lab, m in [("early", stp < 20), ("mid", (stp >= 20) & (stp < 60)), ("late", stp >= 60)]:
            if m.sum() < 50: row.append(float("nan")); continue
            q = F[m][RNG.choice(int(m.sum()), min(1200, int(m.sum())), replace=False)]
            row.append(float(knn_dist(q, sub).mean()))
            res.setdefault(f"nfe{n}", {})[f"knn_{lab}"] = row[-1]
        print(f"{'nfe'+str(n):>6s} " + " ".join(f"{v:10.5f}" for v in row))
    json.dump(res, open("experiments/topconf/occupancy_shift.json", "w"), indent=2)
    print("\nwrote occupancy_shift.json")


if __name__ == "__main__":
    main()
