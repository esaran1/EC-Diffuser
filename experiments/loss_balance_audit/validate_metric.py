"""PHASE 4: sanity-validate the metric on controlled perturbations of REAL states.

If the metric fails these, it must not be used to judge imagination quality.
CPU only.
"""
import pickle, numpy as np, sys
sys.path.insert(0, "experiments/loss_balance_audit")
from latent_metric import training_stats, block_errors, chamfer_position, hungarian_match

rng = np.random.RandomState(0)
d = pickle.load(open("ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl", "rb"))
O = d["observations"][:400].reshape(-1, 48, 10)
mu, sd = training_stats(O)
states = O[rng.choice(len(O), 60, replace=False)]

def report(name, fn):
    pos, cham = [], []
    for s in states:
        t = fn(s.copy())
        e = block_errors(s, t, mu, sd)
        pos.append(e["pos"]); cham.append(chamfer_position(s, t))
    print(f"  {name:34s} pos_z={np.mean(pos):8.4f}   chamfer={np.mean(cham):8.5f}")
    return np.mean(pos), np.mean(cham)

print("=== PHASE 4: METRIC SANITY CHECKS (real DLP states, controlled perturbations) ===\n")
print("  TEST 1 — permutation invariance (must be ~0)")
ident = report("identity", lambda s: s)
perm  = report("random particle permutation", lambda s: s[rng.permutation(48)])

print("\n  TEST 2 — monotonicity in true displacement magnitude")
res = {}
for eps in (0.001, 0.005, 0.02, 0.08, 0.30):
    res[eps] = report(f"position jitter eps={eps}", lambda s, e=eps: np.concatenate(
        [s[:, :2] + rng.randn(48, 2) * e, s[:, 2:]], axis=1))

print("\n  TEST 3 — appearance-only perturbation (pos block must stay ~0)")
report("visual-feature jitter (pos unchanged)", lambda s: np.concatenate(
    [s[:, :5], s[:, 5:9] + rng.randn(48, 4) * 0.5, s[:, 9:]], axis=1))

print("\n  TEST 4 — particle collapse (all particles to their centroid)")
report("collapse to centroid", lambda s: np.concatenate(
    [np.repeat(s[:, :2].mean(0, keepdims=True), 48, 0), s[:, 2:]], axis=1))

print("\n=== VERDICT ===")
ok_perm = perm[0] < 0.05 and perm[1] < 0.01
mono = all(res[a][1] < res[b][1] for a, b in zip([0.001,0.005,0.02,0.08],[0.005,0.02,0.08,0.30]))
print(f"  permutation invariant (pos_z<0.05 and chamfer<0.01): {ok_perm}")
print(f"  chamfer monotone in displacement:                    {mono}")
print(f"  => metric is {'USABLE' if (ok_perm and mono) else 'NOT USABLE'}")
