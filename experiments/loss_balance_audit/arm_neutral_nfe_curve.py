"""Arm-neutral state/action error vs NFE. Replay data only.

NO Isaac Gym. NO training. NO model-authored targets. Identical replay
construction, sample set, indexing and seeds as ARM_NEUTRAL_SOLVER_BIAS; the
only change is that the NFE ladder is swept with the SAME x0 at every NFE.
"""
import argparse, hashlib, json, pickle, sys
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit")
import solvers
import diffuser.utils as utils

DATA = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
NPV = 24; HORIZON = 5; N_COND = 96
N_NOISE = 4                       # reduced from 8 to stay inside 0.5 GPU-h (cost reported)
NOISE_BANK_SEED = 20260901        # SAME bank seed as the arm-neutral study
SAMPLE_SEED = 20260902            # SAME sample selection
NFES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
SEEDS = {42: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
         43: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed43",
         44: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed44"}


def build_sample_set(sf):
    """Byte-identical rule to arm_neutral_eval.build_sample_set."""
    rng = np.random.default_rng(SAMPLE_SEED)
    ok = np.flatnonzero(np.asarray(sf) >= 1.0)
    eps = rng.choice(ok, N_COND, replace=False)
    ts = rng.integers(0, 100 - HORIZON, size=N_COND)
    return eps, ts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()
    raw = pickle.load(open(DATA, "rb"))
    obs, acts, goals, sf = raw["observations"], raw["actions"], raw["goals"], raw["info_goal_success_frac"]
    eps, ts = build_sample_set(sf)
    nfes = [1, 2] if cli.smoke else NFES
    nn = 1 if cli.smoke else N_NOISE
    nc = 4 if cli.smoke else N_COND
    eps, ts = eps[:nc], ts[:nc]

    utils.set_global_device("cuda:0")
    S = {"_protocol": json.dumps({
        "nfes": nfes, "n_noise": nn, "n_cond": nc,
        "noise_bank_seed": NOISE_BANK_SEED, "sample_seed": SAMPLE_SEED,
        "note": ("same x0 reused across every NFE within (seed, condition, noise); "
                 "targets are recorded replay transitions, no policy authored them")})}
    for seed, path in SEEDS.items():
        exp = utils.load_diffusion("data", "panda_push", path, epoch="latest", seed=42,
                                   is_diffusion=True, override_dataset_path=DATA)
        model = exp.ema; model.eval(); norm = exp.dataset.normalizer
        cur = obs[eps, ts]; t1 = obs[eps, ts + 1]; gl = goals[eps, 99]; ta = acts[eps, ts]
        cond = {0: torch.as_tensor(norm.normalize(cur.reshape(nc, -1), "observations"),
                                   device="cuda:0", dtype=torch.float32),
                4: torch.as_tensor(norm.normalize(gl.reshape(nc, -1), "observations"),
                                   device="cuda:0", dtype=torch.float32)}
        gen = torch.Generator(device="cpu").manual_seed(NOISE_BANK_SEED)
        st = {n: [] for n in nfes}; ac = {n: [] for n in nfes}
        hashes = []
        with torch.no_grad():
            for i in range(nn):
                z = torch.randn((nc, HORIZON, model.transition_dim), generator=gen).to("cuda:0")
                if i == 0:
                    hashes = [hashlib.sha256(np.ascontiguousarray(utils.to_np(z)[e]).tobytes()
                                             ).hexdigest()[:16] for e in range(nc)]
                for n in nfes:                     # SAME z for every NFE
                    xe, nfe_used, _ = solvers.integrate(model, cond, z, "euler", n)
                    assert nfe_used == n, f"NFE accounting mismatch {nfe_used} != {n}"
                    xn = utils.to_np(xe)
                    o = norm.unnormalize(xn[:, :, model.action_dim:], "observations")
                    st[n].append(o[:, 1].reshape(nc, 2, NPV, 10)[:, 0])
                    ac[n].append(norm.unnormalize(xn[:, :, :model.action_dim], "actions")[:, 0])
                print(f"  s{seed} noise {i} done", flush=True)
        for n in nfes:
            S[f"s{seed}_nfe{n}_state"] = np.stack(st[n], 1).astype(np.float32)   # (nc,nn,24,10)
            S[f"s{seed}_nfe{n}_action"] = np.stack(ac[n], 1).astype(np.float32)  # (nc,nn,3)
        S[f"s{seed}_target_t1"] = t1.astype(np.float32)
        S[f"s{seed}_target_action"] = ta.astype(np.float32)
        S[f"s{seed}_cur_latent"] = cur.astype(np.float32)
        S[f"s{seed}_episode"] = eps.astype(np.int64)
        S[f"s{seed}_timestep"] = ts.astype(np.int64)
        S[f"s{seed}_x0_hash"] = np.array(hashes)

    outp = ("experiments/loss_balance_audit/nfe_curve_smoke.npz" if cli.smoke
            else "experiments/loss_balance_audit/arm_neutral_nfe_curve.npz")
    np.savez_compressed(outp, **S)
    R = np.load(outp, allow_pickle=True)
    assert set(R.files) == set(S)
    print(f"saved {outp}; reload OK, {len(R.files)} keys")


if __name__ == "__main__":
    main()
