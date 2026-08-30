"""Arm-neutral E16 vs E512 evaluation on RECORDED replay transitions.

NO Isaac Gym. NO simulator stepping. NO model-authored targets.
Every target (next state, action, goal) comes from the recorded dataset, so
neither solver authored anything. The models only predict.

Temporal mapping (traced from diffuser/datasets/sequence.py GoalDataset):
  observations fed to the model = [obs[s], obs[s+1], obs[s+2], obs[s+3], GOAL]
  actions                       = [act[s], act[s+1], act[s+2], act[s+3], 0]
  conditions                    = {0: observations[0], 4: observations[-1]=GOAL}
  => generated t=1  <->  recorded obs[s+1]      (the state target)
  => generated a[0] <->  recorded act[s]        (the action target)
  => act[s] drives obs[s] -> obs[s+1], the exact scored transition.
"""
import argparse, hashlib, json, sys
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit")
import solvers
import diffuser.utils as utils

DATA = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
N_PER_VIEW = 24
N_NOISE = 8
NOISE_BANK_SEED = 20260901
SAMPLE_SEED = 20260902
N_COND = 96
HORIZON = 5
ARMS = [("euler", 16), ("euler", 512)]
SEEDS = {42: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
         43: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed43",
         44: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed44"}


class Args:
    dataset = "panda_push"; horizon = HORIZON; seed = 42; device = "cuda:0"


CACHE_KEYS = ["cond_id", "episode", "timestep", "cur_latent", "goal_latent",
              "target_t1_latent", "target_action", "x0_hash"]
ARM_KEYS = ["{a}_full_norm", "{a}_obs_unnorm", "{a}_act_unnorm"]


def build_sample_set(obs, sf):
    """Deterministic, predeclared: spread across many distinct episodes."""
    rng = np.random.default_rng(SAMPLE_SEED)
    # only successful episodes, so the recorded goal is the achieved goal
    ok = np.flatnonzero(np.asarray(sf) >= 1.0)
    eps = rng.choice(ok, N_COND, replace=False)          # 96 DISTINCT episodes
    # start index must leave room for horizon-1 real observations
    ts = rng.integers(0, 100 - HORIZON, size=N_COND)
    return eps, ts


def preflight():
    """Phase 9: validate the entire save->reload->schema path on dummy data."""
    import tempfile, os
    print("[preflight] enumerating keys and write paths ...")
    src = open(__file__).read()
    for k in CACHE_KEYS:
        n = src.count(f'A["{k}"].append') + src.count(f'A["{k}"] =')
        assert n >= 1, f"key {k} has no write path"
    print(f"[preflight] {len(CACHE_KEYS)} metadata keys + {len(ARM_KEYS)*len(ARMS)} arm keys have write paths")
    S = {}
    for s in SEEDS:
        S[f"s{s}_cond_id"] = np.array([f"ep{i}_t{i}" for i in range(N_COND)])
        S[f"s{s}_episode"] = np.arange(N_COND, dtype=np.int64)
        S[f"s{s}_timestep"] = np.arange(N_COND, dtype=np.int64)
        S[f"s{s}_x0_hash"] = np.array(["deadbeef"] * N_COND)
        for k, shape in [("cur_latent", (N_COND, 48, 10)), ("goal_latent", (N_COND, 48, 10)),
                         ("target_t1_latent", (N_COND, 48, 10)), ("target_action", (N_COND, 3))]:
            S[f"s{s}_{k}"] = np.zeros(shape, np.float32)
        for a in ARMS:
            t = f"{a[0]}{a[1]}"
            S[f"s{s}_{t}_full_norm"] = np.zeros((N_COND, N_NOISE, HORIZON, 483), np.float32)
            S[f"s{s}_{t}_obs_unnorm"] = np.zeros((N_COND, N_NOISE, HORIZON, 480), np.float32)
            S[f"s{s}_{t}_act_unnorm"] = np.zeros((N_COND, N_NOISE, HORIZON, 3), np.float32)
    for k, v in S.items():
        assert not (isinstance(v, list) and len(v) == 0), f"empty {k}"
        assert len(v) == N_COND, f"length mismatch {k}: {len(v)}"
    fd = tempfile.mktemp(suffix=".npz")
    np.savez_compressed(fd, **S)
    R = np.load(fd, allow_pickle=True)
    assert set(R.files) == set(S), "reload key mismatch"
    for k in R.files:
        assert R[k].shape == np.asarray(S[k]).shape, f"shape mismatch {k}"
    h = hashlib.sha256(open(fd, "rb").read()).hexdigest()
    os.unlink(fd)
    print(f"[preflight] save->reload->validate OK; {len(S)} keys; hash producible ({h[:16]})")
    return {"n_keys": len(S), "all_keys_have_write_path": True,
            "save_reload_roundtrip": "OK", "empty_lists": 0,
            "length_agreement": "OK", "hash_producible": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    pf = preflight()
    import pickle
    raw = pickle.load(open(DATA, "rb"))
    obs, acts, sf = raw["observations"], raw["actions"], raw["info_goal_success_frac"]
    goals = raw["goals"]
    eps, ts = build_sample_set(obs, sf)
    ncond = 2 if cli.smoke else N_COND
    nnoise = 2 if cli.smoke else N_NOISE
    eps, ts = eps[:ncond], ts[:ncond]

    args = Args(); utils.set_global_device(args.device)
    S = {"_preflight": json.dumps(pf)}
    for seed, path in SEEDS.items():
        exp = utils.load_diffusion("data", args.dataset, path, epoch="latest", seed=42,
            is_diffusion=True, override_dataset_path=DATA)
        model = exp.ema; model.eval(); norm = exp.dataset.normalizer

        cur = obs[eps, ts]                       # (N,48,10) recorded current
        t1 = obs[eps, ts + 1]                    # (N,48,10) recorded next state  <- TARGET
        gl = goals[eps, 99]                      # recorded episode goal
        ta = acts[eps, ts]                       # (N,3) recorded action          <- TARGET
        cn = norm.normalize(cur.reshape(ncond, -1), "observations")
        gn = norm.normalize(gl.reshape(ncond, -1), "observations")
        cond = {0: torch.as_tensor(cn, device=args.device, dtype=torch.float32),
                4: torch.as_tensor(gn, device=args.device, dtype=torch.float32)}
        gen = torch.Generator(device="cpu").manual_seed(NOISE_BANK_SEED)
        A = {k: [] for k in CACHE_KEYS}
        per = {f"{a[0]}{a[1]}": {"full": [], "obs": [], "act": []} for a in ARMS}
        with torch.no_grad():
            for i in range(nnoise):
                z = torch.randn((ncond, HORIZON, model.transition_dim), generator=gen).to(args.device)
                for a in ARMS:
                    xe, _, _ = solvers.integrate(model, cond, z, *a)
                    xn = utils.to_np(xe); t = f"{a[0]}{a[1]}"
                    per[t]["full"].append(xn)
                    per[t]["obs"].append(norm.unnormalize(xn[:, :, model.action_dim:], "observations"))
                    per[t]["act"].append(norm.unnormalize(xn[:, :, :model.action_dim], "actions"))
                if i == 0:
                    A["x0_hash"] = [hashlib.sha256(
                        np.ascontiguousarray(utils.to_np(z)[e]).tobytes()).hexdigest()[:16]
                        for e in range(ncond)]
                print(f"  s{seed} noise {i}", flush=True)
        A["cond_id"] = [f"ep{int(e)}_t{int(t)}" for e, t in zip(eps, ts)]
        A["episode"] = eps.astype(np.int64); A["timestep"] = ts.astype(np.int64)
        A["cur_latent"] = cur; A["goal_latent"] = gl
        A["target_t1_latent"] = t1; A["target_action"] = ta
        for k in CACHE_KEYS:
            S[f"s{seed}_{k}"] = np.asarray(A[k])
        for a in ARMS:
            t = f"{a[0]}{a[1]}"
            S[f"s{seed}_{t}_full_norm"] = np.stack(per[t]["full"], 1).astype(np.float32)
            S[f"s{seed}_{t}_obs_unnorm"] = np.stack(per[t]["obs"], 1).astype(np.float32)
            S[f"s{seed}_{t}_act_unnorm"] = np.stack(per[t]["act"], 1).astype(np.float32)

    outp = ("experiments/loss_balance_audit/arm_neutral_smoke.npz" if cli.smoke
            else "experiments/loss_balance_audit/arm_neutral_endpoints.npz")
    np.savez_compressed(outp, **S)
    R = np.load(outp, allow_pickle=True)
    print(f"saved {outp}; reload OK, {len(R.files)} keys")
    print({k: R[k].shape for k in list(R.files)[1:5]})


if __name__ == "__main__":
    main()
