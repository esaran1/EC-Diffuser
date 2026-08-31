"""Offline demonstrated-action error vs inference budget, lerobot/diffusion_pusht.

ALIGNMENT (verified from source, see OFFLINE_ALIGNMENT.md):
  n_obs_steps=2, horizon=16, n_action_steps=8.
  generate_actions() returns actions[:, start:end] with start = n_obs_steps-1 = 1,
  end = 1+8 = 9. So the EXECUTED actions correspond to demo actions at
  frame offsets [0, +1, ..., +7] relative to the CURRENT frame t.
  Observations used are frames [t-1, t] (delta_timestamps -1/fps, 0).

PRIMARY metric: L2 over the 8 actions the policy would actually execute.
SECONDARY: full 16-step trajectory error.
"""
import argparse, json, os, time
import numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

CKPT = ("hf_cache/hub/models--lerobot--diffusion_pusht/snapshots/"
        "84a7c23178445c6bbf7e1a884ff497017910f653")
SAMPLE_SEED = 20260920
NOISE_SEED = 20260921


def build(nsteps, device="cuda"):
    p = DiffusionPolicy.from_pretrained(CKPT)
    if nsteps is not None:
        p.config.num_inference_steps = nsteps
        p.diffusion.num_inference_steps = nsteps
    p.to(device); p.eval()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, nargs="+", default=[1,2,4,5,10,20,50,100])
    ap.add_argument("--n-cond", type=int, default=200)
    ap.add_argument("--K", type=int, default=4)
    cli = ap.parse_args()
    dev = "cuda"
    fps = 10
    # observations at t-1,t ; actions at t..t+15 (full horizon), executed = first 8
    dt = {"observation.image": [-1/fps, 0.0],
          "observation.state": [-1/fps, 0.0],
          "action": [i/fps for i in range(16)]}
    ds = LeRobotDataset("lerobot/pusht", delta_timestamps=dt)
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(len(ds), cli.n_cond, replace=False)
    idx.sort()

    # ---- build the batch once (shared by every budget) ----
    obs_img, obs_st, act = [], [], []
    for i in idx:
        it = ds[int(i)]
        obs_img.append(it["observation.image"]); obs_st.append(it["observation.state"])
        act.append(it["action"])
    OI = torch.stack(obs_img).to(dev); OS = torch.stack(obs_st).to(dev)
    A = torch.stack(act).to(dev)                       # (N,16,2) demo actions
    print(f"[align] obs.image {tuple(OI.shape)} obs.state {tuple(OS.shape)} demo action {tuple(A.shape)}")
    assert OI.shape[1] == 2 and A.shape[1] == 16

    out = {"protocol": {"n_cond": int(cli.n_cond), "K": int(cli.K),
                        "sample_seed": SAMPLE_SEED, "noise_seed": NOISE_SEED,
                        "dataset": "lerobot/pusht", "n_frames": len(ds),
                        "executed_slice": "[0:8] of the 16-step demo chunk (start=n_obs_steps-1=1 in model output)",
                        "delta_timestamps": {k: v for k, v in dt.items()}},
           "results": {}}
    for n in cli.nsteps:
        p = build(n); p.diffusion.noise_scheduler.set_timesteps(p.diffusion.num_inference_steps)
        ts = [int(t) for t in p.diffusion.noise_scheduler.timesteps]
        errs_exec, errs_full = [], []
        t0 = time.time()
        for k in range(cli.K):
            torch.manual_seed(NOISE_SEED + k); torch.cuda.manual_seed_all(NOISE_SEED + k)
            preds = []
            with torch.no_grad():
                for s in range(0, len(idx), 50):
                    b = {"observation.image": OI[s:s+50], "observation.state": OS[s:s+50]}
                    bn = p.normalize_inputs(b)
                    bn = dict(bn); bn["observation.images"] = torch.stack(
                        [bn[key] for key in p.config.image_features], dim=-4)
                    gc = p.diffusion._prepare_global_conditioning(bn)
                    a = p.diffusion.conditional_sample(len(bn["observation.state"]), global_cond=gc)
                    a = p.unnormalize_outputs({"action": a})["action"]
                    preds.append(a)
            P = torch.cat(preds, 0)                     # (N,16,2) full predicted horizon
            ex = P[:, 1:9]                              # EXECUTED slice per generate_actions
            errs_exec.append(torch.linalg.norm(ex - A[:, 0:8], dim=-1).mean(dim=1).cpu().numpy())
            errs_full.append(torch.linalg.norm(P - A, dim=-1).mean(dim=1).cpu().numpy())
        E = np.stack(errs_exec); F = np.stack(errs_full)
        out["results"][str(n)] = {
            "timesteps": ts, "n_timesteps": len(ts), "first_t": ts[0], "last_t": ts[-1],
            "exec_action_l2_mean": float(E.mean()), "exec_action_l2_median": float(np.median(E)),
            "exec_across_sample_sd": float(E.mean(axis=1).std(ddof=1)),
            "exec_per_cond_sd_over_K": float(E.std(axis=0, ddof=1).mean()),
            "full_horizon_l2_mean": float(F.mean()), "wall_s": time.time()-t0}
        r = out["results"][str(n)]
        print(f"  nfe{n:3d} steps={len(ts):3d} exec_L2={r['exec_action_l2_mean']:.4f} "
              f"(sd over K {r['exec_per_cond_sd_over_K']:.4f})  full_L2={r['full_horizon_l2_mean']:.4f} "
              f"{r['wall_s']:.0f}s")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/offline_action_error.json", "w"), indent=2)
    print("wrote results/offline_action_error.json")


if __name__ == "__main__":
    main()
