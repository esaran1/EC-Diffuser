"""Stage C direct-action metrics (instruction item 5). GPU, frozen checkpoints only.

Measures, for a direct-action model and its matched control:
  - action replay error (all horizon positions, and position 0 = what control executes)
  - NEAR-GOAL / STOPPING behaviour: does the policy reduce action magnitude as
    the state approaches the goal? The terminal supervision target is a
    normalized zero action, so a policy that has learned "stop at the goal"
    should show action magnitude falling with goal proximity.
  - state imagination quality (multi-step state prediction error)
  - inference model calls and wall latency
No training. No optimizer steps.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "diffuser")
sys.path.insert(0, ".")


def load(savepath, epoch, device, which="ema"):
    from diffuser.utils.serialization import load_config
    ds_cfg = load_config(savepath, "dataset_config.pkl")
    ds_cfg._dict["dataset_path"] = (
        "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl")
    dataset = ds_cfg(seed=42)
    model_cfg = load_config(savepath, "model_config.pkl")
    diff_cfg = load_config(savepath, "diffusion_config.pkl")
    diffusion = diff_cfg(model_cfg())
    ck = torch.load(os.path.join(savepath, "state_{}.pt".format(epoch)),
                    map_location="cpu")
    diffusion.load_state_dict(ck[which])
    diffusion.to(device).eval()
    return dataset, diffusion, ck


def batch_from(dataset, n, seed, device):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(dataset), size=n, replace=False)
    xs, conds = [], {}
    for i in idx:
        b = dataset[int(i)]
        xs.append(torch.as_tensor(b.trajectories))
        for k, v in b.conditions.items():
            conds.setdefault(k, []).append(torch.as_tensor(v))
    return (torch.stack(xs).float().to(device),
            {k: torch.stack(v).float().to(device) for k, v in conds.items()})


@torch.no_grad()
def measure(diffusion, x1, cond, steps, device, n_latency=5):
    ad = diffusion.action_dim
    H = x1.shape[1]
    x1c = x1.clone()
    diffusion._apply_conditioning(x1c, cond)
    mask = diffusion._make_conditioning_mask(x1c, cond)
    if getattr(diffusion, "action_space", "raw") == "latent":
        # samples come back decoded to normalized raw actions; put the targets
        # in the same space so every arm is scored on physical actions
        x1c = diffusion.decode_actions(x1c)

    # --- sampling, with latency ---
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.time()
    s = diffusion.conditional_sample(cond, n_steps=steps)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t_first = time.time() - t0
    lat = []
    for _ in range(n_latency):
        torch.cuda.synchronize() if device.startswith("cuda") else None
        a = time.time()
        diffusion.conditional_sample(cond, n_steps=steps)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        lat.append(time.time() - a)
    out = s.trajectories

    res = {}
    # --- action replay error ---
    ae = (out[:, :, :ad] - x1c[:, :, :ad]).abs()
    res["action_err_all"] = float(ae.mean().item())
    res["action_err_step0"] = float(ae[:, 0, :].mean().item())
    res["action_err_per_step"] = [float(ae[:, t, :].mean().item()) for t in range(H)]

    # --- state / imagination ---
    om = mask[:, :, ad:].to(out.dtype)
    res["state_err"] = float((((out[:, :, ad:] - x1c[:, :, ad:]).abs() * om).sum()
                              / om.sum()).item())
    # per-horizon-step state error = imagination quality over the rollout
    per = []
    for t in range(H):
        mt = om[:, t, :]
        if mt.sum() > 0:
            per.append(float((((out[:, t, ad:] - x1c[:, t, ad:]).abs() * mt).sum()
                              / mt.sum()).item()))
        else:
            per.append(None)
    res["state_err_per_step"] = per

    # --- near-goal / stopping behaviour ---
    # goal is the conditioned final observation; measure predicted action
    # magnitude against distance from current observation to the goal.
    goal_key = max(cond.keys())
    goal = cond[goal_key]                       # [B, OD]
    cur = cond[min(cond.keys())]                # [B, OD]
    dist = (cur - goal).norm(dim=1)             # [B] proximity proxy
    amag = out[:, 0, :ad].norm(dim=1)           # executed action magnitude
    tmag = x1c[:, 0, :ad].norm(dim=1)
    dn = dist.cpu().numpy(); an = amag.cpu().numpy(); tn = tmag.cpu().numpy()
    res["nearGoal_corr_pred"] = float(np.corrcoef(dn, an)[0, 1]) if dn.std() > 0 else None
    res["nearGoal_corr_target"] = float(np.corrcoef(dn, tn)[0, 1]) if dn.std() > 0 else None
    # terminal-row action magnitude: the constructed "stop" target
    res["terminal_action_mag_pred"] = float(out[:, H - 1, :ad].norm(dim=1).mean().item())
    res["terminal_action_mag_target"] = float(x1c[:, H - 1, :ad].norm(dim=1).mean().item())
    # split by goal proximity quartile
    q = np.quantile(dn, [0.25, 0.75])
    near = an[dn <= q[0]]; far = an[dn >= q[1]]
    res["action_mag_near_goal"] = float(near.mean()) if near.size else None
    res["action_mag_far_goal"] = float(far.mean()) if far.size else None

    # --- compute ---
    res["model_calls"] = int(steps)
    res["latency_s_mean"] = float(np.mean(lat))
    res["latency_s_sd"] = float(np.std(lat))
    res["batch"] = int(x1.shape[0])
    res["finite"] = bool(torch.isfinite(out).all().item())
    res["action_absmax"] = float(out[:, :, :ad].abs().max().item())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savepath", required=True)
    ap.add_argument("--epoch", default="final")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    payload = {"savepath": a.savepath, "epoch": a.epoch, "steps": a.steps}
    for which in ("ema", "model"):
        dataset, diffusion, ck = load(a.savepath, a.epoch, a.device, which)
        x1, cond = batch_from(dataset, a.n_batch, 20260916, a.device)
        payload[which] = measure(diffusion, x1, cond, a.steps, a.device)
        payload["checkpoint_step"] = int(ck.get("step", -1))
        del diffusion
        torch.cuda.empty_cache()

    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("wrote", a.out)
    e = payload["ema"]
    print("  action_err  all {:.4f}  @0 {:.4f}".format(
        e["action_err_all"], e["action_err_step0"]))
    print("  state_err   {:.4f}".format(e["state_err"]))
    print("  near-goal   corr(dist, |a|) pred {} target {}".format(
        e["nearGoal_corr_pred"], e["nearGoal_corr_target"]))
    print("  |a| near-goal {} vs far {}".format(
        e["action_mag_near_goal"], e["action_mag_far_goal"]))
    print("  terminal |a| pred {:.4f} target {:.4f}".format(
        e["terminal_action_mag_pred"], e["terminal_action_mag_target"]))
    print("  calls {}  latency {:.4f}s".format(e["model_calls"], e["latency_s_mean"]))


if __name__ == "__main__":
    main()
