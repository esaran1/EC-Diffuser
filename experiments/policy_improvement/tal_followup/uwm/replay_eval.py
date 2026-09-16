"""Offline replay diagnostics for a trained arm (sections 21, 29 gate).

Measures, without any closed-loop control:
  - diagonal replay: does ordinary joint generation still work?
  - executed-action error at horizon position 0 (what control actually consumes)
  - imagination coherence (state prediction over the horizon)
  - NaN / range sanity (the section-29 gate)
  - LIVE vs EMA
  - parameter drift from the canonical source

Runs the PRODUCTION sampler for the diagonal and the path-correct two-time
sampler for non-diagonal paths.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "diffuser")
sys.path.insert(0, ".")
sys.path.insert(0, "experiments/policy_improvement/tal_followup/uwm")
from two_time_sampler import PATHS, two_time_sample  # noqa: E402


def load_arm(savepath, epoch, device, which="ema"):
    import diffuser.utils as utils
    ds_cfg = utils.Config.load(os.path.join(savepath, "dataset_config.pkl"))
    dataset = ds_cfg()
    model_cfg = utils.Config.load(os.path.join(savepath, "model_config.pkl"))
    diff_cfg = utils.Config.load(os.path.join(savepath, "diffusion_config.pkl"))
    diffusion = diff_cfg(model_cfg())
    ck = torch.load(os.path.join(savepath, "state_{}.pt".format(epoch)),
                    map_location="cpu")
    diffusion.load_state_dict(ck[which])
    diffusion.to(device).eval()
    return dataset, diffusion, ck


def make_batch(dataset, n, seed, device):
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
def eval_paths(diffusion, x1, cond, steps, device, paths=None):
    """Sample along each registered path and score against the replay target."""
    ad = diffusion.action_dim
    B, H, TD = x1.shape
    x1c = x1.clone()
    diffusion._apply_conditioning(x1c, cond)
    mask = diffusion._make_conditioning_mask(x1c, cond)
    res = {}

    g = torch.Generator(device="cpu").manual_seed(20260916)
    x_init = torch.randn(x1.shape, generator=g).to(device)

    def velocity(x, t_s, t_a):
        ts = torch.full((B,), t_s, device=device, dtype=x.dtype)
        ta = torch.full((B,), t_a, device=device, dtype=x.dtype)
        return diffusion._call_two_time_model(x, cond, ts, ta)

    def apply_cond(x):
        diffusion._apply_conditioning(x, cond)
        return x

    for name in (paths or list(PATHS)):
        out = two_time_sample(velocity, x_init.clone(), ad, path=name,
                              steps=steps, apply_cond=apply_cond,
                              cond_mask=mask)
        a_err = (out[:, :, :ad] - x1c[:, :, :ad]).abs()
        s_err_m = mask[:, :, ad:].to(out.dtype)
        s_err = (((out[:, :, ad:] - x1c[:, :, ad:]).abs() * s_err_m).sum()
                 / s_err_m.sum()).item()
        res[name] = {
            "action_err_all": float(a_err.mean().item()),
            "action_err_step0": float(a_err[:, 0, :].mean().item()),
            "state_err": float(s_err),
            "finite": bool(torch.isfinite(out).all().item()),
            "action_absmax": float(out[:, :, :ad].abs().max().item()),
            "model_calls": steps,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savepath", required=True)
    ap.add_argument("--epoch", default="final")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--canonical", default=None,
                    help="canonical checkpoint for parameter-drift measurement")
    a = ap.parse_args()

    payload = {"savepath": a.savepath, "epoch": a.epoch, "steps": a.steps}
    for which in ("ema", "model"):
        dataset, diffusion, ck = load_arm(a.savepath, a.epoch, a.device, which)
        x1, cond = make_batch(dataset, a.n_batch, 20260916, a.device)
        payload[which] = eval_paths(diffusion, x1, cond, a.steps, a.device)
        payload["checkpoint_step"] = int(ck.get("step", -1))
        payload["p_diag"] = float(getattr(diffusion, "p_diag", float("nan")))
        if which == "ema" and a.canonical:
            cn = torch.load(a.canonical, map_location="cpu")["ema"]
            cur = {k: v for k, v in ck["ema"].items()}
            drift, n = 0.0, 0
            for k, v in cn.items():
                if k in cur and v.dtype.is_floating_point:
                    drift += (cur[k].float() - v.float()).norm().item() ** 2
                    n += v.numel()
            payload["param_drift_l2"] = float(drift ** 0.5)
            payload["param_count"] = n
        del diffusion
        torch.cuda.empty_cache()

    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("wrote", a.out)
    print("\n  path                 act_err  act_err@0  state_err  finite")
    for name, r in payload["ema"].items():
        print("  {:<20} {:>7.4f}  {:>9.4f}  {:>9.4f}  {}".format(
            name, r["action_err_all"], r["action_err_step0"], r["state_err"],
            r["finite"]))
    if "param_drift_l2" in payload:
        print("\n  param drift from canonical (L2): {:.4f}".format(
            payload["param_drift_l2"]))


if __name__ == "__main__":
    main()
