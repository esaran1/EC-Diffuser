"""Stage E: 2D t_state x t_action field diagnostics (directive section 24).

Offline only -- no closed-loop control, no training. Measures, on a frozen
checkpoint over a fixed grid, using IDENTICAL replay states, action targets and
noise at every cell:

    state velocity error        action velocity error
    cosine(prediction, target)  norm calibration
    distance-from-diagonal behaviour (the confound check)

This is scientifically meaningful ONLY for a model trained on off-diagonal
times. For a diagonal-trained model the same grid measures OOD drift, which is
what the closed warm-start branch established and withdrew.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "diffuser")
sys.path.insert(0, ".")

GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_arm(savepath, epoch="final", device="cuda:0"):
    """Load a trained arm through the production construction path."""
    from diffuser.utils.serialization import load_config
    ds_cfg = load_config(savepath, "dataset_config.pkl")
    ds_cfg._dict["dataset_path"] = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
    dataset = ds_cfg(seed=42)
    model_cfg = load_config(savepath, "model_config.pkl")
    diff_cfg = load_config(savepath, "diffusion_config.pkl")
    model = model_cfg()
    diffusion = diff_cfg(model)
    ck = torch.load(os.path.join(savepath, "state_{}.pt".format(epoch)),
                    map_location="cpu")
    diffusion.load_state_dict(ck["ema"])
    diffusion.to(device).eval()
    return dataset, diffusion, ck


@torch.no_grad()
def field(diffusion, batch, device, n_noise=2, seed=20260916):
    """Velocity-prediction quality across the 2D time grid.

    Identical x1, noise and conditioning at EVERY cell: only (t_s, t_a) vary,
    so differences are attributable to the time conditioning alone.
    """
    x1 = batch["x1"].to(device)
    cond = {k: v.to(device) for k, v in batch["cond"].items()}
    B = x1.shape[0]
    ad = diffusion.action_dim
    out = {}
    g = torch.Generator(device="cpu").manual_seed(seed)
    noises = [torch.randn(x1.shape, generator=g).to(device) for _ in range(n_noise)]

    x1c = x1.clone()
    diffusion._apply_conditioning(x1c, cond)
    mask = diffusion._make_conditioning_mask(x1c, cond)
    am = mask[:, :, :ad]
    om = mask[:, :, ad:]

    for ts in GRID:
        for ta in GRID:
            accs = {k: [] for k in ("s_err", "a_err", "s_cos", "a_cos",
                                    "s_norm_ratio", "a_norm_ratio")}
            for x0 in noises:
                x0c = x0.clone()
                diffusion._apply_conditioning(x0c, cond)
                t_s = torch.full((B,), ts, device=device, dtype=x1.dtype)
                t_a = torch.full((B,), ta, device=device, dtype=x1.dtype)
                xt = diffusion.corrupt(x1c, x0c, t_s, t_a)
                diffusion._apply_conditioning(xt, cond)
                target = x1c - x0c
                pred = diffusion._call_two_time_model(xt, cond, t_s, t_a)

                for nm, sl, m in (("s", slice(ad, None), om),
                                  ("a", slice(0, ad), am)):
                    p = pred[:, :, sl]
                    q = target[:, :, sl]
                    mm = m.to(p.dtype)
                    err = (((p - q).abs() * mm).sum() / mm.sum()).item()
                    pf = (p * mm).flatten(1)
                    qf = (q * mm).flatten(1)
                    cos = torch.nn.functional.cosine_similarity(
                        pf, qf, dim=1).mean().item()
                    nr = (pf.norm(dim=1) / qf.norm(dim=1).clamp_min(1e-8)).mean().item()
                    accs[nm + "_err"].append(err)
                    accs[nm + "_cos"].append(cos)
                    accs[nm + "_norm_ratio"].append(nr)
            out["{:.2f}_{:.2f}".format(ts, ta)] = {
                k: float(np.mean(v)) for k, v in accs.items()}
    return out


def make_batch(dataset, n=32, seed=20260916):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(dataset), size=n, replace=False)
    xs, conds = [], {}
    for i in idx:
        b = dataset[int(i)]
        xs.append(torch.as_tensor(b.trajectories))
        for k, v in b.conditions.items():
            conds.setdefault(k, []).append(torch.as_tensor(v))
    return {"x1": torch.stack(xs).float(),
            "cond": {k: torch.stack(v).float() for k, v in conds.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savepath", required=True)
    ap.add_argument("--epoch", default="final")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    dataset, diffusion, ck = load_arm(a.savepath, a.epoch, a.device)
    batch = make_batch(dataset, a.n_batch)
    res = field(diffusion, batch, a.device)

    payload = {
        "savepath": a.savepath, "epoch": a.epoch,
        "checkpoint_step": int(ck.get("step", -1)),
        "p_diag": float(getattr(diffusion, "p_diag", float("nan"))),
        "grid": GRID, "n_batch": a.n_batch, "cells": res,
    }
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("wrote", a.out)

    print("\naction velocity error (rows t_state, cols t_action):")
    print("        " + "".join("{:>9.2f}".format(t) for t in GRID))
    for ts in GRID:
        row = "  {:.2f}  ".format(ts)
        for ta in GRID:
            row += "{:>9.4f}".format(res["{:.2f}_{:.2f}".format(ts, ta)]["a_err"])
        print(row)


if __name__ == "__main__":
    main()
