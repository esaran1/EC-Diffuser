"""Mechanism Tests 1, 2 and 4: temporal decomposition, multi-noise executed-action
distribution, and replay-regime stratification.

Why: the executed one-sample replay error at t=0 is essentially identical between
A and B, yet closed-loop control differs. A single noise draw per condition can
hide a change in the POLICY DISTRIBUTION. These tests look for one.

Uses the diagnostics' own loader, dataset override and seed conventions -- imported
rather than reimplemented -- so numbers are comparable to the accepted results. An
earlier probe in this project produced confidently wrong numbers by rebuilding the
data pipeline; that is not repeated here.

Every statistic below is defined explicitly. The old unvalidated "dispersion"
metric is NOT revived.

TEST 1  temporal decomposition
    per-timestep L1 action error, t=0..H-1, terminal reported SEPARATELY and
    never pooled into a cross-arm action headline.

TEST 2  multi-noise executed-action distribution
    For each replay condition, K independent policy-noise draws, PAIRED across
    A and B (same condition, same draw index, same seed). Statistics on the
    executed action a_0:
      mean_err      mean_k || a_0(k) - a* ||_1        (accuracy)
      bias          || mean_k a_0(k) - a* ||_2        (systematic offset)
      var_trace     mean_k || a_0(k) - mean_k a_0 ||^2 (spread; = trace of the
                    sample covariance)
      per_dim_var   variance of each action coordinate across draws
      norm_mean     mean_k || a_0(k) ||_2
      cos_consist   mean pairwise cosine between draws (directional agreement)
    bias^2 + var_trace decomposes the expected squared error, so a change in
    control with unchanged mean error could still show up as a bias/variance
    trade.

TEST 4  regime stratification
    t=0 statistics split by PREDECLARED quartiles of (a) demonstrated action
    magnitude and (b) object-to-goal distance taken from the trajectory itself.
    Exploratory: interpreted as hypothesis-generating, not confirmatory.
"""
import argparse
import json
import os
import sys

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np
import torch

import diffuser.utils as utils

sys.path.insert(0, "experiments/scripts")
from isaacgym_control import Args  # noqa: E402

NOISE_SEED = 20260902           # same constant the diagnostics uses
SUB_TMPL = "3C_dlp_adalnpint_randcolor_H5_T4_seed{}"
OVERRIDE_DATASET = ("ecdiffuser-data/push_cubes/3C_randcolor/"
                    "panda_push_replay_buffer_dlp.pkl")


def load_arm(arm, seed, args):
    loadpath = "lossft_arm{}/{}".format(arm, SUB_TMPL.format(seed))
    return utils.load_diffusion("data", args.dataset, loadpath, epoch="final",
                                seed=args.seed, is_diffusion=True,
                                override_dataset_path=OVERRIDE_DATASET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45")
    ap.add_argument("--n_conditions", type=int, default=256)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--out", default="experiments/policy_improvement/loss_finetune/"
                                     "mechanism/ACTION_DISTRIBUTION.json")
    cli = ap.parse_args()

    args = Args()
    utils.set_global_device(args.device)
    dev = args.device
    results = {}

    for seed in [int(s) for s in cli.seeds.split(",")]:
        per_arm = {}
        meta = None
        for arm in ("A", "B"):
            exp = load_arm(arm, seed, args)
            diff = exp.ema                      # EMA, as the control evaluator uses
            diff.eval()
            ds = exp.dataset
            ad, H = diff.action_dim, diff.horizon

            loader = torch.utils.data.DataLoader(ds, batch_size=cli.batch_size,
                                                 shuffle=False, num_workers=0)
            temporal = [[] for _ in range(H)]
            dist = {k: [] for k in ("mean_err", "bias", "var_trace", "norm_mean",
                                    "cos_consist")}
            per_dim_var = []
            strat_keys = {"act_mag": [], "goal_dist": []}
            n_done = 0
            with torch.no_grad():
                for b in loader:
                    if n_done >= cli.n_conditions:
                        break
                    x = b.trajectories.to(dev)
                    c = {k: v.to(dev) for k, v in b.conditions.items()}
                    bs = x.shape[0]

                    # --- TEST 1: single-draw temporal decomposition -----------
                    torch.manual_seed(NOISE_SEED)
                    kw = dict(verbose=False)
                    if hasattr(diff, "n_solver_steps"):
                        kw["n_steps"] = cli.nfe
                    s = diff.conditional_sample(c, **kw)
                    xs = s.trajectories if hasattr(s, "trajectories") else s
                    for t in range(H):
                        temporal[t].append(
                            float((xs[:, t, :ad] - x[:, t, :ad]).abs().mean()))

                    # --- TEST 2: K paired noise draws, executed action --------
                    draws = []
                    for k in range(cli.K):
                        torch.manual_seed(NOISE_SEED + 1000 * k)   # paired across arms
                        sk = diff.conditional_sample(c, **kw)
                        xk = sk.trajectories if hasattr(sk, "trajectories") else sk
                        draws.append(xk[:, 0, :ad].clone())
                    D = torch.stack(draws, 0)                      # [K, bs, ad]
                    target = x[:, 0, :ad]                          # [bs, ad]
                    mean_a = D.mean(0)
                    dist["mean_err"].append(
                        float((D - target.unsqueeze(0)).abs().mean()))
                    dist["bias"].append(
                        float((mean_a - target).pow(2).sum(-1).sqrt().mean()))
                    centered = D - mean_a.unsqueeze(0)
                    dist["var_trace"].append(float(centered.pow(2).sum(-1).mean()))
                    dist["norm_mean"].append(float(D.pow(2).sum(-1).sqrt().mean()))
                    per_dim_var.append(D.var(0, unbiased=True).mean(0).cpu().numpy())
                    Dn = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
                    cs = torch.einsum("kbd,lbd->klb", Dn, Dn)
                    iu = torch.triu_indices(cli.K, cli.K, offset=1)
                    dist["cos_consist"].append(float(cs[iu[0], iu[1], :].mean()))

                    # --- TEST 4: predeclared stratification keys --------------
                    strat_keys["act_mag"].append(
                        target.pow(2).sum(-1).sqrt().cpu().numpy())
                    # object-to-goal proxy: distance between the conditioned
                    # current and goal observations, both already in the batch
                    ks = sorted(c)
                    gd = (c[ks[-1]] - c[ks[0]]).pow(2).sum(-1).sqrt()
                    strat_keys["goal_dist"].append(gd.cpu().numpy())
                    n_done += bs

            per_arm[arm] = {
                "n_conditions": n_done, "K": cli.K,
                "temporal_action_l1": [float(np.mean(v)) for v in temporal],
                "distribution": {k: float(np.mean(v)) for k, v in dist.items()},
                "per_dim_var": np.concatenate([p[None] for p in per_dim_var]).mean(0).tolist(),
            }
            meta = {k: np.concatenate(v) for k, v in strat_keys.items()}
            del exp, diff
            torch.cuda.empty_cache()

        results[seed] = per_arm
        A, B = per_arm["A"], per_arm["B"]
        print("seed {}: t0 L1 A {:.5f} B {:.5f} | var_trace A {:.3e} B {:.3e} "
              "| bias A {:.5f} B {:.5f} | cos A {:.4f} B {:.4f}".format(
                  seed, A["temporal_action_l1"][0], B["temporal_action_l1"][0],
                  A["distribution"]["var_trace"], B["distribution"]["var_trace"],
                  A["distribution"]["bias"], B["distribution"]["bias"],
                  A["distribution"]["cos_consist"], B["distribution"]["cos_consist"]),
              flush=True)

    with open(cli.out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print("written", cli.out)


if __name__ == "__main__":
    main()
