"""Per-timestep decomposition of the replay action error.

Determines WHERE the replay action error lives across the horizon, so the
A-vs-B/C/D comparison of that metric can be interpreted correctly.

`replay_error` in phase2_diagnostics.py averages the action term over ALL
horizon steps, including t=H-1 -- the slice B/C/D no longer supervise -- while
the state term restricts itself to t=1:4. Only t=0 is executed in closed loop.

This script reuses the diagnostics' OWN loader, dataset override, seed and
sampling conventions (imported, not reimplemented) so its numbers are directly
comparable to the reported replay_action_error. An earlier version rebuilt the
dataset from the arm config and used the wrong seed; its numbers were not
comparable and are superseded.

Run ONLY when the GPU is free -- never overlapping the pinned scientific runs.
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
sys.path.insert(0, "experiments/policy_improvement")
from isaacgym_control import Args  # noqa: E402

# Same constant the diagnostics uses for replay sampling.
NOISE_SEED = 20260902
SUB_TMPL = "3C_dlp_adalnpint_randcolor_H5_T4_seed{}"
OVERRIDE_DATASET = ("ecdiffuser-data/push_cubes/3C_randcolor/"
                    "panda_push_replay_buffer_dlp.pkl")


def per_timestep_replay(diff, ds, dev, nfe=4, n_batches=12):
    """Replay exactly as phase2_diagnostics.replay_error, but kept per timestep."""
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    ad = diff.action_dim
    H = diff.horizon
    per_t = [[] for _ in range(H)]
    all_steps = []
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= n_batches:
                break
            x = b.trajectories.to(dev)
            c = {k: v.to(dev) for k, v in b.conditions.items()}
            torch.manual_seed(NOISE_SEED)
            kw = dict(verbose=False)
            if hasattr(diff, "n_solver_steps"):
                kw["n_steps"] = nfe
            s = diff.conditional_sample(c, **kw)
            xs = s.trajectories if hasattr(s, "trajectories") else s
            # the exact quantity the diagnostics reports
            all_steps.append(float((xs[:, :, :ad] - x[:, :, :ad]).abs().mean()))
            for t in range(H):
                per_t[t].append(float((xs[:, t, :ad] - x[:, t, :ad]).abs().mean()))
    means = [float(np.mean(v)) for v in per_t]
    return means, float(np.mean(all_steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--seed", type=int, default=42,
                    help="fine-tuning stochastic seed identifying the run dirs")
    ap.add_argument("--out", default="experiments/policy_improvement/loss_finetune/"
                                     "rerun/REPLAY_PER_TIMESTEP.json")
    cli = ap.parse_args()

    args = Args()
    utils.set_global_device(args.device)
    dev = args.device
    results = {}

    for arm in [a.strip() for a in cli.arms.split(",") if a.strip()]:
        loadpath = "lossft_arm{}/{}".format(arm, SUB_TMPL.format(cli.seed))
        if not os.path.exists(os.path.join("data", args.dataset, loadpath,
                                           "state_final.pt")):
            print("skip arm {}: no final checkpoint".format(arm))
            continue
        exp = utils.load_diffusion("data", args.dataset, loadpath, epoch="final",
                                   seed=args.seed, is_diffusion=True,
                                   override_dataset_path=OVERRIDE_DATASET)
        entry = {}
        for tag, model in (("LIVE", exp.diffusion), ("EMA", exp.ema)):
            means, all_steps = per_timestep_replay(model, exp.dataset, dev)
            H = len(means)
            entry[tag] = {
                "per_timestep_action_error": means,
                "all_timestep_mean_as_reported": all_steps,
                "executed_action_t0": means[0],
                "terminal_action_tHm1": means[H - 1],
                "mean_excluding_terminal": float(np.mean(means[:H - 1])),
            }
            print("arm {} {}: per-t {} | reported-style {:.5f} | t0 {:.5f} | "
                  "t_H-1 {:.5f} | excl-terminal {:.5f}".format(
                      arm, tag, ["{:.5f}".format(m) for m in means], all_steps,
                      means[0], means[H - 1], entry[tag]["mean_excluding_terminal"]),
                  flush=True)
        results[arm] = entry
        del exp
        torch.cuda.empty_cache()

    with open(cli.out, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print("\nwritten", cli.out)


if __name__ == "__main__":
    main()
