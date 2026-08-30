"""Closed-loop Flow NFE4 vs NFE32, three training seeds, frozen replicate0.

Drives the CANONICAL isaacgym_nfe_study harness. Nothing in the evaluator or
policy is modified: this only sets PLAN to the two arms and repoints the flow
checkpoint per training seed. NFE is the sole experimental variable.
"""
import argparse, json, os, sys
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/scripts")

import isaacgym_nfe_study as NS
from isaacgym_control import ARMS, Args
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env

SEEDS = (42, 43, 44)
STEPS = (4, 32)
OUT = "experiments/isaacgym_control/nfe4_vs_nfe32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    cli = ap.parse_args()

    args = Args(); utils.set_global_device(args.device)
    os.makedirs(OUT, exist_ok=True)
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    episode_set, _ = NS.get_episode_set(env, cli.replicate, cli.episodes)
    print(f"[set] replicate{cli.replicate} sha={episode_set['sha256'][:16]} "
          f"n={len(episode_set['init'])}", flush=True)

    for tseed in cli.seeds:
        ARMS["flow"]["loadpath"] = f"flow/3C_dlp_adalnpint_randcolor_H5_T4_seed{tseed}"
        for steps in cli.steps:
            label = f"s{tseed}_nfe{steps}"
            out = os.path.join(OUT, f"r{cli.replicate}_{label}.json")
            if os.path.exists(out):
                print(f"[skip] {out}", flush=True); continue
            policy = NS.build_policy("flow", steps, args)
            records, elapsed = NS.evaluate(policy, env, episode_set, args, label)
            summary = NS.summarize(records, policy, label, "flow", steps, episode_set, elapsed)
            summary["training_seed"] = tseed
            measured = summary["measured_calls_per_plan"]
            if abs(measured - float(steps)) > 1e-6:
                summary["CALL_COUNT_MISMATCH"] = True
                print(f"  !! expected {steps} calls/plan, measured {measured}", flush=True)
            with open(out, "w") as h:
                json.dump({"summary": summary, "episodes": records}, h, indent=2)
            print(f"  -> {label}: success={summary['success_rate']:.4f} "
                  f"({measured:.1f} calls, {summary['latency_mean_ms']:.2f} ms, "
                  f"{elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
