"""PHASE 6: objective Flow-vs-Gaussian future-state comparison, no training.

For each checkpoint we generate a trajectory from the SAME current/goal pair,
take the generated interior-horizon latents, and compare them to the DLP
encoding of the state the environment ACTUALLY reached after executing the
policy's own first action (predicted-vs-realized, one step ahead).

Metrics are the validated permutation-invariant ones from latent_metric.py.
A "copy the current state" baseline is included: a model that merely echoes its
input must score no better than that.
"""
import argparse, json, os, pickle, sys
import isaacgym  # noqa: F401,E402  must precede torch
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit")
sys.path.insert(0, "experiments/scripts")
from latent_metric import training_stats, block_errors, chamfer_position

import diffuser.utils as utils
from diffuser.configuration import flow_sampling_kwargs
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import array_to_state_dict
from isaacgym_fourcube_probe import ENTITY_TO_STEPS  # noqa: F401

N_PER_VIEW = 24
CKPTS = {
    "gaussian":  ("ecdiffuser-data/pretrained_models", "diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100", 100),
    "flow_s42":  ("data", "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42", None),
    "flow_s43":  ("data", "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed43", None),
    "flow_s44":  ("data", "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed44", None),
}


class Args:
    env_config_dir = "env_config/generalization_num_cubes"
    dataset = "panda_push"; num_entity = 3; horizon = 5
    max_episode_length = 100; planning_only = True; push_t = False
    multiview = True; verbose = False; seed = 42; device = "cuda:0"
    preprocess_fns = []; push_t_num_color = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--nfe", type=int, nargs="+", default=[1, 4])
    cli = ap.parse_args()

    args = Args(); utils.set_global_device(args.device)
    obs_raw = pickle.load(open("ecdiffuser-data/push_cubes/3C_randcolor/"
                               "panda_push_replay_buffer_dlp.pkl", "rb"))["observations"][:400]
    mu, sd = training_stats(obs_raw.reshape(-1, 48, 10))

    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    ep = pickle.load(open("experiments/isaacgym_episode_sets/replicate0_n96.pkl", "rb"))
    keys = ep["keys"]
    print(f"[set] {ep['sha256'][:16]} n={len(ep['init'])} cubes={ep['init'].shape[1]}", flush=True)

    results = {}
    for name, (base, path, fixed_nfe) in CKPTS.items():
        nfes = [fixed_nfe] if fixed_nfe else cli.nfe
        for nfe in nfes:
            tag = f"{name}_nfe{nfe}"
            exp = utils.load_diffusion(base, args.dataset, path, epoch="latest", seed=args.seed,
                                       is_diffusion=True,
                                       override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/"
                                                             "panda_push_replay_buffer_dlp.pkl")
            policy = utils.Config("sampling.GoalConditionedPolicy",
                diffusion_model=exp.ema, normalizer=exp.dataset.normalizer,
                preprocess_fns=[], verbose=False, horizon=args.horizon,
                **flow_sampling_kwargs(exp.ema, nfe))()

            rows, copy_rows = [], []
            idx = list(range(min(cli.episodes, env.num_envs)))
            idx += [idx[-1]] * (env.num_envs - len(idx))
            obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, env.device),
                            set_goal_states=array_to_state_dict(ep["goal"][idx], keys, env.device))
            for _ in range(cli.steps):
                observation = obs["achieved_goal"].reshape(env.num_envs, -1)
                goal = obs["desired_goal"].reshape(env.num_envs, -1)
                _, samples = policy({0: observation, args.horizon - 1: goal},
                                    batch_size=1, verbose=False)
                gen = np.asarray(samples.observations).reshape(
                    env.num_envs, args.horizon, 2, N_PER_VIEW, 10)[:, 1, 0]   # slot 1, front view
                cur = observation.reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]
                obs, _, _, _ = env.step(samples.actions[:, 0])
                real = obs["achieved_goal"].reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]
                for e in range(min(cli.episodes, env.num_envs)):
                    rows.append({**block_errors(gen[e], real[e], mu, sd),
                                 "chamfer": chamfer_position(gen[e], real[e])})
                    copy_rows.append({**block_errors(cur[e], real[e], mu, sd),
                                      "chamfer": chamfer_position(cur[e], real[e])})
            agg = {k: {"mean": float(np.mean([r[k] for r in rows])),
                       "median": float(np.median([r[k] for r in rows])),
                       "sd": float(np.std([r[k] for r in rows]))}
                   for k in rows[0]}
            agg["copy_baseline"] = {k: float(np.mean([r[k] for r in copy_rows])) for k in copy_rows[0]}
            agg["n"] = len(rows)
            results[tag] = agg
            print(f"  {tag:18s} pos_z={agg['pos']['mean']:.4f} chamfer={agg['chamfer']['mean']:.5f} "
                  f"vis_z={agg['vis']['mean']:.4f} (copy: pos_z={agg['copy_baseline']['pos']:.4f} "
                  f"chamfer={agg['copy_baseline']['chamfer']:.5f})", flush=True)

    os.makedirs("experiments/loss_balance_audit", exist_ok=True)
    with open("experiments/loss_balance_audit/imagination_comparison.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote experiments/loss_balance_audit/imagination_comparison.json")


if __name__ == "__main__":
    main()
