"""NFE-vs-imagination sweep with FIXED initial noise. No training, no loss change.

Central control: within each (episode, rollout step, training seed), the current
observation, goal, conditioning and the initial noise z are IDENTICAL across
NFE in {1,2,4,8,16}. Only the Euler discretisation changes.

`conditional_sample` draws its own noise internally with no injection point, so
the diagnostic re-implements the SAME left-endpoint fixed-step Euler loop
(flow_matching.py:358-373) with an externally supplied x0. The loop is copied
verbatim in structure: same dt, same time grid, same velocity masking, same
re-application of conditioning after each step.

Also records vector-field diagnostics along the integration path.
"""
import argparse, hashlib, json, os, pickle, sys
import isaacgym  # noqa: F401,E402  must precede torch
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit")
sys.path.insert(0, "experiments/scripts")
from latent_metric import training_stats, block_errors, chamfer_position

import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import array_to_state_dict

N_PER_VIEW = 24
NFES = [1, 2, 4, 8, 16]
FLOW_SEEDS = {
    42: ("data", "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
         "861dc34434474455a25dc3a15ea4e1754066202df538364cf41114b42f4fcc3b"),
    43: ("data", "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed43",
         "c8e00eadfed9b8a0b54c5423864457e5330434b748c50f820cb7d7a0328ac826"),
    44: ("data", "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed44",
         "c2c13f557aca7cf0eeb29b7baa572cf62a0027021e90ef75a7ca059b3f0e2bd3"),
}
GAUSSIAN = ("ecdiffuser-data/pretrained_models",
            "diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100", 100)


class Args:
    env_config_dir = "env_config/generalization_num_cubes"
    dataset = "panda_push"; num_entity = 3; horizon = 5
    max_episode_length = 100; planning_only = True; push_t = False
    multiview = True; verbose = False; seed = 42; device = "cuda:0"
    preprocess_fns = []; push_t_num_color = 1


def euler_sample(model, cond, x0, steps, want_diag=False):
    """Verbatim re-implementation of flow_matching.py:358-373 with fixed x0."""
    x = x0.clone()
    model._apply_conditioning(x, cond)
    cmask = model._make_conditioning_mask(x, cond)
    dt = 1.0 / float(steps)
    diag = {"update_norm": [], "cos_consecutive": []}
    prev_v = None
    for step in range(steps):
        t = x.new_full((x.shape[0],), float(step) / float(steps))
        v = model.model(x, cond, t * model.time_scale)
        v = v * cmask.to(x.dtype)
        if want_diag:
            diag["update_norm"].append((dt * v).norm(dim=(1, 2)).mean().item())
            if prev_v is not None:
                a, b = v.flatten(1), prev_v.flatten(1)
                diag["cos_consecutive"].append(
                    torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item())
            prev_v = v.clone()
        x = x + dt * v
        model._apply_conditioning(x, cond)
    return x, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--noise-seed", type=int, default=777)
    cli = ap.parse_args()

    args = Args(); utils.set_global_device(args.device)
    raw = pickle.load(open("ecdiffuser-data/push_cubes/3C_randcolor/"
                           "panda_push_replay_buffer_dlp.pkl", "rb"))["observations"][:400]
    mu, sd = training_stats(raw.reshape(-1, 48, 10))

    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    ep = pickle.load(open("experiments/isaacgym_episode_sets/replicate0_n96.pkl", "rb"))
    keys = ep["keys"]
    print(f"[set] {ep['sha256'][:16]} n={len(ep['init'])} cubes={ep['init'].shape[1]}", flush=True)

    results, diags = {}, {}

    def rollout(tag, base, path, nfe_list, gaussian=False, sha=None):
        exp = utils.load_diffusion(base, args.dataset, path, epoch="latest", seed=args.seed,
            is_diffusion=True, override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/"
                                                     "panda_push_replay_buffer_dlp.pkl")
        model = exp.ema; model.eval()
        norm = exp.dataset.normalizer
        rows = {n: [] for n in nfe_list}; copy_rows = []
        dg = {n: {"update_norm": [], "cos": []} for n in nfe_list}

        idx = list(range(min(cli.episodes, env.num_envs)))
        idx += [idx[-1]] * (env.num_envs - len(idx))
        obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, env.device),
                        set_goal_states=array_to_state_dict(ep["goal"][idx], keys, env.device))
        gen_noise = torch.Generator(device="cpu").manual_seed(cli.noise_seed)

        for rs in range(cli.steps):
            observation = obs["achieved_goal"].reshape(env.num_envs, -1)
            goal = obs["desired_goal"].reshape(env.num_envs, -1)
            cond = {0: torch.as_tensor(norm.normalize(observation, "observations"),
                                       device=env.device, dtype=torch.float32),
                    args.horizon - 1: torch.as_tensor(norm.normalize(goal, "observations"),
                                                      device=env.device, dtype=torch.float32)}
            # ONE noise draw shared by every NFE at this (episode, step)
            x0 = torch.randn((env.num_envs, args.horizon, model.transition_dim),
                             generator=gen_noise).to(env.device)

            per_nfe_actions = {}
            for n in nfe_list:
                with torch.no_grad():
                    if gaussian:
                        s = model(cond, verbose=False, sort_by_value=False)
                        traj = s.trajectories
                    else:
                        traj, d = euler_sample(model, cond, x0, n, want_diag=True)
                        if d["update_norm"]:
                            dg[n]["update_norm"].append(float(np.mean(d["update_norm"])))
                        if d["cos_consecutive"]:
                            dg[n]["cos"].append(float(np.mean(d["cos_consecutive"])))
                obs_part = norm.unnormalize(
                    utils.to_np(traj)[:, :, model.action_dim:], "observations")
                per_nfe_actions[n] = norm.unnormalize(
                    utils.to_np(traj)[:, :, :model.action_dim], "actions")[:, 0]
                gen = obs_part.reshape(env.num_envs, args.horizon, 2, N_PER_VIEW, 10)[:, 1, 0]
                rows[n].append(gen)

            cur = observation.reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]
            # advance with the LOWEST-NFE action so every arm sees one shared trajectory
            obs, _, _, _ = env.step(per_nfe_actions[nfe_list[0]])
            real = obs["achieved_goal"].reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]

            for n in nfe_list:
                g = rows[n][-1]
                rows[n][-1] = [{**block_errors(g[e], real[e], mu, sd),
                                "chamfer": chamfer_position(g[e], real[e])}
                               for e in range(min(cli.episodes, env.num_envs))]
            copy_rows += [{**block_errors(cur[e], real[e], mu, sd),
                           "chamfer": chamfer_position(cur[e], real[e])}
                          for e in range(min(cli.episodes, env.num_envs))]

        for n in nfe_list:
            flat = [r for chunk in rows[n] for r in chunk]
            results[f"{tag}_nfe{n}"] = {
                "chamfer": [r["chamfer"] for r in flat],
                "pos_z": [r["pos"] for r in flat],
                "vis_z": [r["vis"] for r in flat], "n": len(flat)}
            if not gaussian:
                diags[f"{tag}_nfe{n}"] = {
                    "mean_update_norm": float(np.mean(dg[n]["update_norm"])) if dg[n]["update_norm"] else None,
                    "mean_cos_consecutive": float(np.mean(dg[n]["cos"])) if dg[n]["cos"] else None}
            print(f"  {tag}_nfe{n:<3d} chamfer={np.mean(results[f'{tag}_nfe{n}']['chamfer']):.5f} "
                  f"pos_z={np.mean(results[f'{tag}_nfe{n}']['pos_z']):.4f}", flush=True)
        results[f"{tag}_copy"] = {"chamfer": [r["chamfer"] for r in copy_rows],
                                  "pos_z": [r["pos"] for r in copy_rows], "n": len(copy_rows)}

    for s, (base, path, sha) in FLOW_SEEDS.items():
        rollout(f"flow_s{s}", base, path, NFES, sha=sha)
    rollout("gaussian", GAUSSIAN[0], GAUSSIAN[1], [100], gaussian=True)

    out = {"results": results, "vector_field_diagnostics": diags,
           "protocol": {"episode_set": ep["sha256"], "episodes": cli.episodes,
                        "rollout_steps": cli.steps, "noise_seed": cli.noise_seed,
                        "nfes": NFES,
                        "noise_control": ("one x0 drawn per (episode, rollout step) and "
                                          "shared by every NFE; env advanced with the "
                                          "lowest-NFE action so all arms share one trajectory")}}
    with open("experiments/loss_balance_audit/nfe_imagination_sweep.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote experiments/loss_balance_audit/nfe_imagination_sweep.json")


if __name__ == "__main__":
    main()
