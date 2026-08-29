"""Ground-truth imagination error of the CONVERGED Flow endpoint (Euler@512),
now that raw-space convergence is validated. 3 seeds, full 96-sample set."""
import json, pickle, sys
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit"); sys.path.insert(0, "experiments/scripts")
from latent_metric import training_stats, chamfer_position
import solvers
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import array_to_state_dict

N_PER_VIEW = 24
SEEDS = {42: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
         43: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed43",
         44: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed44"}
ARMS = [("euler", 8), ("euler", 16), ("euler", 128), ("euler", 512)]


class Args:
    env_config_dir="env_config/generalization_num_cubes"; dataset="panda_push"
    num_entity=3; horizon=5; max_episode_length=100; planning_only=True
    push_t=False; multiview=True; verbose=False; seed=42; device="cuda:0"
    preprocess_fns=[]; push_t_num_color=1


def main():
    args = Args(); utils.set_global_device(args.device)
    raw = pickle.load(open("ecdiffuser-data/push_cubes/3C_randcolor/"
                           "panda_push_replay_buffer_dlp.pkl", "rb"))["observations"][:400]
    mu, sd = training_stats(raw.reshape(-1, 48, 10))
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    ep = pickle.load(open("experiments/isaacgym_episode_sets/replicate0_n96.pkl", "rb"))
    keys = ep["keys"]; res = {}
    for seed, path in SEEDS.items():
        exp = utils.load_diffusion("data", args.dataset, path, epoch="latest", seed=42,
            is_diffusion=True, override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/"
                                                     "panda_push_replay_buffer_dlp.pkl")
        model = exp.ema; model.eval(); norm = exp.dataset.normalizer
        idx = list(range(16)); idx += [idx[-1]] * (env.num_envs - len(idx))
        obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, env.device),
                        set_goal_states=array_to_state_dict(ep["goal"][idx], keys, env.device))
        gen = torch.Generator(device="cpu").manual_seed(777)
        rows = {a: [] for a in ARMS}
        for rs in range(6):
            observation = obs["achieved_goal"].reshape(env.num_envs, -1)
            goal = obs["desired_goal"].reshape(env.num_envs, -1)
            cond = {0: torch.as_tensor(norm.normalize(observation, "observations"),
                                       device=env.device, dtype=torch.float32),
                    4: torch.as_tensor(norm.normalize(goal, "observations"),
                                       device=env.device, dtype=torch.float32)}
            x0 = torch.randn((env.num_envs, 5, model.transition_dim), generator=gen).to(env.device)
            acts = {}
            with torch.no_grad():
                for arm in ARMS:
                    xe, _, _ = solvers.integrate(model, cond, x0, *arm)
                    ob = norm.unnormalize(utils.to_np(xe)[:, :, model.action_dim:], "observations")
                    rows[arm].append(ob.reshape(env.num_envs, 5, 2, N_PER_VIEW, 10)[:, 1, 0])
                    acts[arm] = norm.unnormalize(utils.to_np(xe)[:, :, :model.action_dim], "actions")[:, 0]
            obs, _, _, _ = env.step(acts[ARMS[0]])
            real = obs["achieved_goal"].reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]
            for arm in ARMS:
                g = rows[arm][-1]
                rows[arm][-1] = [chamfer_position(g[e], real[e]) for e in range(16)]
            print(f"  s{seed} step {rs}", flush=True)
        for arm in ARMS:
            flat = [float(v) for c in rows[arm] for v in c]
            res[f"s{seed}_{arm[0]}{arm[1]}"] = flat
            print(f"    s{seed} {arm[0]}@{arm[1]}: {np.mean(flat):.5f}", flush=True)
    json.dump(res, open("experiments/loss_balance_audit/converged_reference_gt.json", "w"), indent=2)
    print("\n=== SUMMARY (3-seed mean) ===")
    for arm in ARMS:
        v = [np.mean(res[f"s{s}_{arm[0]}{arm[1]}"]) for s in SEEDS]
        print(f"  {arm[0]}@{arm[1]:4d}: mean={np.mean(v):.5f} sd={np.std(v, ddof=1):.5f}  per-seed={[round(z,5) for z in v]}")


if __name__ == "__main__":
    main()
