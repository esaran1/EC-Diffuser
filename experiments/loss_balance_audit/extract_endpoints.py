"""Cache the LATENT endpoints (not just chamfer scalars) for the manifold audit.

Deterministic replay of converged_reference_gt.py's exact protocol: same frozen
episode set, same noise seed 777, same env advance (Euler@8 action). Verifies
reproduction by recomputing chamfer and comparing to the cached scalars.
"""
import json, pickle, sys
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit"); sys.path.insert(0, "experiments/scripts")
from latent_metric import chamfer_position
import solvers
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import array_to_state_dict

N_PER_VIEW = 24
SEEDS = {42: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
         43: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed43",
         44: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed44"}
GAUSSIAN = ("ecdiffuser-data/pretrained_models",
            "diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100")
ARMS = [("euler", 8), ("euler", 16), ("euler", 512)]


class Args:
    env_config_dir="env_config/generalization_num_cubes"; dataset="panda_push"
    num_entity=3; horizon=5; max_episode_length=100; planning_only=True
    push_t=False; multiview=True; verbose=False; seed=42; device="cuda:0"
    preprocess_fns=[]; push_t_num_color=1


def main():
    args = Args(); utils.set_global_device(args.device)
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    ep = pickle.load(open("experiments/isaacgym_episode_sets/replicate0_n96.pkl", "rb"))
    keys = ep["keys"]
    store = {}

    def run(tag, base, path, gaussian=False):
        exp = utils.load_diffusion(base, args.dataset, path, epoch="latest", seed=42,
            is_diffusion=True, override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/"
                                                     "panda_push_replay_buffer_dlp.pkl")
        model = exp.ema; model.eval(); norm = exp.dataset.normalizer
        idx = list(range(16)); idx += [idx[-1]] * (env.num_envs - len(idx))
        obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, env.device),
                        set_goal_states=array_to_state_dict(ep["goal"][idx], keys, env.device))
        gen = torch.Generator(device="cpu").manual_seed(777)
        acc = {}
        for rs in range(6):
            observation = obs["achieved_goal"].reshape(env.num_envs, -1)
            goal = obs["desired_goal"].reshape(env.num_envs, -1)
            cond = {0: torch.as_tensor(norm.normalize(observation, "observations"),
                                       device=env.device, dtype=torch.float32),
                    4: torch.as_tensor(norm.normalize(goal, "observations"),
                                       device=env.device, dtype=torch.float32)}
            x0 = torch.randn((env.num_envs, 5, model.transition_dim), generator=gen).to(env.device)
            step_act = None
            with torch.no_grad():
                if gaussian:
                    traj = model(cond, verbose=False, sort_by_value=False).trajectories
                    ob = norm.unnormalize(utils.to_np(traj)[:, :, model.action_dim:], "observations")
                    acc.setdefault("gaussian100", []).append(
                        ob.reshape(env.num_envs, 5, 2, N_PER_VIEW, 10)[:16, 1, 0])
                    step_act = norm.unnormalize(utils.to_np(traj)[:, :, :model.action_dim], "actions")[:, 0]
                else:
                    for arm in ARMS:
                        xe, _, _ = solvers.integrate(model, cond, x0, *arm)
                        ob = norm.unnormalize(utils.to_np(xe)[:, :, model.action_dim:], "observations")
                        acc.setdefault(f"{arm[0]}{arm[1]}", []).append(
                            ob.reshape(env.num_envs, 5, 2, N_PER_VIEW, 10)[:16, 1, 0])
                        if arm == ARMS[0]:
                            step_act = norm.unnormalize(
                                utils.to_np(xe)[:, :, :model.action_dim], "actions")[:, 0]
            cur = observation.reshape(env.num_envs, 2, N_PER_VIEW, 10)[:16, 0]
            acc.setdefault("current", []).append(cur)
            obs, _, _, _ = env.step(step_act)
            real = obs["achieved_goal"].reshape(env.num_envs, 2, N_PER_VIEW, 10)[:16, 0]
            acc.setdefault("real_future", []).append(real)
            print(f"  {tag} step {rs}", flush=True)
        for k, v in acc.items():
            store[f"{tag}_{k}"] = np.concatenate(v, 0).astype(np.float32)

    for s, p in SEEDS.items():
        run(f"s{s}", "data", p)
    run("gauss", GAUSSIAN[0], GAUSSIAN[1], gaussian=True)

    np.savez_compressed("experiments/loss_balance_audit/cached_endpoints.npz", **store)
    # verify exact reproduction against the cached scalars
    prev = json.load(open("experiments/loss_balance_audit/converged_reference_gt.json"))
    print("\n=== reproduction check vs converged_reference_gt.json ===")
    for s in SEEDS:
        for arm in ["euler8", "euler16", "euler512"]:
            g = store[f"s{s}_{arm}"]; r = store[f"s{s}_real_future"]
            ch = np.array([chamfer_position(g[i], r[i]) for i in range(len(g))])
            old = np.array(prev[f"s{s}_{arm}"])
            print(f"  s{s}_{arm}: max|Δchamfer| = {np.abs(ch - old).max():.3e}")
    print("\nshapes:", {k: v.shape for k, v in list(store.items())[:4]})


if __name__ == "__main__":
    main()
