"""Instrumented replay of the multi-noise protocol. IDENTICAL generation; the
only change is how much is written to disk. No training, no solver change.

Protocol preserved exactly from multinoise_solver_bias.py: seeds 42/43/44, EMA,
96 conditions, 8 noises, CPU generator seed 20260830, Euler@16 vs Euler@512 on
the same x0, env advanced by the E16 noise-0 action.
"""
import hashlib, json, pickle, sys
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit"); sys.path.insert(0, "experiments/scripts")
import solvers
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import array_to_state_dict

N_PER_VIEW = 24
N_NOISE = 8
NOISE_BANK_SEED = 20260830
ARMS = [("euler", 16), ("euler", 512)]
ADVANCE = ("euler", 16)
SEEDS = {42: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
         43: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed43",
         44: "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed44"}


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
    S = {}

    for seed, path in SEEDS.items():
        exp = utils.load_diffusion("data", args.dataset, path, epoch="latest", seed=42,
            is_diffusion=True, override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/"
                                                     "panda_push_replay_buffer_dlp.pkl")
        model = exp.ema; model.eval(); norm = exp.dataset.normalizer
        idx = list(range(16)); idx += [idx[-1]] * (env.num_envs - len(idx))
        obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, env.device),
                        set_goal_states=array_to_state_dict(ep["goal"][idx], keys, env.device))
        gen = torch.Generator(device="cpu").manual_seed(NOISE_BANK_SEED)
        nE = 16
        A = {k: [] for k in ["cur_raw", "goal_raw", "cur_norm", "goal_norm", "real_t1_raw",
                             "real_action", "cond_mask", "x0_hash", "cond_id"]}
        for a in ARMS:
            A[f"{a[0]}{a[1]}_full_norm"] = []       # (nE, N_NOISE, H, D) model space
            A[f"{a[0]}{a[1]}_obs_unnorm"] = []      # (nE, N_NOISE, H, 480)
            A[f"{a[0]}{a[1]}_act_unnorm"] = []      # (nE, N_NOISE, H, 3)

        for rs in range(6):
            observation = obs["achieved_goal"].reshape(env.num_envs, -1)
            goal = obs["desired_goal"].reshape(env.num_envs, -1)
            cn = norm.normalize(observation, "observations")
            gn = norm.normalize(goal, "observations")
            cond = {0: torch.as_tensor(cn, device=env.device, dtype=torch.float32),
                    4: torch.as_tensor(gn, device=env.device, dtype=torch.float32)}
            bank = [torch.randn((env.num_envs, 5, model.transition_dim), generator=gen).to(env.device)
                    for _ in range(N_NOISE)]
            cmask = None
            per = {f"{a[0]}{a[1]}": {"full": [], "obs": [], "act": []} for a in ARMS}
            step_act = None
            with torch.no_grad():
                for i, z in enumerate(bank):
                    if cmask is None:
                        cmask = utils.to_np(model._make_conditioning_mask(z, cond))[0]
                    for a in ARMS:
                        xe, _, _ = solvers.integrate(model, cond, z, *a)
                        xn = utils.to_np(xe)
                        k = f"{a[0]}{a[1]}"
                        per[k]["full"].append(xn[:nE])
                        per[k]["obs"].append(norm.unnormalize(xn[:, :, model.action_dim:],
                                                              "observations")[:nE])
                        per[k]["act"].append(norm.unnormalize(xn[:, :, :model.action_dim],
                                                              "actions")[:nE])
                        if a == ADVANCE and i == 0:
                            step_act = norm.unnormalize(xn[:, :, :model.action_dim], "actions")[:, 0]
                    A["x0_hash"].append([hashlib.sha256(
                        np.ascontiguousarray(utils.to_np(z)[e]).tobytes()).hexdigest()[:16]
                        for e in range(nE)])
            for a in ARMS:
                k = f"{a[0]}{a[1]}"
                A[f"{k}_full_norm"].append(np.stack(per[k]["full"], 1))
                A[f"{k}_obs_unnorm"].append(np.stack(per[k]["obs"], 1))
                A[f"{k}_act_unnorm"].append(np.stack(per[k]["act"], 1))
            A["cur_raw"].append(observation[:nE].reshape(nE, 2, N_PER_VIEW, 10))
            A["goal_raw"].append(goal[:nE].reshape(nE, 2, N_PER_VIEW, 10))
            A["cur_norm"].append(cn[:nE]); A["goal_norm"].append(gn[:nE])
            A["cond_mask"] = cmask
            A["cond_id"].append(np.array([f"rs{rs}_ep{e}" for e in range(nE)]))
            obs, _, _, _ = env.step(step_act)
            A["real_t1_raw"].append(
                obs["achieved_goal"].reshape(env.num_envs, 2, N_PER_VIEW, 10)[:nE])
            A["real_action"].append(step_act[:nE])
            print(f"  s{seed} step {rs}", flush=True)

        empty = [k for k, v in A.items() if not isinstance(v, np.ndarray) and len(v) == 0]
        if empty:
            raise RuntimeError(f"unpopulated cache keys, fix before saving: {empty}")
        for k, v in A.items():
            if k == "cond_mask":
                S[f"s{seed}_{k}"] = np.asarray(v)          # already a bare array
            elif k in ("cond_id", "x0_hash"):
                S[f"s{seed}_{k}"] = np.concatenate([np.asarray(x) for x in v], 0)
            else:
                S[f"s{seed}_{k}"] = np.concatenate(v, 0).astype(np.float32)

    np.savez_compressed("experiments/loss_balance_audit/enriched_endpoints.npz", **S)
    print("saved", {k: np.asarray(v).shape for k, v in list(S.items())[:6]})


if __name__ == "__main__":
    main()
