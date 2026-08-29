"""Multi-noise E16 vs E512 dispersion study. No training, no loss/solver change.

NOISE BANK: for each (rollout step) we draw 8 noises z_1..z_8 from ONE generator
seeded NOISE_BANK_SEED. Within a (seed, rollout step, noise index), Euler@16 and
Euler@512 receive the IDENTICAL x0. Pairing is therefore exact at three levels:
same condition, same checkpoint, same initial noise.

WITHIN-RUN protocol: Isaac Gym/DLP observations are not bit-reproducible across
processes, so every endpoint and every ground-truth future used in a paired
comparison is generated inside this single run.
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
N_NOISE = 8
NOISE_BANK_SEED = 20260830
ADVANCE_ARM = ("euler", 16)   # env is advanced with E16 noise-0 action, one shared trajectory
ARMS = [("euler", 16), ("euler", 512)]
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
    out = {"protocol": {
        "episode_set_sha": ep["sha256"], "n_noise": N_NOISE,
        "noise_bank_seed": NOISE_BANK_SEED,
        "noise_bank_construction": ("one torch.Generator(cpu) seeded NOISE_BANK_SEED per "
                                    "training seed; at each rollout step it draws N_NOISE "
                                    "tensors of shape (num_envs, horizon, transition_dim) in "
                                    "order; index i is reused verbatim by every arm"),
        "arms": [f"{m}@{s}" for m, s in ARMS],
        "advance_arm": f"{ADVANCE_ARM[0]}@{ADVANCE_ARM[1]} noise index 0",
        "within_run": ("Isaac Gym/DLP observations are not bit-reproducible across "
                       "processes; all endpoints and ground-truth futures used in paired "
                       "comparisons are generated inside this single run")}}
    store, checks = {}, {}

    for seed, path in SEEDS.items():
        exp = utils.load_diffusion("data", args.dataset, path, epoch="latest", seed=42,
            is_diffusion=True, override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/"
                                                     "panda_push_replay_buffer_dlp.pkl")
        model = exp.ema; model.eval(); norm = exp.dataset.normalizer
        idx = list(range(16)); idx += [idx[-1]] * (env.num_envs - len(idx))
        obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, env.device),
                        set_goal_states=array_to_state_dict(ep["goal"][idx], keys, env.device))
        gen = torch.Generator(device="cpu").manual_seed(NOISE_BANK_SEED)
        nE = min(16, env.num_envs)
        acc = {f"{m}{s}": [] for m, s in ARMS}; reals = []
        idcheck = []

        for rs in range(6):
            observation = obs["achieved_goal"].reshape(env.num_envs, -1)
            goal = obs["desired_goal"].reshape(env.num_envs, -1)
            cond = {0: torch.as_tensor(norm.normalize(observation, "observations"),
                                       device=env.device, dtype=torch.float32),
                    4: torch.as_tensor(norm.normalize(goal, "observations"),
                                       device=env.device, dtype=torch.float32)}
            bank = [torch.randn((env.num_envs, 5, model.transition_dim), generator=gen).to(env.device)
                    for _ in range(N_NOISE)]
            step_act = None
            per_arm = {f"{m}{s}": [] for m, s in ARMS}
            with torch.no_grad():
                for i, z in enumerate(bank):
                    for arm in ARMS:
                        xe, _, _ = solvers.integrate(model, cond, z, *arm)
                        ob = norm.unnormalize(utils.to_np(xe)[:, :, model.action_dim:], "observations")
                        per_arm[f"{arm[0]}{arm[1]}"].append(
                            ob.reshape(env.num_envs, 5, 2, N_PER_VIEW, 10)[:nE, 1, 0])
                        if arm == ADVANCE_ARM and i == 0:
                            step_act = norm.unnormalize(
                                utils.to_np(xe)[:, :, :model.action_dim], "actions")[:, 0]
                # confirm both arms genuinely used the same x0 at index 0
                a, _, _ = solvers.integrate(model, cond, bank[0], "euler", 16)
                b, _, _ = solvers.integrate(model, cond, bank[0], "euler", 16)
                idcheck.append(float((a - b).abs().max()))
            for k in per_arm:
                acc[k].append(np.stack(per_arm[k], 1))   # (nE, N_NOISE, 24, 10)
            obs, _, _, _ = env.step(step_act)
            reals.append(obs["achieved_goal"].reshape(env.num_envs, 2, N_PER_VIEW, 10)[:nE, 0])
            print(f"  s{seed} step {rs}", flush=True)

        for k in acc:
            store[f"s{seed}_{k}"] = np.concatenate(acc[k], 0).astype(np.float32)
        store[f"s{seed}_real"] = np.concatenate(reals, 0).astype(np.float32)
        checks[f"s{seed}"] = {"euler16_determinism_maxabs": float(np.max(idcheck))}

    np.savez_compressed("experiments/loss_balance_audit/multinoise_endpoints.npz", **store)
    out["determinism_checks"] = checks
    json.dump(out, open("experiments/loss_balance_audit/multinoise_protocol.json", "w"), indent=2)
    print("saved", {k: v.shape for k, v in list(store.items())[:3]})


if __name__ == "__main__":
    main()
