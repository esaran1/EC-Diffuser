"""PHASE 13: run the ACTUAL ground-truth future latent through the exact
imagination-metric path used by every arm, to measure the metric's own floor."""
import json, pickle, sys
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/loss_balance_audit"); sys.path.insert(0, "experiments/scripts")
from latent_metric import training_stats, block_errors, chamfer_position
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import array_to_state_dict

N_PER_VIEW = 24


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
    keys = ep["keys"]
    exp = utils.load_diffusion("data", args.dataset,
        "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42", epoch="latest", seed=42,
        is_diffusion=True, override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/"
                                                 "panda_push_replay_buffer_dlp.pkl")
    model = exp.ema; model.eval(); norm = exp.dataset.normalizer

    idx = list(range(16)); idx += [idx[-1]] * (env.num_envs - len(idx))
    obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, env.device),
                    set_goal_states=array_to_state_dict(ep["goal"][idx], keys, env.device))
    gen = torch.Generator(device="cpu").manual_seed(777)
    perfect, roundtrip, copy_cur = [], [], []
    nE = 16
    for rs in range(6):
        observation = obs["achieved_goal"].reshape(env.num_envs, -1)
        goal = obs["desired_goal"].reshape(env.num_envs, -1)
        cond = {0: torch.as_tensor(norm.normalize(observation, "observations"),
                                   device=env.device, dtype=torch.float32),
                4: torch.as_tensor(norm.normalize(goal, "observations"),
                                   device=env.device, dtype=torch.float32)}
        x0 = torch.randn((env.num_envs, 5, model.transition_dim), generator=gen).to(env.device)
        with torch.no_grad():
            x = x0.clone(); model._apply_conditioning(x, cond)
            cm = model._make_conditioning_mask(x, cond)
            for k in range(8):
                t = x.new_full((x.shape[0],), k / 8)
                x = x + (1/8) * model.model(x, cond, t * model.time_scale) * cm.to(x.dtype)
                model._apply_conditioning(x, cond)
            act = norm.unnormalize(utils.to_np(x)[:, :, :model.action_dim], "actions")[:, 0]
        cur = observation.reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]
        obs, _, _, _ = env.step(act)
        real = obs["achieved_goal"].reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]
        # PERFECT: the true future latent itself, scored against itself
        for e in range(nE):
            perfect.append(chamfer_position(real[e], real[e]))
            copy_cur.append(chamfer_position(cur[e], real[e]))
        # ROUND-TRIP: true future pushed through normalize->unnormalize (the arms' path)
        rt = norm.unnormalize(norm.normalize(
            obs["achieved_goal"].reshape(env.num_envs, -1), "observations"), "observations")
        rt = rt.reshape(env.num_envs, 2, N_PER_VIEW, 10)[:, 0]
        for e in range(nE):
            roundtrip.append(chamfer_position(rt[e], real[e]))
    out = {"perfect_identity": {"mean": float(np.mean(perfect)), "max": float(np.max(perfect)),
                                "n": len(perfect)},
           "perfect_roundtrip_normalize_unnormalize": {"mean": float(np.mean(roundtrip)),
                                                       "max": float(np.max(roundtrip)),
                                                       "n": len(roundtrip)},
           "copy_current": {"mean": float(np.mean(copy_cur)), "n": len(copy_cur)}}
    print(json.dumps(out, indent=2))
    with open("experiments/loss_balance_audit/perfect_future_floor.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
