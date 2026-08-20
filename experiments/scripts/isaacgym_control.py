"""Isaac Gym PushCube control: Gaussian positive control and matched Flow arm.

Records one fixed episode set (initial + goal cube states), hashes it, and
evaluates any arm on exactly that set. Both arms therefore see identical
episodes, which is what makes the Gaussian-versus-Flow comparison paired.

Diagnostics collected per episode implement item 8 of the directive:
approach, contact, displacement, direction, saturation.

Usage:
  python experiments/scripts/isaacgym_control.py --arm gaussian --episodes 32
  python experiments/scripts/isaacgym_control.py --arm flow --episodes 32
"""

import argparse  # noqa: E402
import hashlib
import json
import os
import pickle
import time

# isaacgym binds CUDA before torch does and raises if torch is imported first.
import isaacgym  # noqa: F401,E402  (must precede torch)

import numpy as np
import torch

import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from diffuser.configuration import flow_sampling_kwargs

EPISODE_SET = "experiments/isaacgym_episode_set_v1.pkl"
RESULTS_DIR = "experiments/isaacgym_control"

ARMS = {
    "gaussian": dict(
        loadbase="ecdiffuser-data/pretrained_models",
        loadpath="diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100",
        n_diffusion_steps=100,
    ),
    "flow": dict(
        loadbase="data",
        loadpath="flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
        n_diffusion_steps=4,
    ),
}


class Args:
    """Minimal stand-in for the project ArgsParser, with only the fields used."""

    env_config_dir = "env_config/generalization_num_cubes"
    dataset = "panda_push"
    num_entity = 3
    horizon = 5
    max_episode_length = 100
    planning_only = True
    push_t = False
    multiview = True
    verbose = False
    seed = 42
    device = "cuda:0"
    preprocess_fns = []
    push_t_num_color = 1


def state_dicts_to_array(state_dict):
    """Flatten a {cube_i: (N,13)} dict into (N, num_cubes, 2).

    Only xy is stored because `_set_init_states` consumes exactly `[:2]`; the
    remaining state is regenerated deterministically by the env on reset.
    """
    keys = sorted(state_dict.keys())
    return np.stack(
        [state_dict[k][:, :2].detach().cpu().numpy() for k in keys], axis=1
    ), keys


def record_episode_set(env, num_episodes, seed=12345):
    """Roll random resets once and store the resulting init/goal cube states."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    inits, goals, keys = [], [], None
    needed = int(np.ceil(num_episodes / env.num_envs))
    for _ in range(needed):
        env.reset()
        init_arr, keys = state_dicts_to_array(env.env.get_init_obj_state_dict())
        goal_arr, _ = state_dicts_to_array(env.goal_obs_dict)
        inits.append(init_arr)
        goals.append(goal_arr)

    inits = np.concatenate(inits, axis=0)[:num_episodes]
    goals = np.concatenate(goals, axis=0)[:num_episodes]
    payload = {"init": inits, "goal": goals, "keys": keys, "seed": seed}
    payload["sha256"] = hashlib.sha256(
        inits.tobytes() + goals.tobytes()
    ).hexdigest()
    return payload


def entity_positions(env):
    """(num_envs, num_entities, 3) xyz for [eef, cube1..cubeN] from the raw obs buffer.

    `_get_state_obs` truncates to xy and appends a one-hot, so it cannot answer
    questions about height. The underlying `obs_buf` holds full xyz per entity.
    """
    obs = env.env.obs_dict["obs"].reshape(env.num_envs, env.num_objects + 1, -1)
    return obs[..., :3].detach().cpu().numpy()


def array_to_state_dict(array, keys, device):
    """`_set_init_states` calls `torch.FloatTensor(...)`, which requires host
    memory, so hand it numpy rather than a CUDA tensor."""
    del device  # kept for call-site symmetry
    return {key: np.ascontiguousarray(array[:, i]) for i, key in enumerate(keys)}


def evaluate(policy, env, episode_set, args, arm_name):
    """Run every recorded episode and collect success plus failure diagnostics."""
    keys = episode_set["keys"]
    inits, goals = episode_set["init"], episode_set["goal"]
    n_episodes = len(inits)
    n_envs = env.num_envs
    threshold = 0.04  # dist_threshold from Config.yaml: the cube's effective radius

    records = []
    started = time.time()

    for batch_start in range(0, n_episodes, n_envs):
        batch = slice(batch_start, min(batch_start + n_envs, n_episodes))
        n_active = batch.stop - batch.start
        if n_active < n_envs:  # pad the final partial batch by repeating
            index = list(range(batch.start, batch.stop)) + [batch.stop - 1] * (n_envs - n_active)
        else:
            index = list(range(batch.start, batch.stop))

        obs = env.reset(
            set_init_states=array_to_state_dict(inits[index], keys, env.device),
            set_goal_states=array_to_state_dict(goals[index], keys, env.device),
        )

        # Per-episode diagnostic accumulators
        actions_log = [[] for _ in range(n_envs)]
        eef_log = [[] for _ in range(n_envs)]
        cube_log = [[] for _ in range(n_envs)]
        info_last = None

        step = 0
        while step < env.horizon:
            observation = obs["achieved_goal"].reshape(n_envs, -1)
            goal = obs["desired_goal"].reshape(n_envs, -1)
            conditions = {0: observation, args.horizon - 1: goal}
            _, samples = policy(conditions, batch_size=1, verbose=False)

            action = samples.actions[:, 0]
            obs, _, _, infos = env.step(action)
            info_last = infos

            state = entity_positions(env)  # (N, entities, 3): eef then cubes
            for e in range(n_envs):
                actions_log[e].append(np.asarray(action[e], dtype=np.float64))
                eef_log[e].append(state[e, 0, :3].copy())   # xyz, for the height check
                cube_log[e].append(state[e, 1:, :2].copy()) # xy, to match env.goal_pos
            step += 1

        # env.goal_pos is the xy the environment itself scores against
        # (isaac_env_wrappers.py:233), so distances here match info["avg_obj_dist"].
        goal_state = np.asarray(env.goal_pos)

        for e in range(n_active):
            records.append(
                summarize_episode(
                    episode=index[e],
                    actions=np.array(actions_log[e]),
                    eef=np.array(eef_log[e]),
                    cubes=np.array(cube_log[e]),
                    goal_cubes=goal_state[e],  # env.goal_pos already excludes the eef
                    info=info_last[e],
                    threshold=threshold,
                )
            )
        print(f"  [{arm_name}] {len(records)}/{n_episodes} episodes", flush=True)

    return records, time.time() - started


def summarize_episode(episode, actions, eef, cubes, goal_cubes, info, threshold):
    """Item 8 diagnostics for one episode.

    cubes: (T, n_cubes, 3) trajectory. goal_cubes: (n_cubes, 3).
    """
    start = cubes[0]
    final = cubes[-1]

    start_dist = np.linalg.norm(start - goal_cubes, axis=-1)
    final_dist = np.linalg.norm(final - goal_cubes, axis=-1)
    displacement = np.linalg.norm(final - start, axis=-1)

    # Did a cube move toward its goal? Positive means it closed distance.
    progress = start_dist - final_dist

    # Approach: minimum EE-to-cube distance over the episode, per cube.
    ee_to_cube = np.linalg.norm(cubes - eef[:, None, :2], axis=-1)  # (T, n_cubes), xy
    min_ee_dist = ee_to_cube.min(axis=0)

    # Contact proxy: the EE came within a cube radius AND the cube then moved.
    contact_radius = 0.06
    contacted = (ee_to_cube < contact_radius).any(axis=0) & (displacement > 1e-3)
    first_contact = None
    close = np.where((ee_to_cube < contact_radius).any(axis=1))[0]
    if close.size:
        first_contact = int(close[0])

    return {
        "episode": int(episode),
        "success": float(info["goal_success_frac"] == 1),
        "goal_success_frac": float(info["goal_success_frac"]),
        "avg_obj_dist": float(info["avg_obj_dist"]),
        "max_obj_dist": float(info["max_obj_dist"]),
        "cubes_placed": int((final_dist < threshold).sum()),
        "cubes_moved": int((displacement > 5e-3).sum()),
        "mean_displacement": float(displacement.mean()),
        "max_displacement": float(displacement.max()),
        "mean_progress": float(progress.mean()),
        "cubes_closer": int((progress > 1e-3).sum()),
        "cubes_farther": int((progress < -1e-3).sum()),
        "min_ee_to_cube": float(min_ee_dist.min()),
        "n_contacted": int(contacted.sum()),
        "first_contact_step": first_contact,
        "action_abs_mean": float(np.abs(actions).mean()),
        "clip_fraction": float((np.abs(actions) > 0.99).mean()),
        "action_z_mean": float(actions[:, 2].mean()),
        "eef_z_start": float(eef[0, 2]),
        "eef_z_end": float(eef[-1, 2]),
        "eef_path_length": float(np.linalg.norm(np.diff(eef, axis=0), axis=-1).sum()),
    }


def aggregate(records):
    def mean(key):
        return float(np.mean([r[key] for r in records]))

    successes = sum(r["success"] for r in records)
    n = len(records)
    # Clopper-Pearson upper bound for a zero or low count
    from scipy.stats import beta

    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(0.975, successes + 1, n - successes))

    return {
        "episodes": n,
        "successes": int(successes),
        "success_rate": successes / n,
        "success_ci95": [lower, upper],
        "goal_success_frac": mean("goal_success_frac"),
        "avg_obj_dist": mean("avg_obj_dist"),
        "cubes_placed": mean("cubes_placed"),
        "cubes_moved": mean("cubes_moved"),
        "mean_displacement": mean("mean_displacement"),
        "mean_progress": mean("mean_progress"),
        "cubes_closer": mean("cubes_closer"),
        "cubes_farther": mean("cubes_farther"),
        "min_ee_to_cube": mean("min_ee_to_cube"),
        "n_contacted": mean("n_contacted"),
        "contact_rate": float(np.mean([r["n_contacted"] > 0 for r in records])),
        "action_abs_mean": mean("action_abs_mean"),
        "clip_fraction": mean("clip_fraction"),
        "action_z_mean": mean("action_z_mean"),
        "eef_z_start": mean("eef_z_start"),
        "eef_z_end": mean("eef_z_end"),
        "eef_path_length": mean("eef_path_length"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument(
        "--epoch", default="latest",
        help="checkpoint step to load, e.g. 100000; 'latest' uses the final one",
    )
    parser.add_argument("--tag", default="", help="suffix for the results filename")
    args_cli = parser.parse_args()

    args = Args()
    spec = ARMS[args_cli.arm]
    utils.set_global_device(args.device)

    env = setup_isaac_env(args)
    env.horizon = args.max_episode_length

    if os.path.exists(EPISODE_SET):
        with open(EPISODE_SET, "rb") as handle:
            episode_set = pickle.load(handle)
        print(f"Loaded episode set {episode_set['sha256'][:16]} ({len(episode_set['init'])} episodes)")
    else:
        episode_set = record_episode_set(env, args_cli.episodes)
        os.makedirs(os.path.dirname(EPISODE_SET), exist_ok=True)
        with open(EPISODE_SET, "wb") as handle:
            pickle.dump(episode_set, handle)
        print(f"Recorded episode set {episode_set['sha256'][:16]} ({len(episode_set['init'])} episodes)")

    experiment = utils.load_diffusion(
        spec["loadbase"], args.dataset, spec["loadpath"],
        epoch=args_cli.epoch, seed=args.seed, is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl",
    )
    diffusion = experiment.ema
    policy = utils.Config(
        "sampling.GoalConditionedPolicy",
        diffusion_model=diffusion,
        normalizer=experiment.dataset.normalizer,
        preprocess_fns=[],
        verbose=False,
        horizon=args.horizon,
        count_denoiser_calls=True,
        **flow_sampling_kwargs(diffusion, spec["n_diffusion_steps"]),
    )()

    records, elapsed = evaluate(policy, env, episode_set, args, args_cli.arm)
    summary = aggregate(records)
    summary.update(
        arm=args_cli.arm,
        checkpoint=os.path.join(spec["loadbase"], args.dataset, spec["loadpath"]),
        n_diffusion_steps=spec["n_diffusion_steps"],
        denoiser_calls_per_plan=policy.denoiser_calls / max(policy.planner_calls, 1),
        episode_set_sha256=episode_set["sha256"],
        wall_seconds=elapsed,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary["epoch"] = args_cli.epoch
    name = f"{args_cli.arm}{args_cli.tag}_results.json"
    out = os.path.join(RESULTS_DIR, name)
    with open(out, "w") as handle:
        json.dump({"summary": summary, "episodes": records}, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
