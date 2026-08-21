"""4-cube zero-shot compositional headroom probe.

Question: does low-NFE entity-centric Flow retain useful zero-shot performance
when the task goes from 3 to 4 objects, and how does that compare with Gaussian
EC-Diffuser?

Validity of the test is established in experiments/fourcube_validation.md.
No retraining: both arms use the canonical 3-cube checkpoints unchanged.

Arm set is deliberately minimal (three arms, one episode set) -- this is a
headroom probe, not a replication study.
"""

import argparse  # noqa: E402
import hashlib
import json
import os
import pickle
import time

# isaacgym binds CUDA before torch and raises if torch is imported first.
import isaacgym  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import diffuser.utils as utils  # noqa: E402
from diffuser.configuration import flow_sampling_kwargs  # noqa: E402
from diffuser.eval_utils import setup_isaac_env  # noqa: E402

from isaacgym_control import ARMS, array_to_state_dict, entity_positions  # noqa: E402

RESULTS_DIR = "experiments/isaacgym_control/fourcube"
THRESHOLD = 0.04  # dist_threshold from Config.yaml, absolute per object

# Flow 4 is the recommended operating point from the completed NFE study:
# best on every aggregate metric at 3 cubes, narrowest between-replicate spread.
# Flow 1 is the cheap contrast and the one arm that was significantly worse.
PLAN = [
    ("gaussian_nfe100", "gaussian", 100),
    ("flow_nfe4", "flow", 4),
    ("flow_nfe1", "flow", 1),
]


class Args:
    """Config for the 4-cube task. num_entity is the only substantive change."""

    env_config_dir = "env_config/generalization_num_cubes"
    dataset = "panda_push"
    num_entity = 4
    horizon = 5
    max_episode_length = 150  # entity_to_steps[4], already defined upstream
    planning_only = True
    push_t = False
    multiview = True
    verbose = False
    seed = 42
    device = "cuda:0"
    preprocess_fns = []
    push_t_num_color = 1


def state_dicts_to_array(state_dict):
    keys = sorted(state_dict.keys())
    return np.stack([state_dict[k][:, :2].detach().cpu().numpy() for k in keys], axis=1), keys


def record_episode_set(env, num_episodes, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    inits, goals, keys = [], [], None
    for _ in range(int(np.ceil(num_episodes / env.num_envs))):
        env.reset()
        init_arr, keys = state_dicts_to_array(env.env.get_init_obj_state_dict())
        goal_arr, _ = state_dicts_to_array(env.goal_obs_dict)
        inits.append(init_arr)
        goals.append(goal_arr)
    inits = np.concatenate(inits)[:num_episodes]
    goals = np.concatenate(goals)[:num_episodes]
    payload = {"init": inits, "goal": goals, "keys": keys, "seed": seed,
               "num_cubes": inits.shape[1]}
    payload["sha256"] = hashlib.sha256(inits.tobytes() + goals.tobytes()).hexdigest()
    return payload


def summarize_episode(episode, actions, eef, cubes, goal_cubes, info, n_cubes):
    start, final = cubes[0], cubes[-1]
    start_dist = np.linalg.norm(start - goal_cubes, axis=-1)
    final_dist = np.linalg.norm(final - goal_cubes, axis=-1)
    displacement = np.linalg.norm(final - start, axis=-1)
    progress = start_dist - final_dist

    ee_to_cube = np.linalg.norm(cubes - eef[:, None, :2], axis=-1)
    contact_radius = 0.06
    contacted = (ee_to_cube < contact_radius).any(axis=0) & (displacement > 1e-3)
    close = np.where((ee_to_cube < contact_radius).any(axis=1))[0]

    placed = final_dist < THRESHOLD
    return {
        "episode": int(episode),
        "n_cubes": int(n_cubes),
        "success": float(info["goal_success_frac"] == 1),
        "goal_success_frac": float(info["goal_success_frac"]),
        # Per-object success is first-class here: full success needs all N, so
        # p^3 -> p^4 predicts a drop even at unchanged per-cube competence.
        "per_object_success": float(placed.mean()),
        "cubes_placed": int(placed.sum()),
        "avg_obj_dist": float(info["avg_obj_dist"]),
        "max_obj_dist": float(info["max_obj_dist"]),
        "cubes_moved": int((displacement > 5e-3).sum()),
        "mean_progress": float(progress.mean()),
        "cubes_closer": int((progress > 1e-3).sum()),
        "cubes_farther": int((progress < -1e-3).sum()),
        "min_ee_to_cube": float(ee_to_cube.min()),
        "n_contacted": int(contacted.sum()),
        "first_contact_step": int(close[0]) if close.size else None,
        "action_abs_mean": float(np.abs(actions).mean()),
        "clip_fraction": float((np.abs(actions) > 0.99).mean()),
        "eef_path_length": float(np.linalg.norm(np.diff(eef, axis=0), axis=-1).sum()),
    }


def evaluate(policy, env, episode_set, args, label):
    keys = episode_set["keys"]
    inits, goals = episode_set["init"], episode_set["goal"]
    n_episodes, n_envs = len(inits), env.num_envs
    n_cubes = inits.shape[1]

    records = []
    started = time.time()
    for batch_start in range(0, n_episodes, n_envs):
        stop = min(batch_start + n_envs, n_episodes)
        index = list(range(batch_start, stop))
        index += [index[-1]] * (n_envs - len(index))

        obs = env.reset(
            set_init_states=array_to_state_dict(inits[index], keys, env.device),
            set_goal_states=array_to_state_dict(goals[index], keys, env.device),
        )
        actions_log = [[] for _ in range(n_envs)]
        eef_log = [[] for _ in range(n_envs)]
        cube_log = [[] for _ in range(n_envs)]
        info_last = None

        for _ in range(env.horizon):
            observation = obs["achieved_goal"].reshape(n_envs, -1)
            goal = obs["desired_goal"].reshape(n_envs, -1)
            _, samples = policy({0: observation, args.horizon - 1: goal},
                                batch_size=1, verbose=False)
            action = samples.actions[:, 0]
            obs, _, _, info_last = env.step(action)

            state = entity_positions(env)
            for e in range(n_envs):
                actions_log[e].append(np.asarray(action[e], dtype=np.float64))
                eef_log[e].append(state[e, 0, :3].copy())
                cube_log[e].append(state[e, 1:, :2].copy())

        goal_state = np.asarray(env.goal_pos)
        for e in range(stop - batch_start):
            records.append(summarize_episode(
                index[e], np.array(actions_log[e]), np.array(eef_log[e]),
                np.array(cube_log[e]), goal_state[e], info_last[e], n_cubes))
        print(f"  [{label}] {len(records)}/{n_episodes}", flush=True)
    return records, time.time() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--num-entity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=40404)
    cli = parser.parse_args()

    args = Args()
    args.num_entity = cli.num_entity
    utils.set_global_device(args.device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = setup_isaac_env(args)
    env.horizon = args.max_episode_length
    print(f"[env] num_objects={env.num_objects} horizon={env.horizon}", flush=True)
    assert env.num_objects == cli.num_entity, (
        f"env built {env.num_objects} objects, expected {cli.num_entity}")

    set_path = os.path.join(RESULTS_DIR, f"episode_set_{cli.num_entity}cube.pkl")
    if os.path.exists(set_path):
        with open(set_path, "rb") as handle:
            episode_set = pickle.load(handle)
        print(f"[set] loaded {episode_set['sha256'][:16]} "
              f"({len(episode_set['init'])} ep, {episode_set['num_cubes']} cubes)", flush=True)
    else:
        episode_set = record_episode_set(env, cli.episodes, cli.seed)
        with open(set_path, "wb") as handle:
            pickle.dump(episode_set, handle)
        print(f"[set] recorded {episode_set['sha256'][:16]} "
              f"({len(episode_set['init'])} ep, {episode_set['num_cubes']} cubes)", flush=True)
    assert episode_set["num_cubes"] == cli.num_entity, "episode set cube count mismatch"

    for label, arm, steps in PLAN:
        out = os.path.join(RESULTS_DIR, f"{cli.num_entity}cube_{label}.json")
        if os.path.exists(out):
            print(f"[skip] {out}", flush=True)
            continue

        spec = ARMS[arm]
        experiment = utils.load_diffusion(
            spec["loadbase"], args.dataset, spec["loadpath"],
            epoch="latest", seed=args.seed, is_diffusion=True,
            override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl",
        )
        policy = utils.Config(
            "sampling.GoalConditionedPolicy",
            diffusion_model=experiment.ema,
            normalizer=experiment.dataset.normalizer,
            preprocess_fns=[], verbose=False, horizon=args.horizon,
            measure_planning_latency=True, planning_warmup_calls=10,
            count_denoiser_calls=True,
            **flow_sampling_kwargs(experiment.ema, steps),
        )()

        records, elapsed = evaluate(policy, env, episode_set, args, label)
        stats = policy.planning_stats()
        from scipy.stats import beta

        n = len(records)
        successes = int(sum(r["success"] for r in records))
        lo = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, n - successes + 1))
        hi = 1.0 if successes == n else float(beta.ppf(0.975, successes + 1, n - successes))

        def mean(key):
            return float(np.mean([r[key] for r in records]))

        summary = {
            "label": label, "arm": arm, "requested_nfe": steps,
            "measured_calls_per_plan": stats["denoiser_calls"] / max(stats["total_planner_calls"], 1),
            "latency_mean_ms": stats.get("mean_ms"),
            "latency_per_episode_step_ms": (stats.get("mean_ms") or 0) / env.num_envs,
            "num_cubes": cli.num_entity,
            "episodes": n, "successes": successes,
            "success_rate": successes / n, "success_ci95": [lo, hi],
            "goal_success_frac": mean("goal_success_frac"),
            "per_object_success": mean("per_object_success"),
            "cubes_placed": mean("cubes_placed"),
            "avg_obj_dist": mean("avg_obj_dist"),
            "max_obj_dist": mean("max_obj_dist"),
            "cubes_moved": mean("cubes_moved"),
            "cubes_closer": mean("cubes_closer"),
            "cubes_farther": mean("cubes_farther"),
            "mean_progress": mean("mean_progress"),
            "contact_rate": float(np.mean([r["n_contacted"] > 0 for r in records])),
            "n_contacted": mean("n_contacted"),
            "min_ee_to_cube": mean("min_ee_to_cube"),
            "action_abs_mean": mean("action_abs_mean"),
            "clip_fraction": mean("clip_fraction"),
            "eef_path_length": mean("eef_path_length"),
            "episode_set_sha256": episode_set["sha256"],
            "wall_seconds": elapsed,
        }
        if abs(summary["measured_calls_per_plan"] - steps) > 1e-6:
            summary["CALL_COUNT_MISMATCH"] = True
            print(f"  !! expected {steps} calls, measured "
                  f"{summary['measured_calls_per_plan']}", flush=True)

        with open(out, "w") as handle:
            json.dump({"summary": summary, "episodes": records}, handle, indent=2)
        print(f"  -> {label}: success {summary['success_rate']:.4f} "
              f"per-object {summary['per_object_success']:.4f} "
              f"({summary['measured_calls_per_plan']:.1f} calls)", flush=True)


if __name__ == "__main__":
    main()
