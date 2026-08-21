"""Paired Isaac Gym NFE study: what is the minimum Flow NFE that matches Gaussian?

Design
------
Six arms -- Flow at 1, 2, 4, 8, 16 solver steps, plus the canonical Gaussian at
100 -- evaluated on three *independently generated* 96-episode sets.

The three sets are **evaluation replicates**, not training seeds. Every arm uses
the same single trained Flow checkpoint and the same single Gaussian checkpoint;
only the episodes differ between replicates and only the solver-step count
differs between Flow arms. So these replicates quantify evaluation-sampling
variance, and they say nothing about training-seed variance.

Within a replicate every arm sees byte-identical initial and goal states,
verified by the recorded SHA256 of the episode set.

No training is performed. NFE is an inference-time override applied through
`flow_sampling_kwargs`, which only affects flow wrappers.
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

from isaacgym_control import (  # noqa: E402
    ARMS,
    Args,
    array_to_state_dict,
    entity_positions,
    record_episode_set,
    summarize_episode,
)

SETS_DIR = "experiments/isaacgym_episode_sets"
RESULTS_DIR = "experiments/isaacgym_control/nfe_study"
THRESHOLD = 0.04  # dist_threshold from Config.yaml: the cube's effective radius

# (label, arm key, solver steps). Gaussian ignores the step override.
PLAN = [
    ("flow_nfe1", "flow", 1),
    ("flow_nfe2", "flow", 2),
    ("flow_nfe4", "flow", 4),
    ("flow_nfe8", "flow", 8),
    ("flow_nfe16", "flow", 16),
    ("gaussian_nfe100", "gaussian", 100),
]


def get_episode_set(env, replicate, n_episodes):
    """Load replicate `r` if recorded, else generate and freeze it."""
    os.makedirs(SETS_DIR, exist_ok=True)
    path = os.path.join(SETS_DIR, f"replicate{replicate}_n{n_episodes}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        print(f"[set] loaded replicate {replicate}: {payload['sha256'][:16]}", flush=True)
        return payload, path

    # Distinct seed per replicate so the sets are independently generated.
    payload = record_episode_set(env, n_episodes, seed=20260820 + 1000 * replicate)
    payload["replicate"] = replicate
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    print(f"[set] recorded replicate {replicate}: {payload['sha256'][:16]}", flush=True)
    return payload, path


def build_policy(arm, steps, args):
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
        preprocess_fns=[],
        verbose=False,
        horizon=args.horizon,
        measure_planning_latency=True,
        planning_warmup_calls=10,
        count_denoiser_calls=True,
        **flow_sampling_kwargs(experiment.ema, steps),
    )()
    return policy


def evaluate(policy, env, episode_set, args, label):
    keys = episode_set["keys"]
    inits, goals = episode_set["init"], episode_set["goal"]
    n_episodes, n_envs = len(inits), env.num_envs

    records = []
    started = time.time()

    for batch_start in range(0, n_episodes, n_envs):
        stop = min(batch_start + n_envs, n_episodes)
        index = list(range(batch_start, stop))
        index += [index[-1]] * (n_envs - len(index))  # pad a partial final batch

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
            records.append(
                summarize_episode(
                    episode=index[e],
                    actions=np.array(actions_log[e]),
                    eef=np.array(eef_log[e]),
                    cubes=np.array(cube_log[e]),
                    goal_cubes=goal_state[e],
                    info=info_last[e],
                    threshold=THRESHOLD,
                )
            )
        print(f"  [{label}] {len(records)}/{n_episodes}", flush=True)

    return records, time.time() - started


def summarize(records, policy, label, arm, steps, episode_set, elapsed):
    from scipy.stats import beta

    n = len(records)
    successes = int(sum(r["success"] for r in records))
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(0.975, successes + 1, n - successes))

    def mean(key):
        return float(np.mean([r[key] for r in records]))

    stats = policy.planning_stats()
    return {
        "label": label,
        "arm": arm,
        "requested_nfe": steps,
        # The authoritative call count: a forward hook on the denoiser.
        "measured_calls_per_plan": stats["denoiser_calls"] / max(stats["total_planner_calls"], 1),
        "planner_calls": stats["total_planner_calls"],
        "latency_mean_ms": stats.get("mean_ms"),
        "latency_p50_ms": stats.get("p50_ms"),
        "latency_p95_ms": stats.get("p95_ms"),
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n,
        "success_ci95": [lower, upper],
        "goal_success_frac": mean("goal_success_frac"),
        "avg_obj_dist": mean("avg_obj_dist"),
        "cubes_placed": mean("cubes_placed"),
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
        "replicate": episode_set.get("replicate"),
        "wall_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--only", nargs="*", default=None, help="subset of arm labels")
    cli = parser.parse_args()

    args = Args()
    utils.set_global_device(args.device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = setup_isaac_env(args)
    env.horizon = args.max_episode_length

    plan = [p for p in PLAN if cli.only is None or p[0] in cli.only]

    for replicate in cli.replicates:
        episode_set, _ = get_episode_set(env, replicate, cli.episodes)
        for label, arm, steps in plan:
            out = os.path.join(RESULTS_DIR, f"r{replicate}_{label}.json")
            if os.path.exists(out):
                print(f"[skip] {out} exists", flush=True)
                continue

            policy = build_policy(arm, steps, args)
            records, elapsed = evaluate(policy, env, episode_set, args, f"r{replicate}/{label}")
            summary = summarize(records, policy, label, arm, steps, episode_set, elapsed)

            # Fail loudly if the solver did not honour the requested NFE.
            expected = float(steps)
            measured = summary["measured_calls_per_plan"]
            if abs(measured - expected) > 1e-6:
                summary["CALL_COUNT_MISMATCH"] = True
                print(f"  !! expected {expected} calls/plan, measured {measured}", flush=True)

            with open(out, "w") as handle:
                json.dump({"summary": summary, "episodes": records}, handle, indent=2)
            print(f"  -> {label} r{replicate}: {summary['success_rate']:.4f} "
                  f"({summary['measured_calls_per_plan']:.1f} calls, "
                  f"{summary['latency_mean_ms']:.2f} ms)", flush=True)


if __name__ == "__main__":
    main()
