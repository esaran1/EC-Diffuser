"""Evaluate the canonical Flow seed-43 checkpoint on the frozen H=100 matrix.

Six arms: Flow@1 and Flow@4 at 3, 4 and 5 cubes, all at H=100.

The 3-cube task uses all three frozen replicate sets, matching how the seed-42
gap was computed (n=288 pooled), so the two seeds are comparable task by task.
4- and 5-cube use their single frozen 96-episode sets.

Every episode set is LOADED from disk, never regenerated, and its hash is
checked against the set seed 42 used for that object count.
"""

import argparse  # noqa: E402
import hashlib
import json
import os
import pickle
import time

import isaacgym  # noqa: F401,E402  (must precede torch)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import diffuser.utils as utils  # noqa: E402
from diffuser.configuration import flow_sampling_kwargs  # noqa: E402
from diffuser.eval_utils import setup_isaac_env  # noqa: E402

from isaacgym_control import array_to_state_dict, entity_positions  # noqa: E402
from isaacgym_fourcube_probe import summarize_episode  # noqa: E402

CKPT_DIR = ("/home/jren313/ecdiff-seed44-7506ce48/data/panda_push/flow/"
            "3C_dlp_adalnpint_randcolor_H5_T4_seed44")
CKPT_SHA = "c2c13f557aca7cf0eeb29b7baa572cf62a0027021e90ef75a7ca059b3f0e2bd3"
OUT_DIR = "experiments/isaacgym_control/seed44"
HORIZON = 100  # fixed for every arm in this matrix

# (task key, cubes, [episode-set files], expected hash prefix from seed 42)
TASKS = [
    ("3cube", 3, [f"experiments/isaacgym_episode_sets/replicate{r}_n96.pkl" for r in (0, 1, 2)],
     ["35144910b1471b7b", "0047468fa69c00b8", "586e5b8d2f7c44f8"]),
    ("4cube", 4, ["experiments/isaacgym_control/fourcube/episode_set_4cube.pkl"],
     ["5962c3abb4367eaa"]),
    ("5cube", 5, ["experiments/isaacgym_control/fourcube/episode_set_5cube.pkl"],
     ["f8dff00dfd7b1752"]),
]
NFES = [1, 4]


class Args:
    env_config_dir = "env_config/generalization_num_cubes"
    dataset = "panda_push"
    num_entity = 3
    horizon = 5
    max_episode_length = HORIZON
    planning_only = True
    push_t = False
    multiview = True
    verbose = False
    seed = 42          # evaluation RNG, unrelated to the training seed
    device = "cuda:0"
    preprocess_fns = []
    push_t_num_color = 1


def file_sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def load_set(path, expect_prefix, cubes):
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    recomputed = hashlib.sha256(
        payload["init"].tobytes() + payload["goal"].tobytes()).hexdigest()
    assert recomputed == payload["sha256"], f"{path}: stored hash != recomputed"
    assert recomputed.startswith(expect_prefix), (
        f"{path}: hash {recomputed[:16]} != seed-42 set {expect_prefix}")
    assert payload["init"].shape[1] == cubes, f"{path}: cube count mismatch"
    assert len(payload["init"]) == 96, f"{path}: expected 96 episodes"
    return payload


def evaluate(policy, env, episode_set, args, label, offset):
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
            set_goal_states=array_to_state_dict(goals[index], keys, env.device))
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
            rec = summarize_episode(
                index[e], np.array(actions_log[e]), np.array(eef_log[e]),
                np.array(cube_log[e]), goal_state[e], info_last[e], n_cubes)
            # Globally unique episode id across pooled replicate sets.
            rec["episode"] = offset + index[e]
            rec["episode_set_sha256"] = episode_set["sha256"]
            records.append(rec)
        print(f"  [{label}] {len(records)}/{n_episodes}", flush=True)
    return records, time.time() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    cli = ap.parse_args()

    assert file_sha(f"{CKPT_DIR}/state_400000.pt") == CKPT_SHA, "checkpoint hash mismatch"
    os.makedirs(OUT_DIR, exist_ok=True)
    args = Args()
    utils.set_global_device(args.device)

    gpu_at_start = os.popen(
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
        "--format=csv,noheader").read().strip()

    for task, cubes, set_paths, expect in TASKS:
        if cli.only and task not in cli.only:
            continue
        args.num_entity = cubes
        env = setup_isaac_env(args)
        env.horizon = HORIZON
        assert env.num_objects == cubes, f"env built {env.num_objects}, want {cubes}"
        assert env.horizon == HORIZON, "horizon override failed"
        print(f"[env] cubes={env.num_objects} horizon={env.horizon}", flush=True)

        sets = [load_set(p, e, cubes) for p, e in zip(set_paths, expect)]
        print(f"[sets] {task}: " + ", ".join(s['sha256'][:16] for s in sets), flush=True)

        for nfe in NFES:
            out = f"{OUT_DIR}/seed44_{task}_H{HORIZON}_flow_nfe{nfe}.json"
            if os.path.exists(out):
                print(f"[skip] {out}", flush=True)
                continue
            experiment = utils.load_diffusion(
                "data", args.dataset,
                "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed44",
                epoch="latest", seed=args.seed, is_diffusion=True,
                override_dataset_path=("ecdiffuser-data/push_cubes/3C_randcolor/"
                                       "panda_push_replay_buffer_dlp.pkl"))
            policy = utils.Config(
                "sampling.GoalConditionedPolicy",
                diffusion_model=experiment.ema,
                normalizer=experiment.dataset.normalizer,
                preprocess_fns=[], verbose=False, horizon=args.horizon,
                measure_planning_latency=True, planning_warmup_calls=10,
                count_denoiser_calls=True,
                **flow_sampling_kwargs(experiment.ema, nfe))()

            records = []
            elapsed = 0.0
            for i, s in enumerate(sets):
                r, t = evaluate(policy, env, s, args, f"{task}/nfe{nfe}/set{i}", i * 1000)
                records.extend(r)
                elapsed += t

            st = policy.planning_stats()
            from scipy.stats import beta
            n = len(records)
            k = int(sum(r["success"] for r in records))
            lo = 0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1))
            hi = 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))
            placed = np.array([r["cubes_placed"] for r in records], dtype=float)

            summary = {
                "training_seed": 44, "label": f"flow_nfe{nfe}", "arm": "flow",
                "task": task, "num_cubes": cubes, "horizon": HORIZON,
                "requested_nfe": nfe,
                "measured_calls_per_plan": st["denoiser_calls"] / max(st["total_planner_calls"], 1),
                "checkpoint": f"{CKPT_DIR}/state_400000.pt",
                "checkpoint_sha256": CKPT_SHA,
                "checkpoint_internal_step": 499000, "weights": "ema",
                "episode_set_sha256": [s["sha256"] for s in sets],
                "episodes": n, "successes": k, "success_rate": k / n,
                "success_ci95": [lo, hi],
                "per_object_success": float((placed / cubes).mean()),
                "cubes_placed": float(placed.mean()),
                "latency_mean_ms": st.get("mean_ms"),
                "latency_p50_ms": st.get("p50_ms"),
                "latency_p95_ms": st.get("p95_ms"),
                "wall_seconds": elapsed,
                "gpu_at_start": gpu_at_start,
            }
            if abs(summary["measured_calls_per_plan"] - nfe) > 1e-6:
                summary["CALL_COUNT_MISMATCH"] = True
            with open(out, "w") as fh:
                json.dump({"summary": summary, "episodes": records}, fh, indent=2)
            print(f"  -> {task} nfe{nfe}: succ {k}/{n}={k/n:.4f} "
                  f"per-obj {summary['per_object_success']:.4f} "
                  f"({summary['measured_calls_per_plan']:.1f} calls)", flush=True)
        del env


if __name__ == "__main__":
    main()
