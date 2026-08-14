#!/usr/bin/env python3
"""Evaluate one frozen generative-policy checkpoint in native OGBench Puzzle."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from diffuser.datasets.benchmark_sequence import TrainSplitNormalizer
from diffuser.models import AuxiliaryIntervalTemporalUnet, IntervalTemporalUnet
from diffuser.scripts.train_phase7_pilot import METHODS, build_method


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalized_conditions(normalizer, observation, goal, device, horizon):
    current = normalizer.normalize(observation, "observations").astype(np.float32)
    target = normalizer.normalize(goal, "goals").astype(np.float32)
    return {
        0: torch.from_numpy(current[None]).to(device),
        horizon - 1: torch.from_numpy(target[None]).to(device),
    }


def backbone_class_for_method(method_name):
    if method_name == "auxiliary_improved_meanflow":
        return AuxiliaryIntervalTemporalUnet
    return IntervalTemporalUnet


def main():
    import ogbench
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-checkpoint-step", type=int)
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("experiments/pilots/ogbench_puzzle_state_extension_v1.json"),
    )
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--evaluation-seed", type=int, default=42000)
    parser.add_argument("--warmup-plans", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.nfe < 1 or args.episodes_per_task < 1 or args.warmup_plans < 0:
        raise ValueError("NFE, episodes, and warmup settings are invalid")
    if not args.task_ids or any(task_id not in range(1, 6) for task_id in args.task_ids):
        raise ValueError("Puzzle task IDs must be in [1, 5]")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        raise ValueError("max episode steps must be positive")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires exactly one visible CUDA device")

    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    task = protocol["task"]
    method_config = protocol["methods"][args.method]
    if args.method == "gaussian_diffusion":
        expected_calls = method_config["training_timesteps"]
        if args.nfe != expected_calls:
            raise ValueError("Gaussian NFE must equal its frozen training timestep count")
    else:
        expected_calls = args.nfe

    device = torch.device("cuda:0")
    set_seed(args.evaluation_seed)
    backbone_config = protocol["backbone"]
    backbone = backbone_class_for_method(args.method)(
        horizon=task["horizon"],
        transition_dim=task["observation_dim"] + task["action_dim"],
        cond_dim=task["observation_dim"],
        dim=backbone_config["dim"],
        dim_mults=tuple(backbone_config["dim_mults"]),
        attention=backbone_config["attention"],
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    expected_parameter_count = method_config.get(
        "parameter_count", backbone_config["parameter_count"]
    )
    if parameter_count != expected_parameter_count:
        raise RuntimeError("backbone parameter count does not match protocol")
    policy = build_method(args.method, backbone, task, method_config).to(device)

    checkpoint_hash = sha256_file(args.checkpoint)
    if (
        args.expected_checkpoint_sha256 is not None
        and checkpoint_hash != args.expected_checkpoint_sha256
    ):
        raise RuntimeError("checkpoint SHA256 does not match the frozen protocol")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if set(checkpoint) != {"step", "model", "ema"}:
        raise RuntimeError("checkpoint does not have the repository Trainer schema")
    checkpoint_step = int(checkpoint["step"])
    if (
        args.expected_checkpoint_step is not None
        and checkpoint_step != args.expected_checkpoint_step
    ):
        raise RuntimeError("checkpoint step does not match the frozen protocol")
    policy.load_state_dict(checkpoint["ema"], strict=True)
    del checkpoint
    policy.eval()

    normalizer_bundle = json.loads(Path(task["normalizer_bundle"]).read_text())
    normalizer_entry = normalizer_bundle["tasks"]["ogbench_puzzle_4x4_play_state"]
    if normalizer_entry["normalizer_sha256"] != task["normalizer_sha256"]:
        raise RuntimeError("normalizer hash does not match protocol")
    normalizer = TrainSplitNormalizer.from_state_dict(normalizer_entry["normalizer"])

    env = ogbench.make_env_and_datasets("puzzle-4x4-play-v0", env_only=True)
    native_horizon = int(env.spec.max_episode_steps)
    episode_horizon = native_horizon
    protocol_label = "NATIVE_FULL_HORIZON"
    if args.max_episode_steps is not None:
        episode_horizon = min(native_horizon, args.max_episode_steps)
        if episode_horizon < native_horizon:
            protocol_label = "BOUNDED_INTEGRATION_ONLY"

    forward_calls = {"count": 0}
    hook = backbone.register_forward_hook(
        lambda _module, _inputs, _output: forward_calls.__setitem__(
            "count", forward_calls["count"] + 1
        )
    )

    def plan(observation, goal):
        conditions = normalized_conditions(
            normalizer, observation, goal, device, task["horizon"]
        )
        calls_before = forward_calls["count"]
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            if args.method == "gaussian_diffusion":
                sample = policy(conditions, verbose=False)
            else:
                sample = policy(conditions, n_steps=args.nfe, verbose=False)
        torch.cuda.synchronize(device)
        latency = time.perf_counter() - start
        observed_calls = forward_calls["count"] - calls_before
        if observed_calls != expected_calls:
            raise RuntimeError(
                "requested {} NFEs but observed {} model calls".format(
                    expected_calls, observed_calls
                )
            )
        trajectory = sample.trajectories
        if trajectory.shape != (1, task["horizon"], task["observation_dim"] + task["action_dim"]):
            raise RuntimeError("planner returned an unexpected trajectory shape")
        if not torch.isfinite(trajectory).all():
            raise FloatingPointError("planner returned non-finite values")
        torch.testing.assert_close(
            trajectory[:, 0, task["action_dim"]:], conditions[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            trajectory[:, -1, task["action_dim"]:],
            conditions[task["horizon"] - 1], rtol=0, atol=0,
        )
        normalized_action = trajectory[0, 0, :task["action_dim"]].cpu().numpy()
        action = normalizer.unnormalize(normalized_action, "actions")
        unclipped = action.copy()
        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        clipped = bool(np.any(action != unclipped))
        return action, latency, observed_calls, clipped

    first_observation, first_info = env.reset(
        seed=args.evaluation_seed, options={"task_id": args.task_ids[0]}
    )
    for warmup_index in range(args.warmup_plans):
        set_seed(args.evaluation_seed - 1000 + warmup_index)
        plan(first_observation, first_info["goal"])

    set_seed(args.evaluation_seed)
    torch.cuda.reset_peak_memory_stats(device)
    episodes = []
    all_latencies = []
    evaluation_start = time.perf_counter()
    for task_id in args.task_ids:
        for episode_index in range(args.episodes_per_task):
            seed = args.evaluation_seed + task_id * 10000 + episode_index
            set_seed(seed)
            observation, info = env.reset(seed=seed, options={"task_id": task_id})
            goal = np.asarray(info["goal"])
            episode_return = 0.0
            clipped_actions = 0
            success = False
            terminated = False
            truncated = False
            steps = 0
            calls = 0
            for steps in range(1, episode_horizon + 1):
                action, latency, observed_calls, clipped = plan(observation, goal)
                all_latencies.append(latency)
                calls += observed_calls
                clipped_actions += int(clipped)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                success = bool(info.get("success", terminated))
                if terminated or truncated:
                    break
            episodes.append({
                "task_id": task_id,
                "episode_index": episode_index,
                "seed": seed,
                "steps": steps,
                "success": success,
                "return": episode_return,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "model_calls": calls,
                "clipped_action_steps": clipped_actions,
            })
    torch.cuda.synchronize(device)
    runtime = time.perf_counter() - evaluation_start
    hook.remove()
    env.close()

    successes = sum(record["success"] for record in episodes)
    result = {
        "schema_version": "ogbench-native-generative-policy-eval-v1",
        "status": "PASS",
        "protocol_label": protocol_label,
        "method": args.method,
        "wrapper": type(policy).__name__,
        "backbone": type(backbone).__name__,
        "parameter_count": parameter_count,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_weights": "ema",
        "protocol": str(args.protocol),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "normalizer_sha256": normalizer_entry["normalizer_sha256"],
        "dataset_manifest": task["dataset_manifest"],
        "ogbench_version": importlib.metadata.version("ogbench"),
        "environment": "puzzle-4x4-play-v0",
        "task_ids": args.task_ids,
        "episodes_per_task": args.episodes_per_task,
        "episodes": len(episodes),
        "environment_steps": sum(record["steps"] for record in episodes),
        "native_episode_horizon": native_horizon,
        "executed_episode_horizon": episode_horizon,
        "evaluation_seed": args.evaluation_seed,
        "requested_nfe": args.nfe,
        "verified_calls_per_plan": expected_calls,
        "full_successes": int(successes),
        "success_rate": successes / len(episodes),
        "mean_return": float(np.mean([record["return"] for record in episodes])),
        "action_clip_step_fraction": sum(
            record["clipped_action_steps"] for record in episodes
        ) / sum(record["steps"] for record in episodes),
        "planning_latency": {
            "n": len(all_latencies),
            "mean_seconds": float(statistics.fmean(all_latencies)),
            "std_seconds": float(statistics.stdev(all_latencies)) if len(all_latencies) > 1 else 0.0,
            "p50_seconds": percentile(all_latencies, 50),
            "p90_seconds": percentile(all_latencies, 90),
            "p95_seconds": percentile(all_latencies, 95),
            "p99_seconds": percentile(all_latencies, 99),
            "methodology": "CUDA synchronize; perf_counter around trajectory generation only; warmup excluded",
            "warmup_plans": args.warmup_plans,
        },
        "evaluation_runtime_seconds": runtime,
        "peak_vram_mib_torch": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "episode_records": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
