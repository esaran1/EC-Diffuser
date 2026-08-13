#!/usr/bin/env python3
"""Build the proposed Phase 7 matrix without launching experiments."""

import argparse
import json
from pathlib import Path


METHODS = {
    "gaussian_diffusion": [100],
    "conditional_flow_matching": [1, 2, 4, 8, 16],
    "improved_meanflow": [1, 2, 4, 8],
    "shortcut_model": [1, 2, 4, 8],
}
EVALUATION_SEEDS = [101, 202, 303, 404, 505]
TASKS = {
    "pushcube_3c_randcolor_legacy": {
        "tier": "A",
        "training_seeds": [42, 43, 44],
        "dataset": "EC-Diffuser PushCube-3 random-color DLP",
        "policy_representation": "entity-structured DLP trajectory; AdaLNPINTDenoiser",
        "evaluation_protocol": {
            "episodes": 96,
            "task_count": 1,
            "episodes_per_task": 96,
            "episode_horizon": 100,
            "source": "matched EC-Diffuser Gaussian control protocol",
        },
        "training_hours": {
            "gaussian_diffusion": [35.0, 45.0],
            "conditional_flow_matching": [39.04, 39.04],
            "improved_meanflow": [82.86, 82.86],
            "shortcut_model": [39.12, 39.12],
        },
        "estimate_basis": "Flow is the measured 500k wall time. MeanFlow and Shortcut extrapolate measured 1000-step pilots. Gaussian requires a matched pilot.",
    },
    "ogbench_puzzle_4x4_play_state": {
        "tier": "A",
        "training_seeds": [42, 43, 44],
        "dataset": "OGBench puzzle-4x4-play-v0 state",
        "policy_representation": "flat state/action sequence; task-general backbone required",
        "evaluation_protocol": {
            "episodes": 250,
            "task_count": 5,
            "episodes_per_task": 50,
            "episode_horizon": 1000,
            "source": "official OGBench hyperparameters and task-wise evaluation loop",
        },
        "training_hours": {method: [30.0, 90.0] for method in METHODS},
        "estimate_basis": "Planning range only; a 1000-step task-native benchmark is mandatory before any full run.",
    },
    "mimicgen_three_piece_assembly_d1_large_interpolation": {
        "tier": "B",
        "training_seeds": [42],
        "dataset": "MimicGen large_interpolation/three_piece_assembly_d1",
        "policy_representation": "low-dimensional state/action sequence for the bounded pilot; visual policy is a separate ablation",
        "evaluation_protocol": {
            "episodes": 50,
            "task_count": 1,
            "episodes_per_task": 50,
            "episode_horizon": 700,
            "source": "official MimicGen large-interpolation task horizon; 50-rollout robomimic convention",
        },
        "training_hours": {method: [30.0, 90.0] for method in METHODS},
        "estimate_basis": "Single-seed screening only; a 1000-step task-native benchmark is mandatory.",
    },
    "dexjoco_hammer_nail_rand_full": {
        "tier": "B",
        "training_seeds": [42],
        "dataset": "DexJoCo hammer_nail rand_full LeRobot",
        "policy_representation": "23-D non-privileged state and 22-D action sequence; task-general backbone required",
        "evaluation_protocol": {
            "episodes": 50,
            "task_count": 1,
            "episodes_per_task": 50,
            "episode_horizon": 1000,
            "action_chunk_horizon": 30,
            "replan_ratio": 0.8,
            "source": "official DexJoCo OpenPI evaluator and hammer-nail environment",
        },
        "training_hours": {method: [30.0, 90.0] for method in METHODS},
        "estimate_basis": "Single-seed screening only; a 1000-step task-native benchmark is mandatory.",
    },
}


def build():
    training_runs = []
    evaluation_runs = []
    for task, task_spec in TASKS.items():
        for method, nfe_values in METHODS.items():
            for training_seed in task_spec["training_seeds"]:
                run_id = "{}__{}__train{}".format(task, method, training_seed)
                training_runs.append(
                    {
                        "run_id": run_id,
                        "tier": task_spec["tier"],
                        "task": task,
                        "dataset": task_spec["dataset"],
                        "policy_representation": task_spec["policy_representation"],
                        "method": method,
                        "training_seed": training_seed,
                        "checkpoint_selection": "final fixed-budget EMA checkpoint; no test-set selection",
                        "estimated_training_hours_range": task_spec["training_hours"][method],
                        "estimate_basis": task_spec["estimate_basis"],
                    }
                )
                for nfe in nfe_values:
                    for evaluation_seed in EVALUATION_SEEDS:
                        evaluation_runs.append(
                            {
                                "training_run_id": run_id,
                                "tier": task_spec["tier"],
                                "task": task,
                                "method": method,
                                "training_seed": training_seed,
                                "checkpoint": "final_ema",
                                "nfe": nfe,
                                "evaluation_seed": evaluation_seed,
                                "episodes": task_spec["evaluation_protocol"]["episodes"],
                            }
                        )

    training_low = sum(row["estimated_training_hours_range"][0] for row in training_runs)
    training_high = sum(row["estimated_training_hours_range"][1] for row in training_runs)
    checkpoint_bytes = 502228922
    return {
        "schema_version": "phase7-experiment-matrix-v2",
        "status": "APPROVED_FOR_BOUNDED_PILOTS_NOT_FROZEN_FOR_FULL_RUNS",
        "compute_gate": {
            "one_gpu": "NVIDIA GeForce RTX 4080 16 GB",
            "parallel_gpu_jobs": 1,
            "full_runs_started": 0,
            "required_next_step": "Implement and validate dataset/evaluation adapters, then run sequential <=1000-step task-native timing pilots. Re-estimate before any run over two GPU-hours.",
        },
        "protocol": {
            "methods": list(METHODS),
            "nfe": METHODS,
            "evaluation_seeds": EVALUATION_SEEDS,
            "task_native_evaluation_protocols": {
                task: spec["evaluation_protocol"] for task, spec in TASKS.items()
            },
            "paired_evaluation_seeds": True,
            "shared": [
                "task dataset and episode split",
                "training examples seen",
                "backbone capacity unless a method intrinsically requires an added embedding",
                "policy horizon and conditioning",
                "normalization and action representation",
                "optimizer family, batch size, and evaluation environments",
                "planner-only CUDA timing instrumentation",
            ],
            "metrics": [
                "benchmark success",
                "task-specific progress",
                "reward where defined",
                "mean planning latency",
                "P50 planning latency",
                "P95 planning latency",
                "NFE and verified model-call count",
                "parameter count",
                "peak inference VRAM",
                "training GPU-hours",
            ],
            "statistics": "Independent training seeds are the inferential unit. Use mean, sample SD, SE, and 95% CI; use paired evaluation-seed differences where justified.",
        },
        "counts": {
            "total_training_runs": len(training_runs),
            "tier_a_training_runs": sum(row["tier"] == "A" for row in training_runs),
            "tier_b_training_runs": sum(row["tier"] == "B" for row in training_runs),
            "total_evaluations": len(evaluation_runs),
            "total_evaluation_episodes": sum(row["episodes"] for row in evaluation_runs),
        },
        "cost": {
            "estimated_training_gpu_hours_range": [round(training_low, 2), round(training_high, 2)],
            "estimated_one_gpu_training_days_range": [round(training_low / 24.0, 2), round(training_high / 24.0, 2)],
            "checkpoint_bytes_each_observed_upper": checkpoint_bytes,
            "one_final_checkpoint_total_bytes": checkpoint_bytes * len(training_runs),
            "rolling_resume_plus_final_total_bytes": 2 * checkpoint_bytes * len(training_runs),
            "evaluation_gpu_hours": "Unknown cross-benchmark; measure one 16-episode run per task/method. PushCube reference is about 20.2 ms/NFE for the current backbone and 1.989 s/plan at 100-NFE Gaussian.",
        },
        "execution_order": [
            "Validate and transform each dataset; freeze hashes, splits, normalization, and goals.",
            "Run <=1000-step timing/stability pilots sequentially for each new task and method.",
            "Report measured GPU-hours and storage; obtain approval before full single-seed runs.",
            "Run Tier A single-seed pilots first; evaluate all predeclared NFE values on paired seeds.",
            "Screen Strong, Interesting Failure, Redundant, or Broken; revise compute estimate.",
            "Only after approval, run seeds 43 and 44 for Strong and strategically important Tier A cases.",
            "Run Tier B single-seed experiments only after Tier A evidence is secure.",
        ],
        "training_runs": training_runs,
        "evaluation_runs": evaluation_runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/phase7_experiment_matrix.json"))
    args = parser.parse_args()
    payload = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
