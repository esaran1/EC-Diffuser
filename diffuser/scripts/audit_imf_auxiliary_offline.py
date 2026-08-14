#!/usr/bin/env python3
"""Audit paired iMF checkpoints on fixed held-out OGBench conditions."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from diffuser.datasets.benchmark_sequence import OGBenchPuzzleWindowDataset
from diffuser.scripts.eval_ogbench_policy import backbone_class_for_method, sha256_file
from diffuser.scripts.train_phase7_pilot import build_method, set_seed


def raw_actions(normalized_actions, action_minimum, action_maximum):
    width = action_maximum - action_minimum
    safe_width = torch.where(width > 1e-6, width, torch.ones_like(width))
    values = (normalized_actions + 1.0) * 0.5 * safe_width + action_minimum
    return torch.where(width > 1e-6, values, action_minimum)


def summarize_dimensions(values):
    return {
        "mean": values.mean(axis=0).tolist(),
        "standard_deviation": values.std(axis=0, ddof=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=Path(
            "experiments/pilots/imf_auxiliary_offline_failure_diagnostic_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("offline diagnostic requires exactly one visible CUDA device")

    diagnostic_bytes = args.diagnostic.read_bytes()
    diagnostic = json.loads(diagnostic_bytes)
    task_diagnostic = json.loads(
        Path(diagnostic["parent_task_protocol"]).read_text()
    )
    training_protocol_path = Path(task_diagnostic["evaluation"]["protocol"])
    training_protocol = json.loads(training_protocol_path.read_text())
    task = training_protocol["task"]
    data_config = diagnostic["data"]
    normalizer_bundle = json.loads(Path(task["normalizer_bundle"]).read_text())
    normalizer_entry = normalizer_bundle["tasks"]["ogbench_puzzle_4x4_play_state"]
    dataset = OGBenchPuzzleWindowDataset(
        task["dataset_manifest"],
        split=data_config["split"],
        horizon=task["horizon"],
        goal_seed=data_config["goal_seed"],
        normalizer_state=normalizer_entry["normalizer"],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    batches = []
    for index, batch in enumerate(loader):
        batches.append(batch)
        if index + 1 == data_config["ordered_batches"]:
            break
    if len(batches) != data_config["ordered_batches"]:
        raise RuntimeError("validation split has too few diagnostic batches")

    device = torch.device("cuda:0")
    action_stats = dataset.normalizer.action_stats
    action_minimum = torch.as_tensor(
        action_stats["min"], dtype=torch.float32, device=device
    )
    action_maximum = torch.as_tensor(
        action_stats["max"], dtype=torch.float32, device=device
    )
    nfe = diagnostic["nfe"]
    results = []
    torch.cuda.reset_peak_memory_stats(device)

    for seed in diagnostic["training_seeds"]:
        for method_name in diagnostic["methods"]:
            frozen = task_diagnostic["checkpoints"][str(seed)][method_name]
            checkpoint_path = Path(frozen["path"])
            checkpoint_hash = sha256_file(checkpoint_path)
            if checkpoint_hash != frozen["sha256"]:
                raise RuntimeError("checkpoint SHA256 mismatch")

            backbone_config = training_protocol["backbone"]
            method_config = training_protocol["methods"][method_name]
            backbone = backbone_class_for_method(method_name)(
                horizon=task["horizon"],
                transition_dim=task["observation_dim"] + task["action_dim"],
                cond_dim=task["observation_dim"],
                dim=backbone_config["dim"],
                dim_mults=tuple(backbone_config["dim_mults"]),
                attention=backbone_config["attention"],
            ).to(device)
            policy = build_method(
                method_name, backbone, task, method_config
            ).to(device)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if int(checkpoint["step"]) != diagnostic["checkpoint_step"]:
                raise RuntimeError("checkpoint step mismatch")
            policy.load_state_dict(checkpoint["ema"], strict=True)
            del checkpoint
            policy.eval()

            calls = {"count": 0}
            hook = backbone.register_forward_hook(
                lambda _module, _inputs, _output: calls.__setitem__(
                    "count", calls["count"] + 1
                )
            )
            generated_actions = []
            target_actions = []
            observation_errors = []
            latencies = []
            set_seed(data_config["sample_seed"])
            for batch_index, batch in enumerate(batches):
                trajectories = batch.trajectories.to(device)
                conditions = {
                    int(key): value.to(device)
                    for key, value in batch.conditions.items()
                }
                with torch.random.fork_rng(devices=[device.index]):
                    set_seed(data_config["sample_seed"] + batch_index)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    with torch.no_grad():
                        sample = policy(conditions, n_steps=nfe, verbose=False)
                    torch.cuda.synchronize(device)
                    latencies.append(time.perf_counter() - start)
                generated = sample.trajectories
                if not torch.isfinite(generated).all():
                    raise FloatingPointError("non-finite generated trajectory")
                for timestep, value in conditions.items():
                    torch.testing.assert_close(
                        generated[:, timestep, task["action_dim"]:],
                        value,
                        rtol=0,
                        atol=0,
                    )
                generated_actions.append(generated[:, 0, :task["action_dim"]])
                target_actions.append(trajectories[:, 0, :task["action_dim"]])
                observation_errors.append(
                    generated[:, 1:-1, task["action_dim"]:]
                    - trajectories[:, 1:-1, task["action_dim"]:]
                )
            hook.remove()
            expected_calls = len(batches) * nfe
            if calls["count"] != expected_calls:
                raise RuntimeError("denoiser-call count mismatch")

            generated_normalized = torch.cat(generated_actions)
            target_normalized = torch.cat(target_actions)
            generated_raw = raw_actions(
                generated_normalized, action_minimum, action_maximum
            )
            target_raw = raw_actions(target_normalized, action_minimum, action_maximum)
            action_error = generated_raw - target_raw
            observation_error = torch.cat(observation_errors)
            generated_clip = generated_raw.abs() > 1.0
            target_clip = target_raw.abs() > 1.0
            generated_np = generated_raw.detach().cpu().numpy()
            target_np = target_raw.detach().cpu().numpy()
            result = {
                "training_seed": seed,
                "method": method_name,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_step": diagnostic["checkpoint_step"],
                "checkpoint_weights": "EMA",
                "wrapper": type(policy).__name__,
                "backbone": type(backbone).__name__,
                "validation_windows": int(generated_raw.shape[0]),
                "nfe": nfe,
                "observed_denoiser_calls": calls["count"],
                "expected_denoiser_calls": expected_calls,
                "finite": True,
                "endpoint_conditioning_exact": True,
                "generated_first_action_step_clip_fraction": float(
                    generated_clip.any(dim=-1).float().mean().cpu()
                ),
                "generated_first_action_element_clip_fraction": float(
                    generated_clip.float().mean().cpu()
                ),
                "target_first_action_step_clip_fraction": float(
                    target_clip.any(dim=-1).float().mean().cpu()
                ),
                "target_first_action_element_clip_fraction": float(
                    target_clip.float().mean().cpu()
                ),
                "first_action_raw_mae": float(action_error.abs().mean().cpu()),
                "first_action_raw_rmse": float(
                    torch.sqrt(torch.mean(action_error.square())).cpu()
                ),
                "interior_observation_normalized_mae": float(
                    observation_error.abs().mean().cpu()
                ),
                "interior_observation_normalized_rmse": float(
                    torch.sqrt(torch.mean(observation_error.square())).cpu()
                ),
                "generated_first_action": summarize_dimensions(generated_np),
                "target_first_action": summarize_dimensions(target_np),
                "planning_latency": {
                    "batches": len(latencies),
                    "mean_seconds_per_batch": statistics.fmean(latencies),
                    "total_seconds": sum(latencies),
                },
            }
            results.append(result)
            del policy, backbone, generated_raw, target_raw
            torch.cuda.empty_cache()

    payload = {
        "schema_version": "imf-auxiliary-offline-failure-diagnostic-results-v1",
        "status": "PASS",
        "diagnostic": str(args.diagnostic),
        "diagnostic_sha256": hashlib.sha256(diagnostic_bytes).hexdigest(),
        "training_protocol": str(training_protocol_path),
        "validation_windows_per_checkpoint": data_config["windows"],
        "results": results,
        "peak_vram_mib_torch": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
