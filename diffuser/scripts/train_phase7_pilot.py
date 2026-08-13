#!/usr/bin/env python3
"""Run one predeclared, bounded Phase 7 training-throughput pilot."""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from diffuser.datasets.benchmark_sequence import OGBenchPuzzleWindowDataset
from diffuser.models import (
    ConditionalFlowMatching,
    GaussianDiffusion,
    ImprovedMeanFlow,
    IntervalTemporalUnet,
    ShortcutModel,
)
from diffuser.utils.arrays import set_global_device
from diffuser.utils.training import Trainer


METHODS = {
    "gaussian_diffusion": GaussianDiffusion,
    "conditional_flow_matching": ConditionalFlowMatching,
    "improved_meanflow": ImprovedMeanFlow,
    "shortcut_model": ShortcutModel,
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_method(name, model, task, method_config):
    common = {
        "model": model,
        "horizon": task["horizon"],
        "observation_dim": task["observation_dim"],
        "action_dim": task["action_dim"],
        "loss_type": "l1",
    }
    if name == "gaussian_diffusion":
        return GaussianDiffusion(
            **common,
            n_timesteps=method_config["training_timesteps"],
        )
    common.update(
        n_solver_steps=method_config["default_solver_steps"],
        time_scale=method_config["time_scale"],
    )
    if name == "shortcut_model":
        common["max_base_steps"] = method_config["max_base_steps"]
    return METHODS[name](**common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/pilots/ogbench_puzzle_state_pilot_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    training = protocol["training"]
    task = protocol["task"]
    steps = training["optimizer_steps"] if args.steps is None else args.steps
    if steps < 1 or steps > training["optimizer_steps"]:
        raise ValueError("steps must be in [1, {}]".format(training["optimizer_steps"]))
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    args.output_dir.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase 7 pilot")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("pilot expects exactly one visible CUDA device")
    device = torch.device("cuda:0")
    set_global_device(str(device))
    set_seed(training["seed"])

    normalizer_bundle = json.loads(Path(task["normalizer_bundle"]).read_text())
    normalizer_entry = normalizer_bundle["tasks"]["ogbench_puzzle_4x4_play_state"]
    if normalizer_entry["normalizer_sha256"] != task["normalizer_sha256"]:
        raise RuntimeError("normalizer hash does not match the predeclared protocol")
    dataset = OGBenchPuzzleWindowDataset(
        task["dataset_manifest"],
        split="train",
        horizon=task["horizon"],
        goal_seed=training["seed"],
        normalizer_state=normalizer_entry["normalizer"],
    )
    if (dataset.observation_dim, dataset.action_dim) != (
        task["observation_dim"], task["action_dim"]
    ):
        raise RuntimeError("dataset dimensions do not match the protocol")

    backbone = protocol["backbone"]
    model = IntervalTemporalUnet(
        horizon=task["horizon"],
        transition_dim=task["observation_dim"] + task["action_dim"],
        cond_dim=task["observation_dim"],
        dim=backbone["dim"],
        dim_mults=tuple(backbone["dim_mults"]),
        attention=backbone["attention"],
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != backbone["parameter_count"]:
        raise RuntimeError("backbone parameter count does not match the protocol")
    method = build_method(
        args.method, model, task, protocol["methods"][args.method]
    ).to(device)

    wandb.init(
        project="fast-generative-policy-pilots",
        mode="disabled",
        config={"method": args.method, "steps": steps, "seed": training["seed"]},
    )
    trainer = Trainer(
        method,
        dataset,
        renderer=None,
        ema_decay=training["ema_decay"],
        train_batch_size=training["microbatch_size"],
        train_lr=training["learning_rate"],
        gradient_accumulate_every=training["gradient_accumulation"],
        step_start_ema=training["ema_start_step"],
        update_ema_every=training["ema_update_every"],
        log_freq=training["log_frequency"],
        sample_freq=0,
        save_freq=0,
        results_folder=str(args.output_dir),
        n_reference=0,
    )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    history = trainer.train(steps)
    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - start
    peak_vram_bytes = torch.cuda.max_memory_allocated(device)

    checkpoint_path = args.output_dir / "state_{}.pt".format(trainer.step)
    trainer.save(trainer.step)
    checkpoint_hash = sha256_file(checkpoint_path)
    losses = [record["loss"] for record in history]
    if not losses or not np.isfinite(losses).all():
        raise FloatingPointError("logged losses are missing or non-finite")

    summary = {
        "schema_version": "phase7-training-pilot-result-v1",
        "status": "PASS",
        "method": args.method,
        "wrapper": type(method).__name__,
        "backbone": type(model).__name__,
        "parameter_count": parameter_count,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol": str(args.protocol),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "dataset_manifest": task["dataset_manifest"],
        "normalizer_sha256": normalizer_entry["normalizer_sha256"],
        "dataset_windows": len(dataset),
        "optimizer_steps": trainer.step,
        "microbatch_size": training["microbatch_size"],
        "gradient_accumulation": training["gradient_accumulation"],
        "effective_batch_size": training["effective_batch_size"],
        "runtime_seconds": runtime_seconds,
        "seconds_per_optimizer_step": runtime_seconds / trainer.step,
        "peak_vram_bytes_torch": peak_vram_bytes,
        "peak_vram_mib_torch": peak_vram_bytes / (1024 ** 2),
        "initial_logged_loss": losses[0],
        "final_logged_loss": losses[-1],
        "minimum_logged_loss": min(losses),
        "logged_metrics": history,
        "all_logged_losses_finite": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "ema_present": True,
        "seed": training["seed"],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("PHASE 7 PILOT SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))
    wandb.finish()


if __name__ == "__main__":
    main()
