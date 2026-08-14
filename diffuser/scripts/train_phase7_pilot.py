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
    AuxiliaryImprovedMeanFlow,
    AuxiliaryIntervalTemporalUnet,
    ConditionalFlowMatching,
    GaussianDiffusion,
    ImprovedMeanFlow,
    IntervalTemporalUnet,
    ShortcutModel,
)


METHODS = {
    "auxiliary_improved_meanflow": AuxiliaryImprovedMeanFlow,
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


def resolve_training(protocol, replication_seed=None):
    """Resolve only replication overrides explicitly frozen in the protocol."""
    training = dict(protocol["training"])
    replications = protocol.get("replications")
    if replications is None:
        if replication_seed is not None:
            raise ValueError("protocol does not declare replication seeds")
        return training
    if replication_seed is None:
        raise ValueError("this protocol requires --replication-seed")
    key = str(replication_seed)
    if key not in replications:
        raise ValueError("replication seed {} is not predeclared".format(key))
    overrides = replications[key]["training_overrides"]
    training.update(overrides)
    if training["seed"] != replication_seed:
        raise ValueError("replication override must set the selected training seed")
    return training


def build_method(name, model, task, method_config):
    common = {
        "model": model,
        "horizon": task["horizon"],
        "observation_dim": task["observation_dim"],
        "action_dim": task["action_dim"],
        "loss_type": method_config.get("loss_type", "l1"),
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
    if name in ("improved_meanflow", "auxiliary_improved_meanflow"):
        for key in (
            "time_mean", "time_std", "boundary_probability",
            "adaptive_weighting", "adaptive_power", "adaptive_epsilon",
            "collect_diagnostics",
        ):
            if key in method_config:
                common[key] = method_config[key]
    if name == "shortcut_model":
        common["max_base_steps"] = method_config["max_base_steps"]
    return METHODS[name](**common)


def fixed_validation(model, batches, seed, device):
    """Evaluate fixed batches with fixed stochastic draws without changing RNG state."""
    from diffuser.utils.arrays import batch_to_device
    was_training = model.training
    model.eval()
    records = []
    try:
        for index, cpu_batch in enumerate(batches):
            batch = batch_to_device(cpu_batch)
            fork_devices = [device.index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(seed + index)
                with torch.enable_grad():
                    loss, info = model.loss(*batch)
            record = {"loss": float(loss.detach().cpu())}
            record.update({
                key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                for key, value in info.items()
            })
            if not np.isfinite(list(record.values())).all():
                raise FloatingPointError("non-finite fixed validation metric")
            records.append(record)
    finally:
        model.train(was_training)
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def main():
    from diffuser.utils.arrays import set_global_device
    from diffuser.utils.training import Trainer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/pilots/ogbench_puzzle_state_pilot_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--validation-frequency", type=int)
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--variant")
    parser.add_argument("--replication-seed", type=int)
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    training = resolve_training(protocol, args.replication_seed)
    task = protocol["task"]
    method_config = dict(protocol["methods"][args.method])
    variant = None
    if args.variant is not None:
        if args.method != "improved_meanflow":
            raise ValueError("stability variants are restricted to improved_meanflow")
        variants = protocol.get("variants", {})
        if args.variant not in variants:
            raise ValueError("unknown protocol variant {!r}".format(args.variant))
        variant = variants[args.variant]
        training.update(variant.get("training_overrides", {}))
        method_config.update(variant.get("method_overrides", {}))
    elif protocol.get("variants"):
        raise ValueError("this protocol requires an explicit --variant")
    steps = training["optimizer_steps"] if args.steps is None else args.steps
    if args.validation_frequency is None:
        args.validation_frequency = training.get("fixed_validation_frequency", 0)
    if args.validation_batches is None:
        args.validation_batches = training.get("fixed_validation_batches", 4)
    if steps < 1 or steps > training["optimizer_steps"]:
        raise ValueError("steps must be in [1, {}]".format(training["optimizer_steps"]))
    if args.validation_frequency < 0 or args.validation_batches < 1:
        raise ValueError("validation settings must be non-negative/positive")
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
    validation_goal_seed = training.get(
        "fixed_validation_goal_seed", training["seed"]
    )
    validation_seed = training.get(
        "fixed_validation_seed", training["seed"] + 100000
    )
    validation_dataset = OGBenchPuzzleWindowDataset(
        task["dataset_manifest"],
        split="validation",
        horizon=task["horizon"],
        goal_seed=validation_goal_seed,
        normalizer_state=normalizer_entry["normalizer"],
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=training["microbatch_size"],
        num_workers=0,
        shuffle=False,
        pin_memory=True,
    )
    validation_batches = []
    for batch_index, batch in enumerate(validation_loader):
        validation_batches.append(batch)
        if batch_index + 1 == args.validation_batches:
            break
    if len(validation_batches) != args.validation_batches:
        raise RuntimeError("validation split has too few complete batches")

    if (dataset.observation_dim, dataset.action_dim) != (
        task["observation_dim"], task["action_dim"]
    ):
        raise RuntimeError("dataset dimensions do not match the protocol")

    backbone = protocol["backbone"]
    model_class = (
        AuxiliaryIntervalTemporalUnet
        if args.method == "auxiliary_improved_meanflow"
        else IntervalTemporalUnet
    )
    model = model_class(
        horizon=task["horizon"],
        transition_dim=task["observation_dim"] + task["action_dim"],
        cond_dim=task["observation_dim"],
        dim=backbone["dim"],
        dim_mults=tuple(backbone["dim_mults"]),
        attention=backbone["attention"],
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameter_count = method_config.get(
        "parameter_count", backbone["parameter_count"]
    )
    if parameter_count != expected_parameter_count:
        raise RuntimeError("backbone parameter count does not match the protocol")
    method = build_method(
        args.method, model, task, method_config
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
        max_grad_norm=training.get("max_grad_norm"),
        collect_step_diagnostics=training.get(
            "collect_step_diagnostics", False
        ),
        adam_betas=tuple(training.get("adam_betas", (0.9, 0.999))),
        lr_warmup_steps=training.get("lr_warmup_steps", 0),
        dataloader_seed=training.get("dataloader_seed"),
    )

    if "optimization_seed" in training:
        set_seed(training["optimization_seed"])

    validation_history = [{
        "step": 0,
        **fixed_validation(
            trainer.model, validation_batches, validation_seed, device
        ),
    }]
    ema_validation_history = [{
        "step": 0,
        **fixed_validation(
            trainer.ema_model,
            validation_batches,
            validation_seed,
            device,
        ),
    }]
    torch.cuda.reset_peak_memory_stats(device)
    runtime_seconds = 0.0
    trained = 0
    history = []
    while trained < steps:
        chunk = steps - trained
        if args.validation_frequency:
            chunk = min(chunk, args.validation_frequency)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        history = trainer.train(chunk)
        torch.cuda.synchronize(device)
        runtime_seconds += time.perf_counter() - start
        trained += chunk
        if args.validation_frequency and (
            trained % args.validation_frequency == 0 or trained == steps
        ):
            validation_history.append({
                "step": trainer.step,
                **fixed_validation(
                    trainer.model,
                    validation_batches,
                    validation_seed,
                    device,
                ),
            })
            ema_validation_history.append({
                "step": trainer.step,
                **fixed_validation(
                    trainer.ema_model,
                    validation_batches,
                    validation_seed,
                    device,
                ),
            })
    peak_vram_bytes = torch.cuda.max_memory_allocated(device)

    checkpoint_path = args.output_dir / "state_{}.pt".format(trainer.step)
    trainer.save(trainer.step)
    checkpoint_hash = sha256_file(checkpoint_path)
    losses = [record["loss"] for record in history]
    if not losses or not np.isfinite(losses).all():
        raise FloatingPointError("logged losses are missing or non-finite")

    summary = {
        "schema_version": "phase7-training-pilot-result-v3",
        "status": "PASS",
        "method": args.method,
        "variant": args.variant,
        "variant_description": None if variant is None else variant["description"],
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
        "validation_windows": len(validation_dataset),
        "fixed_validation_batches": args.validation_batches,
        "fixed_validation_frequency": args.validation_frequency,
        "fixed_validation_seed": validation_seed,
        "fixed_validation_goal_seed": validation_goal_seed,
        "fixed_validation_history": validation_history,
        "fixed_validation_history_weights": "live",
        "fixed_validation_history_ema": ema_validation_history,
        "optimizer_steps": trainer.step,
        "microbatch_size": training["microbatch_size"],
        "gradient_accumulation": training["gradient_accumulation"],
        "effective_batch_size": training["effective_batch_size"],
        "learning_rate": training["learning_rate"],
        "adam_betas": list(training.get("adam_betas", (0.9, 0.999))),
        "lr_warmup_steps": training.get("lr_warmup_steps", 0),
        "dataloader_seed": training.get("dataloader_seed"),
        "optimization_seed": training.get("optimization_seed"),
        "max_grad_norm": training.get("max_grad_norm"),
        "collect_step_diagnostics": training.get(
            "collect_step_diagnostics", False
        ),
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
