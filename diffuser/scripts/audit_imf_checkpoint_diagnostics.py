#!/usr/bin/env python3
"""Recover fixed iMF diagnostics from a repository-native checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diffuser.datasets.benchmark_sequence import OGBenchPuzzleWindowDataset
from diffuser.models import IntervalTemporalUnet
from diffuser.scripts.train_phase7_pilot import (
    build_method,
    fixed_validation,
    set_seed,
)
from diffuser.utils.arrays import set_global_device
from diffuser.utils.training import Trainer


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_finite_metrics(label, metrics):
    values = [value for value in metrics.values() if isinstance(value, (int, float))]
    if not values or not np.isfinite(values).all():
        raise FloatingPointError("non-finite {} metrics".format(label))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output))

    protocol_bytes = args.protocol.read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    summary_path = args.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["protocol_sha256"] != protocol_sha256:
        raise RuntimeError("training summary protocol hash mismatch")
    checkpoint = Path(summary["checkpoint"])
    if checkpoint.parent.resolve() != args.run_dir.resolve():
        raise RuntimeError("checkpoint is outside the declared run directory")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != summary["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash mismatch")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("audit requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    set_global_device(str(device))

    task = protocol["task"]
    training = protocol["training"]
    set_seed(training["seed"])
    normalizers = json.loads(Path(task["normalizer_bundle"]).read_text())
    normalizer = normalizers["tasks"]["ogbench_puzzle_4x4_play_state"]
    if normalizer["normalizer_sha256"] != task["normalizer_sha256"]:
        raise RuntimeError("normalizer hash mismatch")
    dataset = OGBenchPuzzleWindowDataset(
        task["dataset_manifest"],
        split="train",
        horizon=task["horizon"],
        goal_seed=training["seed"],
        normalizer_state=normalizer["normalizer"],
    )
    validation_dataset = OGBenchPuzzleWindowDataset(
        task["dataset_manifest"],
        split="validation",
        horizon=task["horizon"],
        goal_seed=training["seed"],
        normalizer_state=normalizer["normalizer"],
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
        if batch_index + 1 == training["fixed_validation_batches"]:
            break
    if len(validation_batches) != training["fixed_validation_batches"]:
        raise RuntimeError("validation split has too few complete batches")

    backbone = protocol["backbone"]
    denoiser = IntervalTemporalUnet(
        horizon=task["horizon"],
        transition_dim=task["observation_dim"] + task["action_dim"],
        cond_dim=task["observation_dim"],
        dim=backbone["dim"],
        dim_mults=tuple(backbone["dim_mults"]),
        attention=backbone["attention"],
    ).to(device)
    if sum(parameter.numel() for parameter in denoiser.parameters()) != backbone[
        "parameter_count"
    ]:
        raise RuntimeError("backbone parameter count mismatch")
    method_config = dict(protocol["methods"]["improved_meanflow"])
    method_config["collect_diagnostics"] = True
    method = build_method(
        "improved_meanflow", denoiser, task, method_config
    ).to(device)
    trainer = Trainer(
        method,
        dataset,
        renderer=None,
        train_batch_size=training["microbatch_size"],
        sample_freq=0,
        save_freq=0,
        results_folder=str(args.run_dir),
        n_reference=0,
    )
    validation_seed = training["seed"] + 100000
    initial = fixed_validation(
        trainer.model, validation_batches, validation_seed, device
    )
    aggregate_initial = summary["fixed_validation_history"][0][
        "unweighted_meanflow_loss"
    ]
    if not np.isclose(
        initial["unweighted_meanflow_loss"], aggregate_initial, rtol=0, atol=1e-6
    ):
        raise RuntimeError("reconstructed initial validation does not match summary")

    expected_step = summary["optimizer_steps"]
    trainer.load(expected_step)
    if trainer.step != expected_step:
        raise RuntimeError("Trainer.load did not restore the expected step")
    live = fixed_validation(
        trainer.model, validation_batches, validation_seed, device
    )
    ema = fixed_validation(
        trainer.ema_model, validation_batches, validation_seed, device
    )
    for label, metrics in (("initial", initial), ("live", live), ("ema", ema)):
        assert_finite_metrics(label, metrics)

    payload = {
        "schema_version": "imf-checkpoint-diagnostic-audit-v1",
        "status": "PASS",
        "protocol": str(args.protocol),
        "protocol_sha256": protocol_sha256,
        "training_summary": str(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "trainer_step": trainer.step,
        "diagnostics_enabled_posthoc_only": True,
        "fixed_validation_seed": validation_seed,
        "fixed_validation_batches": len(validation_batches),
        "initial": initial,
        "live": live,
        "ema": ema,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
