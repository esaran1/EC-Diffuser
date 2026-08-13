#!/usr/bin/env python3
"""Reload every OGBench pilot checkpoint and audit conditioned sampling."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diffuser.datasets.benchmark_sequence import OGBenchPuzzleWindowDataset
from diffuser.models import IntervalTemporalUnet
from diffuser.scripts.train_phase7_pilot import build_method, set_seed
from diffuser.utils.arrays import set_global_device
from diffuser.utils.training import Trainer


EXPECTED_CALLS = {
    "gaussian_diffusion": 100,
    "conditional_flow_matching": 4,
    "improved_meanflow": 4,
    "shortcut_model": 4,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/pilots/ogbench_puzzle_state_pilot_v1.json"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("data/phase7_runs/ogbench_puzzle_state"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output))

    protocol = json.loads(args.protocol.read_text())
    task = protocol["task"]
    training = protocol["training"]
    normalizers = json.loads(Path(task["normalizer_bundle"]).read_text())
    normalizer = normalizers["tasks"]["ogbench_puzzle_4x4_play_state"]
    dataset = OGBenchPuzzleWindowDataset(
        task["dataset_manifest"],
        split="train",
        horizon=task["horizon"],
        goal_seed=training["seed"],
        normalizer_state=normalizer["normalizer"],
    )
    source_batch = dataset[0]
    device = torch.device("cuda:0")
    set_global_device(str(device))
    set_seed(training["seed"])
    conditions = {
        key: torch.from_numpy(value).unsqueeze(0).to(device)
        for key, value in source_batch.conditions.items()
    }

    results = {}
    for method_name, expected_calls in EXPECTED_CALLS.items():
        run_dir = args.runs_root / "{}_seed42_1000".format(method_name)
        summary = json.loads((run_dir / "summary.json").read_text())
        checkpoint = Path(summary["checkpoint"])
        if sha256_file(checkpoint) != summary["checkpoint_sha256"]:
            raise RuntimeError("checkpoint hash mismatch for {}".format(method_name))

        backbone = protocol["backbone"]
        denoiser = IntervalTemporalUnet(
            horizon=task["horizon"],
            transition_dim=task["observation_dim"] + task["action_dim"],
            cond_dim=task["observation_dim"],
            dim=backbone["dim"],
            dim_mults=tuple(backbone["dim_mults"]),
            attention=backbone["attention"],
        ).to(device)
        method = build_method(
            method_name, denoiser, task, protocol["methods"][method_name]
        ).to(device)
        trainer = Trainer(
            method,
            dataset,
            renderer=None,
            train_batch_size=1,
            sample_freq=0,
            save_freq=0,
            results_folder=str(run_dir),
            n_reference=0,
        )
        trainer.load(1000)
        if trainer.step != 1000:
            raise RuntimeError("Trainer did not restore step 1000")

        call_count = [0]
        original_forward = trainer.ema_model.model.forward

        def counted_forward(*call_args, **call_kwargs):
            call_count[0] += 1
            return original_forward(*call_args, **call_kwargs)

        trainer.ema_model.model.forward = counted_forward
        trainer.ema_model.eval()
        with torch.no_grad():
            sample_kwargs = {"verbose": False}
            if method_name != "gaussian_diffusion":
                sample_kwargs["n_steps"] = expected_calls
            sample = trainer.ema_model(conditions, **sample_kwargs)

        trajectories = sample.trajectories
        if tuple(trajectories.shape) != (1, task["horizon"], 88):
            raise RuntimeError("invalid sampled trajectory shape")
        if not torch.isfinite(trajectories).all():
            raise FloatingPointError("non-finite conditioned sample")
        for timestep, value in conditions.items():
            torch.testing.assert_close(
                trajectories[:, timestep, task["action_dim"]:],
                value,
                rtol=0,
                atol=0,
            )
        if call_count[0] != expected_calls:
            raise RuntimeError(
                "{} used {} calls, expected {}".format(
                    method_name, call_count[0], expected_calls
                )
            )
        results[method_name] = {
            "status": "PASS",
            "trainer_step": trainer.step,
            "restored_wrapper": type(trainer.ema_model).__name__,
            "restored_denoiser": type(trainer.ema_model.model).__name__,
            "ema_weights_loaded": True,
            "sample_shape": list(trajectories.shape),
            "sample_finite": True,
            "initial_condition_exact": True,
            "goal_condition_exact": True,
            "observed_model_calls": call_count[0],
            "expected_model_calls": expected_calls,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": summary["checkpoint_sha256"],
        }
        del trainer, method, denoiser, sample, trajectories
        torch.cuda.empty_cache()

    payload = {
        "schema_version": "phase7-pilot-checkpoint-audit-v1",
        "status": "PASS",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
