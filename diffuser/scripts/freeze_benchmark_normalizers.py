#!/usr/bin/env python3
"""Freeze train-split-only normalization for the approved benchmark adapters."""

import argparse
import hashlib
import json
from pathlib import Path

from diffuser.datasets.benchmark_sequence import (
    DexJoCoHammerWindowDataset,
    MimicGenThreePieceWindowDataset,
    OGBenchCubeDoubleWindowDataset,
    OGBenchCubeTripleWindowDataset,
    OGBenchPuzzleWindowDataset,
)


TASKS = {
    "ogbench_puzzle_4x4_play_state": (
        OGBenchPuzzleWindowDataset,
        Path("experiments/datasets/converted/ogbench_puzzle_4x4_play_v0_manifest.json"),
        5,
    ),
    "ogbench_cube_triple_play_state": (
        OGBenchCubeTripleWindowDataset,
        Path("experiments/datasets/converted/ogbench_cube_triple_play_v0_manifest.json"),
        5,
    ),
    "ogbench_cube_double_play_state": (
        OGBenchCubeDoubleWindowDataset,
        Path("experiments/datasets/converted/ogbench_cube_double_play_v0_manifest.json"),
        5,
    ),
    "mimicgen_three_piece_assembly_d1_large_interpolation": (
        MimicGenThreePieceWindowDataset,
        Path("experiments/datasets/converted/mimicgen_three_piece_assembly_d1_large_interpolation_manifest.json"),
        10,
    ),
    "dexjoco_hammer_nail_rand_full": (
        DexJoCoHammerWindowDataset,
        Path("experiments/datasets/converted/dexjoco_hammer_nail_rand_full_manifest.json"),
        30,
    ),
}


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def build():
    tasks = {}
    for task_name, (dataset_class, manifest_path, horizon) in TASKS.items():
        dataset = dataset_class(manifest_path, split="train", horizon=horizon)
        state = dataset.normalizer_state_dict()
        canonical_state = json.dumps(
            state, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest_bytes = manifest_path.read_bytes()
        tasks[task_name] = {
            "task_id": dataset.task_id,
            "adapter": "{}.{}".format(dataset_class.__module__, dataset_class.__name__),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_bytes(manifest_bytes),
            "split_membership_sha256": dataset.manifest["split_membership_sha256"],
            "observation_dim": dataset.observation_dim,
            "action_dim": dataset.action_dim,
            "normalizer": state,
            "normalizer_sha256": _sha256_bytes(canonical_state),
            "normalization_policy": "Gaussian observations and safe train-range actions; training episodes only",
        }
        close = getattr(dataset, "close", None)
        if close is not None:
            close()
    return {
        "schema_version": "phase7-benchmark-normalizers-v1",
        "status": "FROZEN",
        "tasks": tasks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/datasets/benchmark_normalizers_v1.json"),
    )
    args = parser.parse_args()
    payload = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(args.output)


if __name__ == "__main__":
    main()
