#!/usr/bin/env python3
"""Build deterministic zero-copy conversion manifests for Phase 6 datasets."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


SCHEMA_VERSION = "fast-generative-policy-dataset-v1"
TRANSFORM_VERSION = "phase6-zero-copy-adapter-v1"


def split_ids(ids, seed, validation_fraction=0.1):
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = round(len(shuffled) * validation_fraction)
    return {
        "train": sorted(shuffled[:-validation_count]),
        "validation": sorted(shuffled[-validation_count:]),
    }


def membership_hash(splits):
    payload = json.dumps(splits, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


OGBENCH_STATE_TASKS = {
    "puzzle-4x4-play-v0": {
        "task_id": "ogbench-puzzle-4x4-play-v0-state",
        "entity": "button_states (16 binary entities) plus qpos/qvel replay state",
    },
    "cube-triple-play-v0": {
        "task_id": "ogbench-cube-triple-play-v0-state",
        "entity": "three cube poses plus arm qpos/qvel replay state",
    },
    "cube-double-play-v0": {
        "task_id": "ogbench-cube-double-play-v0-state",
        "entity": "two cube poses plus arm qpos/qvel replay state",
    },
}


def ogbench_manifest(root, dataset="puzzle-4x4-play-v0"):
    """Build an OGBench state manifest.

    Every OGBench state task uses the same episode-boundary derivation and the
    same goal policy, so one builder serves all of them. `dataset` selects the
    task; the resulting manifests differ only in paths, task_id, and the entity
    description.
    """
    if dataset not in OGBENCH_STATE_TASKS:
        raise ValueError("unknown OGBench state task: {}".format(dataset))
    spec = OGBENCH_STATE_TASKS[dataset]
    paths = {
        "train": root / "ogbench/{}/train.npz".format(dataset),
        "validation": root / "ogbench/{}/validation.npz".format(dataset),
    }
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                "missing OGBench {} {} split at {}".format(dataset, split, path)
            )
    splits = {}
    offsets = {}
    for split, path in paths.items():
        with np.load(str(path)) as data:
            terminals = data["terminals"]
        ends = (np.flatnonzero(terminals) + 1).tolist()
        starts = [0] + ends[:-1]
        splits[split] = ["{}:{:04d}".format(split, index) for index in range(len(ends))]
        offsets[split] = [
            {"episode_id": episode_id, "start": start, "end": end}
            for episode_id, start, end in zip(splits[split], starts, ends)
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "task_id": spec["task_id"],
        "source_paths": {key: str(path) for key, path in paths.items()},
        "split_policy": "official episode-level train and validation files",
        "splits": splits,
        "split_membership_sha256": membership_hash(splits),
        "episode_offsets": offsets,
        "field_mapping": {
            "observation": "observations",
            "action": "actions",
            "goal": "future observations from the same episode only",
            "episode_id": "manifest episode_id",
            "timestep": "row index minus episode start",
            "success": "not defined for play collection trajectories",
            "task_id": "constant task_id",
            "entity": spec["entity"],
        },
        "goal_policy": "sample goal time uniformly from [current_time, episode_end]; never cross an episode or split",
        "normalization": "fit observations/actions on official train episodes only",
        "output": "zero-copy indexed view; source NPZ files remain immutable",
    }


def mimicgen_manifest(root, seed):
    path = root / "mimicgen/large_interpolation/three_piece_assembly_d1.hdf5"
    accepted = []
    quarantined = []
    lengths = {}
    with h5py.File(str(path), "r") as data:
        for name, demo in data["data"].items():
            episode_id = int(name.split("_")[-1])
            lengths[str(episode_id)] = len(demo["actions"])
            if np.max(demo["rewards"][:]) > 0:
                accepted.append(episode_id)
            else:
                quarantined.append(episode_id)
    splits = split_ids(sorted(accepted), seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "task_id": "mimicgen-three-piece-assembly-d1-large-interpolation",
        "source_path": str(path),
        "split_seed": seed,
        "splits": splits,
        "split_membership_sha256": membership_hash(splits),
        "quarantined_episode_ids": sorted(quarantined),
        "quarantine_reason": "stored reward is zero at every timestep; replay success must be verified before use",
        "episode_lengths": lengths,
        "field_mapping": {
            "observation": "selected non-privileged data/demo_<id>/obs fields",
            "action": "data/demo_<id>/actions (7-D OSC pose plus gripper)",
            "goal": "constant ThreePieceAssembly_D1 task goal",
            "episode_id": "numeric demo suffix",
            "timestep": "row within demo",
            "success": "max(rewards) > 0; quarantined otherwise",
            "task_id": "constant task_id",
            "entity": "object observation and relative base/piece pose fields",
            "rgb": "agentview_image and robot0_eye_in_hand_image",
            "reward": "rewards",
            "termination": "dones",
            "simulator_state": "states (never policy-visible)",
        },
        "goal_policy": "fixed task-conditioned goal; do not synthesize a varying continuous goal",
        "normalization": "fit only on accepted training episodes",
        "output": "zero-copy HDF5 indexed view; source HDF5 remains immutable",
    }


def dexjoco_manifest(root, seed):
    path = root / "dexjoco/hammer_nail/data/chunk-000/file-000.parquet"
    table = pq.read_table(str(path), columns=["episode_index"])
    ids = sorted(set(np.asarray(table["episode_index"]).tolist()))
    splits = split_ids(ids, seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "task_id": "dexjoco-hammer-nail-rand-full",
        "source_path": str(path),
        "split_seed": seed,
        "splits": splits,
        "split_membership_sha256": membership_hash(splits),
        "field_mapping": {
            "observation": "observation.state (23-D non-privileged policy state)",
            "action": "action (22-D rotation-vector policy action)",
            "goal": "constant hammer-nail task/language goal",
            "episode_id": "episode_index",
            "timestep": "frame_index",
            "success": "true for released successful demonstration episodes",
            "task_id": "task_index plus constant task_id",
            "rgb": "external random_camera and wrist videos (not downloaded in the state-only pilot)",
        },
        "goal_policy": "fixed task/language-conditioned goal; do not expose privileged task state",
        "normalization": "recompute from training episodes; do not use full-dataset stats.json for final experiments",
        "output": "zero-copy Parquet indexed view; source table remains immutable",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/benchmarks/source"))
    parser.add_argument("--seed", type=int, default=2606)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--only",
        choices=sorted(OGBENCH_STATE_TASKS) + ["mimicgen", "dexjoco"],
        nargs="*",
        help="build only these sources; default builds every source whose data is present",
    )
    args = parser.parse_args()

    requested = set(args.only) if args.only else None

    def wanted(name):
        return requested is None or name in requested

    manifests = {}
    for dataset in OGBENCH_STATE_TASKS:
        if not wanted(dataset):
            continue
        try:
            name = "ogbench_" + dataset.replace("-", "_")
            manifests[name] = ogbench_manifest(args.root, dataset)
        except FileNotFoundError as error:
            if requested is not None:
                raise
            print("skipping {}: {}".format(dataset, error))
    if wanted("mimicgen"):
        manifests["mimicgen_three_piece_assembly_d1_large_interpolation"] = mimicgen_manifest(
            args.root, args.seed
        )
    if wanted("dexjoco"):
        manifests["dexjoco_hammer_nail_rand_full"] = dexjoco_manifest(args.root, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
        path = args.output_dir / (name + "_manifest.json")
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(path)


if __name__ == "__main__":
    main()
