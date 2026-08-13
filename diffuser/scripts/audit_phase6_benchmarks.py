#!/usr/bin/env python3
"""Audit the bounded OGBench, MimicGen, and DexJoCo Phase 6 subsets."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "median": float(np.quantile(values, 0.5)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def episode_slices(terminals):
    terminal_indices = np.flatnonzero(terminals)
    starts = np.r_[0, terminal_indices[:-1] + 1]
    ends = terminal_indices + 1
    if len(ends) == 0 or ends[-1] != len(terminals):
        raise ValueError("terminal array does not close every episode")
    return list(zip(starts.tolist(), ends.tolist()))


def episode_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def duplicate_pairs(hashes):
    first = {}
    pairs = []
    for index, value in enumerate(hashes):
        if value in first:
            pairs.append([first[value], index])
        else:
            first[value] = index
    return pairs


def audit_ogbench(train_path, validation_path):
    reports = {}
    split_hashes = {}
    for split, path in (("train", train_path), ("validation", validation_path)):
        with np.load(str(path)) as dataset:
            arrays = {key: dataset[key] for key in dataset.files}
        slices = episode_slices(arrays["terminals"])
        hashes = [
            episode_hash(arrays["observations"][start:end], arrays["actions"][start:end])
            for start, end in slices
        ]
        split_hashes[split] = set(hashes)
        lengths = [end - start for start, end in slices]
        action_norm = np.linalg.norm(arrays["actions"], axis=-1)
        button_codes = np.sum(
            arrays["button_states"].astype(np.uint32)
            * (1 << np.arange(arrays["button_states"].shape[1], dtype=np.uint32)),
            axis=1,
        )
        reports[split] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "fields": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
            },
            "episodes": len(slices),
            "transitions": int(len(arrays["actions"])),
            "episode_length": stats(lengths),
            "all_numeric_values_finite": all(
                np.isfinite(value).all()
                for value in arrays.values()
                if np.issubdtype(value.dtype, np.number)
            ),
            "terminal_count": int(np.count_nonzero(arrays["terminals"])),
            "action_limit_violations": int(np.count_nonzero(np.abs(arrays["actions"]) > 1.0 + 1e-6)),
            "action_component_min": arrays["actions"].min(axis=0).tolist(),
            "action_component_max": arrays["actions"].max(axis=0).tolist(),
            "action_norm": stats(action_norm),
            "unique_button_configurations": int(len(np.unique(button_codes))),
            "button_configuration_space_fraction": float(len(np.unique(button_codes)) / (2 ** 16)),
            "exact_duplicate_episode_pairs": duplicate_pairs(hashes),
        }
    reports["train_validation_exact_episode_overlap"] = len(
        split_hashes["train"].intersection(split_hashes["validation"])
    )
    return reports


def audit_mimicgen(path):
    action_min = None
    action_max = None
    action_norm_samples = []
    lengths = []
    hashes = []
    all_finite = True
    done_final_count = 0
    reward_success_count = 0
    image_shapes = set()
    obs_shapes = {}
    with h5py.File(str(path), "r") as dataset:
        data = dataset["data"]
        env_args = json.loads(data.attrs["env_args"])
        demo_names = sorted(data.keys(), key=lambda name: int(name.split("_")[-1]))
        for name in demo_names:
            demo = data[name]
            actions = demo["actions"][:]
            states = demo["states"][:]
            rewards = demo["rewards"][:]
            dones = demo["dones"][:]
            lengths.append(len(actions))
            all_finite = all_finite and all(
                np.isfinite(array).all() for array in (actions, states, rewards, dones)
            )
            action_min = actions.min(axis=0) if action_min is None else np.minimum(action_min, actions.min(axis=0))
            action_max = actions.max(axis=0) if action_max is None else np.maximum(action_max, actions.max(axis=0))
            action_norm_samples.append(np.linalg.norm(actions, axis=-1))
            done_final_count += int(bool(dones[-1]))
            reward_success_count += int(bool(np.max(rewards) > 0))
            hashes.append(episode_hash(actions, states))
            for key, value in demo["obs"].items():
                obs_shapes.setdefault(key, [list(value.shape[1:]), str(value.dtype)])
                if key.endswith("_image"):
                    image_shapes.add(tuple(value.shape[1:]))
                else:
                    block = value[:]
                    all_finite = all_finite and bool(np.isfinite(block).all())
        total_attr = int(data.attrs["total"])

    action_norm = np.concatenate(action_norm_samples)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "episodes": len(lengths),
        "transitions": int(sum(lengths)),
        "total_attribute": total_attr,
        "episode_length": stats(lengths),
        "all_numeric_values_finite": bool(all_finite),
        "exact_duplicate_episode_pairs": duplicate_pairs(hashes),
        "action_dimension": int(len(action_min)),
        "action_component_min": action_min.tolist(),
        "action_component_max": action_max.tolist(),
        "action_limit_violations": int(np.count_nonzero((action_min < -1.0 - 1e-6) | (action_max > 1.0 + 1e-6))),
        "action_norm": stats(action_norm),
        "final_done_count": done_final_count,
        "positive_reward_episode_count": reward_success_count,
        "observation_fields": obs_shapes,
        "image_shapes": [list(shape) for shape in sorted(image_shapes)],
        "environment": env_args,
    }


def fixed_list_to_numpy(column):
    return np.asarray(column.combine_chunks().values).reshape(len(column), column.type.list_size)


def audit_dexjoco(path, info_path):
    table = pq.read_table(str(path))
    actions = fixed_list_to_numpy(table["action"])
    states = fixed_list_to_numpy(table["observation.state"])
    episode_indices = np.asarray(table["episode_index"])
    frame_indices = np.asarray(table["frame_index"])
    timestamps = np.asarray(table["timestamp"])
    hashes = []
    lengths = []
    monotonic_timestamp_episodes = 0
    contiguous_frame_episodes = 0
    for episode in np.unique(episode_indices):
        mask = episode_indices == episode
        lengths.append(int(mask.sum()))
        hashes.append(episode_hash(actions[mask], states[mask]))
        monotonic_timestamp_episodes += int(np.all(np.diff(timestamps[mask]) > 0))
        contiguous_frame_episodes += int(np.array_equal(frame_indices[mask], np.arange(mask.sum())))
    info = json.loads(info_path.read_text())
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "episodes": int(len(np.unique(episode_indices))),
        "transitions": int(len(actions)),
        "episode_length": stats(lengths),
        "all_numeric_values_finite": bool(
            np.isfinite(actions).all() and np.isfinite(states).all() and np.isfinite(timestamps).all()
        ),
        "null_counts": {name: int(table[name].null_count) for name in table.column_names},
        "exact_duplicate_episode_pairs": duplicate_pairs(hashes),
        "action_dimension": int(actions.shape[1]),
        "state_dimension": int(states.shape[1]),
        "action_component_min": actions.min(axis=0).tolist(),
        "action_component_max": actions.max(axis=0).tolist(),
        "action_norm": stats(np.linalg.norm(actions, axis=-1)),
        "monotonic_timestamp_episodes": monotonic_timestamp_episodes,
        "contiguous_frame_index_episodes": contiguous_frame_episodes,
        "metadata": info,
        "download_scope": "state/action parquet and metadata only; official camera videos were not downloaded",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/benchmarks/source"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": "phase6-benchmark-health-v1",
        "ogbench_puzzle_4x4_play": audit_ogbench(
            args.root / "ogbench/puzzle-4x4-play-v0/train.npz",
            args.root / "ogbench/puzzle-4x4-play-v0/validation.npz",
        ),
        "mimicgen_three_piece_assembly_d1_large_interpolation": audit_mimicgen(
            args.root / "mimicgen/large_interpolation/three_piece_assembly_d1.hdf5"
        ),
        "dexjoco_hammer_nail_rand_full": audit_dexjoco(
            args.root / "dexjoco/hammer_nail/data/chunk-000/file-000.parquet",
            args.root / "dexjoco/hammer_nail/meta/info.json",
        ),
        "limitations": [
            "MimicGen and DexJoCo release only successful demonstrations in these selected files.",
            "DexJoCo camera videos were metadata-audited but not downloaded or decoded.",
            "No simulator evaluation reset manifest is available for overlap checks.",
            "Goal leakage requires task-specific relabeling tests after adapters are implemented.",
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
