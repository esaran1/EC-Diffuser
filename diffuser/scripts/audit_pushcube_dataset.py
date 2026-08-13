#!/usr/bin/env python3
"""Audit an EC-Diffuser PushCube replay buffer without modifying it."""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = {
    "observations",
    "actions",
    "goals",
    "state_observations",
    "state_goals",
    "rewards",
    "terminals",
    "info_goals_reached",
    "info_goal_success_frac",
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def array_summary(value):
    array = np.asarray(value)
    summary = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)
        summary.update(
            finite=bool(finite.all()),
            nan_count=int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0,
            inf_count=int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0,
            minimum=float(np.min(array)),
            maximum=float(np.max(array)),
        )
    return summary


def duplicate_episodes(dataset):
    """Hash compact state/action records, avoiding DLP serialization artifacts."""
    first_seen = {}
    duplicates = []
    episode_count = len(dataset["actions"])
    for episode_index in range(episode_count):
        digest = hashlib.sha256()
        for key in ("state_observations", "state_goals", "actions"):
            digest.update(np.ascontiguousarray(dataset[key][episode_index]).view(np.uint8))
        episode_hash = digest.hexdigest()
        if episode_hash in first_seen:
            duplicates.append([first_seen[episode_hash], episode_index])
        else:
            first_seen[episode_hash] = episode_index
    return duplicates


def audit(path):
    with path.open("rb") as stream:
        dataset = pickle.load(stream)

    missing = sorted(REQUIRED_FIELDS.difference(dataset))
    if missing:
        raise ValueError("missing required fields: {}".format(", ".join(missing)))

    actions = np.asarray(dataset["actions"])
    states = np.asarray(dataset["state_observations"])
    goals = np.asarray(dataset["state_goals"])
    rewards = np.asarray(dataset["rewards"])
    terminals = np.asarray(dataset["terminals"])
    successes = np.asarray(dataset["info_goals_reached"])
    success_fraction = np.asarray(dataset["info_goal_success_frac"])

    episode_count, horizon = actions.shape[:2]
    leading_shapes = {
        key: list(np.asarray(value).shape[:2])
        for key, value in dataset.items()
        if np.asarray(value).ndim >= 2 and not key.startswith("info_")
    }
    inconsistent_leading_shapes = {
        key: shape for key, shape in leading_shapes.items() if shape != [episode_count, horizon]
    }

    action_norm = np.linalg.norm(actions, axis=-1)
    object_initial_xy = states[:, 0, 1:, :2]
    object_goal_xy = goals[:, 0, 1:, :2]
    object_final_xy = states[:, -1, 1:, :2]
    initial_goal_distance = np.linalg.norm(object_initial_xy - object_goal_xy, axis=-1)
    final_goal_distance = np.linalg.norm(object_final_xy - object_goal_xy, axis=-1)
    duplicates = duplicate_episodes(dataset)
    endpoint_outlier_indices = np.argwhere(final_goal_distance > 0.1)
    clipped_state_indices = np.argwhere(np.abs(states[:, :, 1:, :2]) >= 1.0)

    entity_coverage = []
    for entity_index in range(object_initial_xy.shape[1]):
        entity_coverage.append(
            {
                "entity_index": entity_index + 1,
                "initial_xy_min": object_initial_xy[:, entity_index].min(axis=0).tolist(),
                "initial_xy_max": object_initial_xy[:, entity_index].max(axis=0).tolist(),
                "goal_xy_min": object_goal_xy[:, entity_index].min(axis=0).tolist(),
                "goal_xy_max": object_goal_xy[:, entity_index].max(axis=0).tolist(),
                "initial_goal_distance": quantiles(initial_goal_distance[:, entity_index]),
                "final_goal_distance": quantiles(final_goal_distance[:, entity_index]),
            }
        )

    fields = {key: array_summary(value) for key, value in sorted(dataset.items())}
    all_finite = all(field.get("finite", True) for field in fields.values())
    action_limit_violations = int((np.abs(actions) > 1.0 + 1e-6).sum())
    success_values, success_counts = np.unique(successes, return_counts=True)
    frac_values, frac_counts = np.unique(success_fraction, return_counts=True)

    return {
        "schema_version": "ecdiffuser-dataset-health-v1",
        "source": {
            "path": str(path),
            "size_bytes": os.path.getsize(str(path)),
            "sha256": sha256_file(path),
        },
        "inventory": {
            "episodes": int(episode_count),
            "transitions": int(episode_count * horizon),
            "fixed_horizon": int(horizon),
            "fields": fields,
        },
        "integrity": {
            "required_fields_present": True,
            "all_numeric_values_finite": all_finite,
            "inconsistent_leading_shapes": inconsistent_leading_shapes,
            "exact_duplicate_episode_count": len(duplicates),
            "exact_duplicate_episode_pairs": duplicates,
            "action_limit_violations": action_limit_violations,
            "terminal_nonzero_count": int(np.count_nonzero(terminals)),
            "endpoint_distance_over_0_1_count": int(len(endpoint_outlier_indices)),
            "endpoint_distance_over_0_1_episode_indices": sorted(
                {int(index[0]) for index in endpoint_outlier_indices}
            ),
            "possible_clipped_or_invalid_state_count": int(len(clipped_state_indices)),
            "possible_clipped_or_invalid_state_episode_indices": sorted(
                {int(index[0]) for index in clipped_state_indices}
            ),
            "goal_constant_across_time": bool(np.all(goals == goals[:, :1])),
            "dlp_goal_constant_across_time": bool(
                np.all(np.asarray(dataset["goals"]) == np.asarray(dataset["goals"])[:, :1])
            ),
        },
        "outcomes": {
            "full_success_count": int(np.sum(successes == 1)),
            "full_success_rate": float(np.mean(successes == 1)),
            "success_value_counts": {
                str(float(value)): int(count) for value, count in zip(success_values, success_counts)
            },
            "goal_fraction_mean": float(np.mean(success_fraction)),
            "goal_fraction_value_counts": {
                str(float(value)): int(count) for value, count in zip(frac_values, frac_counts)
            },
            "episode_return": quantiles(rewards.sum(axis=tuple(range(1, rewards.ndim)))),
        },
        "coverage": {
            "action_component_min": actions.min(axis=(0, 1)).tolist(),
            "action_component_max": actions.max(axis=(0, 1)).tolist(),
            "action_norm": quantiles(action_norm),
            "action_saturation_fraction": float(np.mean(np.abs(actions) >= 1.0 - 1e-7)),
            "object_entity_coverage": entity_coverage,
            "initial_goal_distance_all_objects": quantiles(initial_goal_distance),
            "final_goal_distance_all_objects": quantiles(final_goal_distance),
        },
        "audit_limitations": [
            "The pickle has no explicit train/validation/test split or episode seeds.",
            "All episodes are padded/fixed at 100 steps and terminals are always zero; boundaries come from array axes.",
            "The source file has no camera RGB frames, simulator state validity flags, or orientation variation.",
            "Train/evaluation overlap cannot be excluded without evaluation reset/goal manifests.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.dataset)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
