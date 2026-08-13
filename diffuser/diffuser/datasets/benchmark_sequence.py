"""Deterministic, zero-copy adapters for the Phase 6 benchmark datasets.

The adapters expose the EC-Diffuser Batch(trajectories, conditions) contract
while retaining task and source metadata through metadata(index). Source
datasets are never modified. Normalization statistics are fit on training
episodes only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .sequence import Batch


class _RunningStats:
    def __init__(self):
        self.count = 0
        self.total = None
        self.square_total = None
        self.minimum = None
        self.maximum = None

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("normalizer input must be a finite rank-2 array")
        block_sum = values.sum(axis=0)
        block_square_sum = np.square(values).sum(axis=0)
        block_min = values.min(axis=0)
        block_max = values.max(axis=0)
        if self.total is None:
            self.total = block_sum
            self.square_total = block_square_sum
            self.minimum = block_min
            self.maximum = block_max
        else:
            self.total += block_sum
            self.square_total += block_square_sum
            self.minimum = np.minimum(self.minimum, block_min)
            self.maximum = np.maximum(self.maximum, block_max)
        self.count += len(values)

    def finish(self):
        if self.count == 0:
            raise ValueError("cannot fit a normalizer on zero transitions")
        mean = self.total / self.count
        variance = np.maximum(self.square_total / self.count - np.square(mean), 0.0)
        return {
            "count": self.count,
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
            "min": self.minimum.astype(np.float32),
            "max": self.maximum.astype(np.float32),
        }


class TrainSplitNormalizer:
    """Gaussian observations and safe train-range actions."""

    def __init__(self, observation_stats, action_stats, epsilon=1e-6):
        self.observation_stats = observation_stats
        self.action_stats = action_stats
        self.epsilon = float(epsilon)

    @classmethod
    def fit(cls, blocks):
        observation_stats = _RunningStats()
        action_stats = _RunningStats()
        for observations, actions in blocks:
            observation_stats.update(observations)
            action_stats.update(actions)
        return cls(observation_stats.finish(), action_stats.finish())

    @classmethod
    def from_state_dict(cls, state):
        if state.get("schema_version") != "train-split-normalizer-v1":
            raise ValueError("unsupported normalizer schema")
        def deserialize(stats):
            return {
                "count": int(stats["count"]),
                "mean": np.asarray(stats["mean"], dtype=np.float32),
                "std": np.asarray(stats["std"], dtype=np.float32),
                "min": np.asarray(stats["min"], dtype=np.float32),
                "max": np.asarray(stats["max"], dtype=np.float32),
            }
        return cls(deserialize(state["observation"]), deserialize(state["action"]))

    def normalize(self, values, key):
        values = np.asarray(values, dtype=np.float32)
        if key in ("observations", "goals"):
            stats = self.observation_stats
            return (values - stats["mean"]) / (stats["std"] + self.epsilon)
        if key == "actions":
            stats = self.action_stats
            width = stats["max"] - stats["min"]
            safe_width = np.where(width > self.epsilon, width, 1.0)
            result = 2.0 * (values - stats["min"]) / safe_width - 1.0
            return np.where(width > self.epsilon, result, 0.0)
        raise KeyError(key)

    def unnormalize(self, values, key):
        values = np.asarray(values, dtype=np.float32)
        if key in ("observations", "goals"):
            stats = self.observation_stats
            return values * (stats["std"] + self.epsilon) + stats["mean"]
        if key == "actions":
            stats = self.action_stats
            width = stats["max"] - stats["min"]
            safe_width = np.where(width > self.epsilon, width, 1.0)
            result = (values + 1.0) * 0.5 * safe_width + stats["min"]
            return np.where(width > self.epsilon, result, stats["min"])
        raise KeyError(key)

    def state_dict(self):
        def serialize(stats):
            return {
                "count": int(stats["count"]),
                "mean": stats["mean"].tolist(),
                "std": stats["std"].tolist(),
                "min": stats["min"].tolist(),
                "max": stats["max"].tolist(),
            }
        return {
            "schema_version": "train-split-normalizer-v1",
            "observation": serialize(self.observation_stats),
            "action": serialize(self.action_stats),
        }


def _load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != "fast-generative-policy-dataset-v1":
        raise ValueError("unsupported dataset manifest schema")
    return manifest


def _stable_integer(*parts):
    payload = json.dumps(parts, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class _BaseBenchmarkDataset(torch.utils.data.Dataset):
    task_id = None

    def __init__(self, horizon, goal_seed=42):
        if horizon < 2:
            raise ValueError("horizon must be at least two")
        self.horizon = int(horizon)
        self.goal_seed = int(goal_seed)
        self.observation_dim = None
        self.action_dim = None
        self.indices = []
        self.normalizer = None

    def metadata(self, index):
        raise NotImplementedError

    def normalizer_state_dict(self):
        return self.normalizer.state_dict()

    def _batch(self, observations, actions, goal=None):
        observations = self.normalizer.normalize(observations, "observations").astype(np.float32)
        actions = self.normalizer.normalize(actions, "actions").astype(np.float32)
        conditions = {0: observations[0].copy()}
        if goal is not None:
            normalized_goal = self.normalizer.normalize(goal, "goals").astype(np.float32)
            observations = observations.copy()
            observations[-1] = normalized_goal
            conditions[self.horizon - 1] = normalized_goal.copy()
        trajectories = np.concatenate((actions, observations), axis=-1).astype(np.float32)
        if trajectories.shape != (self.horizon, self.action_dim + self.observation_dim):
            raise RuntimeError("adapter produced an invalid trajectory shape")
        if not np.isfinite(trajectories).all():
            raise FloatingPointError("adapter produced non-finite values")
        return Batch(trajectories, conditions)


class OGBenchPuzzleWindowDataset(_BaseBenchmarkDataset):
    """Official Puzzle-4x4 state stream with within-episode future goals."""

    task_id = "ogbench-puzzle-4x4-play-v0-state"

    def __init__(self, manifest_path, split="train", horizon=5, goal_seed=42, normalizer_state=None):
        super().__init__(horizon=horizon, goal_seed=goal_seed)
        manifest = _load_manifest(manifest_path)
        if manifest["task_id"] != self.task_id or split not in ("train", "validation"):
            raise ValueError("invalid OGBench manifest or split")
        self.manifest = manifest
        self.split = split
        with np.load(str(Path(manifest["source_paths"][split]))) as source:
            self.observations = np.asarray(source["observations"])
            self.actions = np.asarray(source["actions"])
            self.terminals = np.asarray(source["terminals"])
        self.observation_dim = int(self.observations.shape[1])
        self.action_dim = int(self.actions.shape[1])
        offsets = manifest["episode_offsets"][split]
        for episode_index, offset in enumerate(offsets):
            start, end = int(offset["start"]), int(offset["end"])
            for window_start in range(start, end - self.horizon + 1):
                self.indices.append((episode_index, window_start, end))
        if normalizer_state is not None:
            self.normalizer = TrainSplitNormalizer.from_state_dict(normalizer_state)
        elif split == "train":
            self.normalizer = TrainSplitNormalizer.fit([(self.observations, self.actions)])
        else:
            with np.load(str(Path(manifest["source_paths"]["train"]))) as source:
                train_observations = np.asarray(source["observations"])
                train_actions = np.asarray(source["actions"])
            self.normalizer = TrainSplitNormalizer.fit([(train_observations, train_actions)])

    def __len__(self):
        return len(self.indices)

    def _goal_index(self, index, window_start, episode_end):
        lower = window_start + self.horizon - 1
        span = episode_end - lower
        return lower + _stable_integer(self.goal_seed, index, window_start) % span

    def __getitem__(self, index):
        _, start, episode_end = self.indices[index]
        goal_index = self._goal_index(index, start, episode_end)
        observations = self.observations[start:start + self.horizon].copy()
        actions = self.actions[start:start + self.horizon].copy()
        actions[-1] = 0.0
        return self._batch(observations, actions, goal=self.observations[goal_index])

    def metadata(self, index):
        episode_index, start, episode_end = self.indices[index]
        goal_index = self._goal_index(index, start, episode_end)
        episode_start = int(self.manifest["episode_offsets"][self.split][episode_index]["start"])
        return {
            "episode_id": self.manifest["splits"][self.split][episode_index],
            "timestep": start - episode_start,
            "goal_timestep": goal_index - episode_start,
            "success": None,
            "task_id": self.task_id,
        }


class MimicGenThreePieceWindowDataset(_BaseBenchmarkDataset):
    """Official low-dimensional MimicGen observation selection."""

    task_id = "mimicgen-three-piece-assembly-d1-large-interpolation"
    observation_keys = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    )

    def __init__(self, manifest_path, split="train", horizon=10, goal_seed=42, normalizer_state=None):
        super().__init__(horizon=horizon, goal_seed=goal_seed)
        manifest = _load_manifest(manifest_path)
        if manifest["task_id"] != self.task_id or split not in ("train", "validation"):
            raise ValueError("invalid MimicGen manifest or split")
        self.manifest = manifest
        self.split = split
        self.source_path = Path(manifest["source_path"])
        self.episode_ids = [int(value) for value in manifest["splits"][split]]
        self._handle = None
        self._handle_pid = None
        for episode_id in self.episode_ids:
            length = int(manifest["episode_lengths"][str(episode_id)])
            for start in range(0, length - self.horizon + 1):
                self.indices.append((episode_id, start))
        self.normalizer = (
            TrainSplitNormalizer.from_state_dict(normalizer_state)
            if normalizer_state is not None
            else TrainSplitNormalizer.fit(self._training_blocks())
        )
        self.observation_dim = int(self.normalizer.observation_stats["mean"].shape[0])
        self.action_dim = int(self.normalizer.action_stats["mean"].shape[0])

    def _file(self):
        import h5py
        pid = os.getpid()
        if self._handle is not None and self._handle_pid != pid:
            self._handle.close()
            self._handle = None
        if self._handle is None:
            self._handle = h5py.File(str(self.source_path), "r")
            self._handle_pid = pid
        return self._handle

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._handle_pid = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def _read_observations(self, demo):
        return np.concatenate([np.asarray(demo["obs"][key]) for key in self.observation_keys], axis=-1)

    def _training_blocks(self):
        import h5py
        with h5py.File(str(self.source_path), "r") as source:
            for episode_id in self.manifest["splits"]["train"]:
                demo = source["data"]["demo_{}".format(episode_id)]
                yield self._read_observations(demo), np.asarray(demo["actions"])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        episode_id, start = self.indices[index]
        demo = self._file()["data"]["demo_{}".format(episode_id)]
        observations = np.concatenate(
            [np.asarray(demo["obs"][key][start:start + self.horizon]) for key in self.observation_keys],
            axis=-1,
        )
        actions = np.asarray(demo["actions"][start:start + self.horizon])
        return self._batch(observations, actions)

    def metadata(self, index):
        episode_id, start = self.indices[index]
        return {
            "episode_id": episode_id,
            "timestep": start,
            "success": True,
            "task_id": self.task_id,
            "goal": "assemble all three pieces",
        }


class DexJoCoHammerWindowDataset(_BaseBenchmarkDataset):
    """Official non-privileged state/action slice from the LeRobot release."""

    task_id = "dexjoco-hammer-nail-rand-full"

    def __init__(self, manifest_path, split="train", horizon=30, goal_seed=42, normalizer_state=None):
        super().__init__(horizon=horizon, goal_seed=goal_seed)
        manifest = _load_manifest(manifest_path)
        if manifest["task_id"] != self.task_id or split not in ("train", "validation"):
            raise ValueError("invalid DexJoCo manifest or split")
        self.manifest = manifest
        self.split = split
        import pyarrow.parquet as pq
        table = pq.read_table(
            manifest["source_path"],
            columns=["action", "observation.state", "episode_index", "frame_index", "task_index"],
        )
        self.actions = self._fixed_list(table["action"])
        self.observations = self._fixed_list(table["observation.state"])
        self.episode_indices = np.asarray(table["episode_index"])
        self.frame_indices = np.asarray(table["frame_index"])
        self.task_indices = np.asarray(table["task_index"])
        split_ids = set(int(value) for value in manifest["splits"][split])
        for episode_id in sorted(split_ids):
            rows = np.flatnonzero(self.episode_indices == episode_id)
            if not np.array_equal(self.frame_indices[rows], np.arange(len(rows))):
                raise ValueError("DexJoCo frame indices are not contiguous")
            for offset in range(0, len(rows) - self.horizon + 1):
                self.indices.append((episode_id, int(rows[offset])))
        train_ids = set(int(value) for value in manifest["splits"]["train"])
        train_mask = np.isin(self.episode_indices, list(train_ids))
        self.normalizer = (
            TrainSplitNormalizer.from_state_dict(normalizer_state)
            if normalizer_state is not None
            else TrainSplitNormalizer.fit(
                [(self.observations[train_mask], self.actions[train_mask])]
            )
        )
        self.observation_dim = int(self.observations.shape[1])
        self.action_dim = int(self.actions.shape[1])

    @staticmethod
    def _fixed_list(column):
        combined = column.combine_chunks()
        return np.asarray(combined.values).reshape(len(column), combined.type.list_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        _, start = self.indices[index]
        end = start + self.horizon
        return self._batch(self.observations[start:end], self.actions[start:end])

    def metadata(self, index):
        episode_id, start = self.indices[index]
        return {
            "episode_id": episode_id,
            "timestep": int(self.frame_indices[start]),
            "success": True,
            "task_id": self.task_id,
            "source_task_index": int(self.task_indices[start]),
            "goal": "drive the nail at least 0.04 m into the board",
        }
