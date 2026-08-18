import json

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from diffuser.datasets.benchmark_sequence import (
    DexJoCoHammerWindowDataset,
    MimicGenThreePieceWindowDataset,
    OGBenchCubeDoubleWindowDataset,
    OGBenchCubeTripleWindowDataset,
    OGBenchPuzzleWindowDataset,
)


def _write_manifest(path, payload):
    path.write_text(json.dumps(payload))
    return path


def test_ogbench_goals_are_deterministic_and_episode_local(tmp_path):
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    observations = np.arange(18, dtype=np.float32).reshape(6, 3)
    actions = np.linspace(-1, 1, 30, dtype=np.float32).reshape(6, 5)
    np.savez(train, observations=observations, actions=actions, terminals=[0, 0, 0, 0, 0, 1])
    np.savez(
        validation,
        observations=observations + 1000,
        actions=actions + 100,
        terminals=[0, 0, 0, 0, 0, 1],
    )
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "schema_version": "fast-generative-policy-dataset-v1",
            "task_id": "ogbench-puzzle-4x4-play-v0-state",
            "source_paths": {"train": str(train), "validation": str(validation)},
            "episode_offsets": {
                "train": [{"episode_id": "train:0000", "start": 0, "end": 6}],
                "validation": [{"episode_id": "validation:0000", "start": 0, "end": 6}],
            },
            "splits": {"train": ["train:0000"], "validation": ["validation:0000"]},
        },
    )
    dataset = OGBenchPuzzleWindowDataset(manifest, split="train", horizon=3, goal_seed=9)
    first = dataset[0]
    repeated = dataset[0]
    meta = dataset.metadata(0)

    np.testing.assert_array_equal(first.trajectories, repeated.trajectories)
    assert 2 <= meta["goal_timestep"] < 6
    np.testing.assert_array_equal(first.trajectories[-1, 5:], first.conditions[2])
    restored_action = dataset.normalizer.unnormalize(first.trajectories[-1, :5], "actions")
    np.testing.assert_allclose(restored_action, 0.0, atol=1e-6)
    assert dataset.normalizer.observation_stats["count"] == 6
    np.testing.assert_allclose(dataset.normalizer.observation_stats["mean"], observations.mean(0))

    frozen_normalizer = dataset.normalizer_state_dict()
    train.unlink()
    reloaded = OGBenchPuzzleWindowDataset(
        manifest, split="validation", horizon=3, normalizer_state=frozen_normalizer
    )
    assert reloaded.normalizer_state_dict() == frozen_normalizer
    assert reloaded[0].trajectories.shape == (3, 8)


def _write_mimicgen(path):
    with h5py.File(path, "w") as source:
        data = source.create_group("data")
        for episode_id, offset in ((0, 0.0), (1, 1000.0)):
            demo = data.create_group("demo_{}".format(episode_id))
            demo.create_dataset("actions", data=np.arange(35, dtype=np.float32).reshape(5, 7) + offset)
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_eef_pos", data=np.full((5, 3), offset))
            obs.create_dataset("robot0_eef_quat", data=np.full((5, 4), offset + 1))
            obs.create_dataset("robot0_gripper_qpos", data=np.full((5, 2), offset + 2))
            obs.create_dataset("object", data=np.full((5, 6), offset + 3))


def test_mimicgen_uses_official_lowdim_keys_and_train_only_stats(tmp_path):
    source = tmp_path / "mimic.hdf5"
    _write_mimicgen(source)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "schema_version": "fast-generative-policy-dataset-v1",
            "task_id": "mimicgen-three-piece-assembly-d1-large-interpolation",
            "source_path": str(source),
            "splits": {"train": [0], "validation": [1]},
            "episode_lengths": {"0": 5, "1": 5},
        },
    )
    dataset = MimicGenThreePieceWindowDataset(manifest, split="validation", horizon=3)
    batch = dataset[0]

    assert dataset.observation_dim == 15
    assert dataset.action_dim == 7
    assert dataset.normalizer.observation_stats["count"] == 5
    assert set(batch.conditions) == {0}
    assert batch.trajectories.shape == (3, 22)
    assert dataset.metadata(0)["episode_id"] == 1


def _fixed_list(values, width):
    return pa.array(values.tolist(), type=pa.list_(pa.float32(), width))


def test_dexjoco_uses_nonprivileged_state_and_episode_splits(tmp_path):
    source = tmp_path / "dex.parquet"
    actions = np.arange(8 * 22, dtype=np.float32).reshape(8, 22)
    states = np.arange(8 * 23, dtype=np.float32).reshape(8, 23)
    table = pa.table(
        {
            "action": _fixed_list(actions, 22),
            "observation.state": _fixed_list(states, 23),
            "episode_index": pa.array([0] * 4 + [1] * 4, type=pa.int64()),
            "frame_index": pa.array([0, 1, 2, 3] * 2, type=pa.int64()),
            "task_index": pa.array([0] * 8, type=pa.int64()),
        }
    )
    pq.write_table(table, source)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "schema_version": "fast-generative-policy-dataset-v1",
            "task_id": "dexjoco-hammer-nail-rand-full",
            "source_path": str(source),
            "splits": {"train": [0], "validation": [1]},
        },
    )
    dataset = DexJoCoHammerWindowDataset(manifest, split="validation", horizon=3)
    batch = dataset[0]

    assert dataset.observation_dim == 23
    assert dataset.action_dim == 22
    assert dataset.normalizer.observation_stats["count"] == 4
    assert set(batch.conditions) == {0}
    assert batch.trajectories.shape == (3, 45)
    assert dataset.metadata(0)["episode_id"] == 1
    loader_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)))
    assert loader_batch.trajectories.shape == (1, 3, 45)


def test_invalid_horizon_is_rejected():
    try:
        OGBenchPuzzleWindowDataset.__new__(OGBenchPuzzleWindowDataset).__init__(
            "missing", horizon=1
        )
    except ValueError as error:
        assert "horizon" in str(error)
    else:
        raise AssertionError("horizon=1 should be rejected")


def _write_ogbench_pair(tmp_path, task_id, dim=3, action_dim=5, length=6):
    tmp_path.mkdir(parents=True, exist_ok=True)
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    observations = np.arange(length * dim, dtype=np.float32).reshape(length, dim)
    actions = np.linspace(-1, 1, length * action_dim, dtype=np.float32).reshape(length, action_dim)
    terminals = [0] * (length - 1) + [1]
    np.savez(train, observations=observations, actions=actions, terminals=terminals)
    np.savez(
        validation,
        observations=observations + 1000,
        actions=actions + 100,
        terminals=terminals,
    )
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "schema_version": "fast-generative-policy-dataset-v1",
            "task_id": task_id,
            "source_paths": {"train": str(train), "validation": str(validation)},
            "episode_offsets": {
                "train": [{"episode_id": "train:0000", "start": 0, "end": length}],
                "validation": [{"episode_id": "validation:0000", "start": 0, "end": length}],
            },
            "splits": {"train": ["train:0000"], "validation": ["validation:0000"]},
        },
    )
    return manifest


def test_cube_datasets_load_and_reject_foreign_manifests(tmp_path):
    """Cube adapters honor their own task_id and reject mismatched manifests."""
    for cls, task_id in (
        (OGBenchCubeTripleWindowDataset, "ogbench-cube-triple-play-v0-state"),
        (OGBenchCubeDoubleWindowDataset, "ogbench-cube-double-play-v0-state"),
    ):
        directory = tmp_path / task_id
        manifest = _write_ogbench_pair(directory, task_id)
        dataset = cls(manifest, split="train", horizon=3, goal_seed=9)
        batch = dataset[0]
        assert batch.trajectories.shape == (3, 8)
        assert np.isfinite(batch.trajectories).all()
        # goal occupies the final observation slot and the terminal condition
        np.testing.assert_array_equal(batch.trajectories[-1, 5:], batch.conditions[2])
        assert dataset.metadata(0)["task_id"] == task_id

        wrong = _write_ogbench_pair(
            tmp_path / (task_id + "-wrong"), "ogbench-puzzle-4x4-play-v0-state"
        )
        try:
            cls(wrong, split="train", horizon=3)
        except ValueError:
            pass
        else:
            raise AssertionError("{} accepted a foreign manifest".format(cls.__name__))


def test_ogbench_tasks_share_one_goal_relabeling_rule(tmp_path):
    """Puzzle and cube adapters must relabel goals identically.

    The cross-task goal-offset comparisons in experiments/protocols/ are only
    valid if the goal rule is literally the same code path for every OGBench
    state task. This pins that invariant.
    """
    specs = [
        (OGBenchPuzzleWindowDataset, "ogbench-puzzle-4x4-play-v0-state"),
        (OGBenchCubeTripleWindowDataset, "ogbench-cube-triple-play-v0-state"),
        (OGBenchCubeDoubleWindowDataset, "ogbench-cube-double-play-v0-state"),
    ]
    goal_timesteps = []
    for cls, task_id in specs:
        directory = tmp_path / ("shared-" + task_id)
        manifest = _write_ogbench_pair(directory, task_id, length=12)
        dataset = cls(manifest, split="train", horizon=4, goal_seed=1234)
        goal_timesteps.append(
            [dataset.metadata(i)["goal_timestep"] for i in range(len(dataset))]
        )
        # goals stay inside the episode and never precede the window endpoint
        for i in range(len(dataset)):
            meta = dataset.metadata(i)
            assert meta["timestep"] + 3 <= meta["goal_timestep"] < 12

    assert goal_timesteps[0] == goal_timesteps[1] == goal_timesteps[2], (
        "OGBench state tasks diverged in goal relabeling; cross-task goal-offset "
        "comparisons would be invalid"
    )
