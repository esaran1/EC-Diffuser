import json

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from diffuser.scripts.audit_phase6_benchmarks import audit_dexjoco, audit_mimicgen, audit_ogbench
from diffuser.scripts.build_phase6_conversion_manifests import mimicgen_manifest


def _write_ogbench(path, offset=0.0):
    terminals = np.array([False, True, False, True])
    np.savez_compressed(
        path,
        observations=np.arange(12, dtype=np.float32).reshape(4, 3) + offset,
        actions=np.zeros((4, 5), dtype=np.float32),
        terminals=terminals,
        qpos=np.zeros((4, 2), dtype=np.float32),
        qvel=np.zeros((4, 2), dtype=np.float32),
        button_states=np.zeros((4, 16), dtype=np.int64),
    )


def test_ogbench_audit_uses_terminal_episode_boundaries(tmp_path):
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    _write_ogbench(train)
    _write_ogbench(validation, offset=100.0)

    report = audit_ogbench(train, validation)

    assert report["train"]["episodes"] == 2
    assert report["train"]["episode_length"]["mean"] == 2.0
    assert report["train_validation_exact_episode_overlap"] == 0


def _write_mimicgen(path):
    with h5py.File(str(path), "w") as dataset:
        data = dataset.create_group("data")
        data.attrs["total"] = 6
        data.attrs["env_args"] = json.dumps({"env_name": "toy"})
        for index, successful in enumerate((True, False)):
            demo = data.create_group("demo_{}".format(index))
            demo.create_dataset("actions", data=np.zeros((3, 7)))
            demo.create_dataset("states", data=np.full((3, 2), index))
            demo.create_dataset("rewards", data=np.array([0.0, 0.0, float(successful)]))
            demo.create_dataset("dones", data=np.array([0, 0, 1]))
            obs = demo.create_group("obs")
            obs.create_dataset("object", data=np.zeros((3, 4)))
            obs.create_dataset("agentview_image", data=np.zeros((3, 2, 2, 3), dtype=np.uint8))


def test_mimicgen_audit_and_manifest_quarantine_zero_reward(tmp_path):
    root = tmp_path / "source"
    path = root / "mimicgen/large_interpolation/three_piece_assembly_d1.hdf5"
    path.parent.mkdir(parents=True)
    _write_mimicgen(path)

    report = audit_mimicgen(path)
    manifest = mimicgen_manifest(root, seed=7)

    assert report["episodes"] == 2
    assert report["positive_reward_episode_count"] == 1
    assert manifest["quarantined_episode_ids"] == [1]
    assert manifest["splits"]["train"] == []
    assert manifest["splits"]["validation"] == [0]


def test_dexjoco_audit_checks_episode_timestamps_and_frames(tmp_path):
    path = tmp_path / "demo.parquet"
    info_path = tmp_path / "info.json"
    actions = np.zeros((4, 22), dtype=np.float32)
    states = np.zeros((4, 23), dtype=np.float32)
    table = pa.table(
        {
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 22)),
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 23)),
            "timestamp": pa.array([0.0, 0.1, 0.0, 0.1], type=pa.float32()),
            "frame_index": pa.array([0, 1, 0, 1], type=pa.int64()),
            "episode_index": pa.array([0, 0, 1, 1], type=pa.int64()),
            "index": pa.array([0, 1, 2, 3], type=pa.int64()),
            "task_index": pa.array([0, 0, 0, 0], type=pa.int64()),
        }
    )
    pq.write_table(table, path)
    info_path.write_text(json.dumps({"total_episodes": 2, "total_frames": 4}))

    report = audit_dexjoco(path, info_path)

    assert report["episodes"] == 2
    assert report["monotonic_timestamp_episodes"] == 2
    assert report["contiguous_frame_index_episodes"] == 2
