import pickle

import numpy as np

from diffuser.scripts.audit_pushcube_dataset import audit


def _dataset():
    episodes, horizon, entities = 3, 4, 4
    states = np.zeros((episodes, horizon, entities, 6), dtype=np.float32)
    goals = np.zeros_like(states)
    states[..., 2:] = np.eye(4, dtype=np.float32)
    goals[..., 2:] = np.eye(4, dtype=np.float32)
    states[:, :, 1, 0] = np.arange(episodes, dtype=np.float32)[:, None] * 0.01
    goals[:, :, 1:, :2] = 0.1
    return {
        "observations": np.zeros((episodes, horizon, 8, 10), dtype=np.float32),
        "goals": np.zeros((episodes, horizon, 8, 10), dtype=np.float32),
        "state_observations": states,
        "state_goals": goals,
        "actions": np.zeros((episodes, horizon, 3), dtype=np.float32),
        "rewards": np.zeros((episodes, horizon, 1), dtype=np.float32),
        "terminals": np.zeros((episodes, horizon, 1), dtype=np.int64),
        "info_goals_reached": np.array([1.0, 0.0, 1.0]),
        "info_goal_success_frac": np.array([1.0, 0.0, 1.0]),
    }


def test_audit_reports_integrity_and_state_outliers(tmp_path):
    dataset = _dataset()
    dataset["state_observations"][1, 2, 2, 0] = 1.1
    path = tmp_path / "replay.pkl"
    with path.open("wb") as stream:
        pickle.dump(dataset, stream)

    report = audit(path)

    assert report["inventory"]["episodes"] == 3
    assert report["inventory"]["transitions"] == 12
    assert report["integrity"]["all_numeric_values_finite"]
    assert report["integrity"]["possible_clipped_or_invalid_state_episode_indices"] == [1]
    assert report["outcomes"]["full_success_count"] == 2


def test_audit_detects_exact_episode_duplicates(tmp_path):
    dataset = _dataset()
    for key in ("state_observations", "state_goals", "actions"):
        dataset[key][2] = dataset[key][0]
    path = tmp_path / "replay.pkl"
    with path.open("wb") as stream:
        pickle.dump(dataset, stream)

    report = audit(path)

    assert report["integrity"]["exact_duplicate_episode_pairs"] == [[0, 2]]
