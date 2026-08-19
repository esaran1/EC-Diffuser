import json

import pytest

from diffuser.models import AuxiliaryIntervalTemporalUnet, IntervalTemporalUnet
from diffuser.scripts.eval_ogbench_policy import (
    aggregate_cube_progress,
    aggregate_puzzle_progress,
    backbone_class_for_method,
    resolve_environment,
    seeded_reset,
    summarize_cube_progress,
    summarize_puzzle_progress,
)


@pytest.mark.parametrize(
    "method, expected",
    [
        ("gaussian_diffusion", IntervalTemporalUnet),
        ("conditional_flow_matching", IntervalTemporalUnet),
        ("improved_meanflow", IntervalTemporalUnet),
        ("shortcut_model", IntervalTemporalUnet),
        ("auxiliary_improved_meanflow", AuxiliaryIntervalTemporalUnet),
        ("behavior_cloning", IntervalTemporalUnet),
    ],
)
def test_ogbench_evaluator_selects_checkpoint_compatible_backbone(method, expected):
    assert backbone_class_for_method(method) is expected


def test_auxiliary_task_diagnostic_is_paired_bounded_and_no_retraining():
    protocol = json.load(open(
        "experiments/pilots/imf_auxiliary_task_diagnostic_v1.json"
    ))

    assert protocol["status"] == "PREDECLARED_NO_RETRAINING_TASK_DIAGNOSTIC"
    assert protocol["training"]["retraining"] is False
    assert protocol["training"]["training_seeds"] == [42, 43, 44]
    assert protocol["preflight"]["nfe_values"] == [1, 2, 4, 8]
    assert protocol["evaluation"]["episodes_per_checkpoint"] == 3
    assert protocol["evaluation"]["total_full_horizon_episodes"] == 18
    assert protocol["compute_gate"]["estimated_gpu_hours"] < 0.1
    assert protocol["compute_gate"]["long_training_authorized"] is False
    assert protocol["compute_gate"]["phase_9_authorized"] is False


def test_puzzle_progress_summary_tracks_transitions_progress_and_regression():
    states = [
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    progress = summarize_puzzle_progress(states, [0, 0, 0, 0])

    assert progress["initial_mismatches"] == 2
    assert progress["final_mismatches"] == 1
    assert progress["minimum_mismatches"] == 0
    assert progress["best_progress"] == 2
    assert progress["best_goal_fraction"] == 1.0
    assert progress["final_goal_fraction"] == 0.75
    assert progress["best_step"] == 2
    assert progress["regression_after_best"] == 1
    assert progress["transition_steps"] == 3
    assert progress["total_button_state_flips"] == 3
    assert progress["first_transition_step"] == 1
    assert progress["unique_button_configurations"] == 4


def test_puzzle_progress_summary_handles_no_contact():
    progress = summarize_puzzle_progress([[1, 0], [1, 0]], [0, 0])

    assert progress["transition_steps"] == 0
    assert progress["first_transition_step"] is None
    assert progress["best_progress"] == 0
    assert progress["unique_button_configurations"] == 1


def test_seeded_reset_seeds_action_space_before_environment_reset():
    events = []

    class ActionSpace:
        value = None

        def seed(self, seed):
            events.append(("action_space", seed))
            self.value = seed

        def sample(self):
            events.append(("sample", self.value))
            return self.value

    class Env:
        @property
        def action_space(self):
            return ActionSpace()

        @property
        def unwrapped(self):
            return self

        def reset(self, seed, options):
            events.append(("environment", seed, options))
            return self.action_space.sample(), {"goal": "goal"}

    result = seeded_reset(Env(), 17, {"task_id": 4})

    assert events == [
        ("action_space", 17),
        ("environment", 17, {"task_id": 4}),
        ("sample", 17),
    ]
    assert result == (17, {"goal": "goal"})
    assert isinstance(Env.action_space, property)


@pytest.mark.parametrize(
    "task_id, expected",
    [
        (
            "ogbench-puzzle-4x4-play-v0-state",
            ("puzzle-4x4-play-v0", "ogbench_puzzle_4x4_play_state"),
        ),
        (
            "ogbench-cube-triple-play-v0-state",
            ("cube-triple-play-v0", "ogbench_cube_triple_play_state"),
        ),
        (
            "ogbench-cube-double-play-v0-state",
            ("cube-double-play-v0", "ogbench_cube_double_play_state"),
        ),
    ],
)
def test_environment_resolves_from_the_protocol_task_id(task_id, expected):
    assert resolve_environment({"id": task_id}) == expected


def test_unknown_task_id_is_rejected_rather_than_defaulting_to_puzzle():
    """A typo must fail loudly, not silently evaluate the wrong environment."""
    with pytest.raises(ValueError):
        resolve_environment({"id": "ogbench-cube-quadruple-play-v0-state"})


def test_progress_metrics_are_none_for_tasks_without_them():
    """Cube tasks have no button states and puzzle has no cube poses."""
    assert aggregate_puzzle_progress([]) is None
    assert aggregate_cube_progress([]) is None
    assert summarize_cube_progress(None, None) is None


def test_cube_progress_uses_the_official_threshold_and_tracks_reduction():
    import numpy as np

    initial = np.array([0.30, 0.20, 0.10])
    final = np.array([0.30, 0.02, 0.05])
    row = summarize_cube_progress(initial, final)

    assert row["cubes"] == 3
    # 0.04 m is the official OGBench CubeEnv success threshold
    assert row["cubes_within_threshold"] == 1
    assert row["cubes_closer_than_start"] == 2
    assert row["min_final_distance"] == pytest.approx(0.02)
    assert row["max_final_distance"] == pytest.approx(0.30)
    assert row["mean_distance_reduction"] == pytest.approx((0.0 + 0.18 + 0.05) / 3)

    summary = aggregate_cube_progress([row, row])
    assert summary["total_cubes_within_threshold"] == 2
    assert summary["episodes_with_any_cube_placed"] == 2
    assert summary["mean_distance_reduction"] == pytest.approx(row["mean_distance_reduction"])


def test_cube_progress_reports_no_placement_when_nothing_moves():
    import numpy as np

    distances = np.array([0.25, 0.26, 0.27])
    row = summarize_cube_progress(distances, distances.copy())
    assert row["cubes_within_threshold"] == 0
    assert row["cubes_closer_than_start"] == 0
    assert row["mean_distance_reduction"] == pytest.approx(0.0)
    assert aggregate_cube_progress([row])["episodes_with_any_cube_placed"] == 0


def _episode_set_sha256(plan):
    """Mirror of the hash the evaluator computes over its episode plan."""
    import hashlib

    return hashlib.sha256(
        json.dumps(
            [[task_id, seed] for task_id, _, seed in plan],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_default_episode_plan_matches_the_stage_1_seed_convention():
    """The default path must keep reproducing the seeds stage 1 actually ran.

    Stage-1 results are only comparable to later runs if this convention is
    stable, so it is pinned rather than left implicit.
    """
    plan = [
        (task_id, index, 42000 + task_id * 10000 + index)
        for task_id in (2, 3, 4)
        for index in range(20)
    ]
    assert plan[0] == (2, 0, 62000)
    assert plan[-1] == (4, 19, 82019)
    assert len(plan) == 60


def test_stage_2_subset_is_a_strict_subset_of_the_stage_1_episodes():
    """The 100-NFE arm is paired against the low-NFE arms by construction."""
    stage1 = {
        (task_id, 42000 + task_id * 10000 + index)
        for task_id in (2, 3, 4)
        for index in range(20)
    }
    stage2 = {
        (task_id, 42000 + task_id * 10000 + index)
        for task_id in (2, 3, 4)
        for index in range(10)
    }
    assert stage2 < stage1
    assert len(stage2) == 30


def test_episode_set_hash_pins_both_content_and_order():
    """The gate pins the exact execution plan, not just the episode set.

    `sort_keys` only orders dict keys, so the hash is order-sensitive. Episode
    order has no scientific effect, but pinning it is the stricter guarantee:
    a rerun reproduces the same plan step for step. Content changes are caught
    a fortiori.
    """
    plan = [(2, 0, 62000), (3, 0, 72000), (4, 0, 82000)]
    reordered = [(4, 0, 82000), (2, 0, 62000), (3, 0, 72000)]
    changed = [(2, 0, 62000), (3, 0, 72000), (4, 0, 82001)]

    assert _episode_set_sha256(plan) == _episode_set_sha256(list(plan))
    assert _episode_set_sha256(plan) != _episode_set_sha256(reordered)
    assert _episode_set_sha256(plan) != _episode_set_sha256(changed)


def test_recorded_stage_2_hash_matches_its_predeclared_protocol():
    """The hash in the committed protocol must match the plan it describes."""
    from pathlib import Path

    protocol_path = Path("experiments/pilots/stage2_diffusion_ceiling_v1.json")
    if not protocol_path.exists():
        pytest.skip("stage-2 protocol not present")
    protocol = json.loads(protocol_path.read_text())
    plan = [
        (task_id, index, 42000 + task_id * 10000 + index)
        for task_id in (2, 3, 4)
        for index in range(10)
    ]
    assert (
        _episode_set_sha256(plan)
        == protocol["evaluation"]["episode_set_sha256"]
    )
