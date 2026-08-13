import torch
from isaaclab_pushcube.reference import LegacyPushCubeSpec, sample_collision_free_xy, scale_action, success_metrics


def test_legacy_timing_and_geometry():
    spec = LegacyPushCubeSpec()
    assert spec.physics_dt == 0.01667
    assert spec.simulation_dt == 0.008335
    assert spec.legacy_outer_steps_per_action == 19
    assert spec.decimation == 38
    assert abs(spec.policy_step_duration - 0.31673) < 1e-12
    assert abs(spec.effective_control_frequency_hz - 3.157263284185268) < 1e-12
    assert abs(spec.episode_length_s - 31.673) < 1e-12
    assert spec.table_surface_z == 1.025
    assert spec.reset_xy_limit == (0.15, 0.19999999999999998)
    assert spec.robot_base_pos == (-0.46, 0.0, 0.945)


def test_action_scaling_clips_and_preserves_input():
    spec = LegacyPushCubeSpec()
    actions = torch.tensor([[-2.0, 0.5, 2.0]])
    before = actions.clone()
    actual = scale_action(actions, spec)
    torch.testing.assert_close(actions, before)
    torch.testing.assert_close(actual, torch.tensor([[-0.125, 0.0625, 0.125]]))


def test_success_predicate_matches_legacy_strict_threshold():
    obj = torch.tensor([[[0.0, 0.0], [0.039, 0.0], [0.04, 0.0]]])
    goal = torch.zeros_like(obj)
    result = success_metrics(obj, goal, 0.04)
    assert not result["success"].item()
    torch.testing.assert_close(result["goal_success_fraction"], torch.tensor([2 / 3]))
    torch.testing.assert_close(result["maximum_object_goal_distance"], torch.tensor([0.04]))


def test_seeded_reset_distribution_is_reproducible_and_collision_free():
    spec = LegacyPushCubeSpec()
    a = sample_collision_free_xy(256, 3, *spec.reset_xy_limit, spec.cube_size * 2**0.5, torch.Generator().manual_seed(7))
    b = sample_collision_free_xy(256, 3, *spec.reset_xy_limit, spec.cube_size * 2**0.5, torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert a[..., 0].abs().max() <= spec.reset_xy_limit[0]
    assert a[..., 1].abs().max() <= spec.reset_xy_limit[1]
    distances = torch.cdist(a, a)
    eye = torch.eye(3, dtype=torch.bool).expand(256, -1, -1)
    assert torch.all(distances[~eye] >= spec.cube_size * 2**0.5)
