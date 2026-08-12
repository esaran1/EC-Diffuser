"""Method-neutral policy-generation timing regression tests."""

import numpy as np
import torch
from torch import nn

from diffuser.models.diffusion import Sample
from diffuser.sampling import GoalConditionedPolicy
from diffuser.utils.arrays import set_global_device


class IdentityNormalizer:
    def normalize(self, value, _key):
        return value

    def unnormalize(self, value, _key):
        return value


class CountingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, trajectory):
        return trajectory + self.anchor * 0.0


class ToyGenerator(nn.Module):
    action_dim = 1
    observation_dim = 2
    horizon = 3

    def __init__(self, steps=4):
        super().__init__()
        self.model = CountingDenoiser()
        self.steps = steps

    def forward(self, conditions, **_kwargs):
        batch = next(iter(conditions.values())).shape[0]
        trajectory = self.model.anchor.new_zeros(batch, self.horizon, 3)
        for _ in range(self.steps):
            trajectory = self.model(trajectory)
        for timestep, value in conditions.items():
            trajectory[:, timestep, self.action_dim:] = value
        return Sample(trajectory, trajectory.new_zeros(batch), None)


def make_policy(measure, warmup=1, steps=4):
    return GoalConditionedPolicy(
        diffusion_model=ToyGenerator(steps),
        normalizer=IdentityNormalizer(),
        preprocess_fns=[],
        measure_planning_latency=measure,
        planning_warmup_calls=warmup,
        count_denoiser_calls=True,
    )


def test_instrumentation_preserves_outputs_and_denoiser_call_count():
    set_global_device("cpu")
    conditions = {0: np.ones((2, 2), dtype=np.float32), 2: np.zeros((2, 2), dtype=np.float32)}
    plain = make_policy(False)
    measured = make_policy(True)

    plain_action, plain_trajectories = plain(conditions, verbose=False)
    measured_action, measured_trajectories = measured(conditions, verbose=False)

    np.testing.assert_array_equal(measured_action, plain_action)
    np.testing.assert_array_equal(measured_trajectories.actions, plain_trajectories.actions)
    np.testing.assert_array_equal(measured_trajectories.observations, plain_trajectories.observations)
    assert plain.denoiser_calls == measured.denoiser_calls == 4


def test_timing_summary_excludes_warmup_and_reports_percentiles():
    set_global_device("cpu")
    policy = make_policy(True, warmup=2, steps=3)
    conditions = {0: np.ones((2, 2), dtype=np.float32), 2: np.zeros((2, 2), dtype=np.float32)}
    for _ in range(7):
        policy(conditions, verbose=False)

    stats = policy.planning_stats()
    assert stats["total_planner_calls"] == 7
    assert stats["warmup_calls"] == 2
    assert stats["timed_calls"] == 5
    assert stats["denoiser_calls"] == 21
    for key in ("mean_ms", "std_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"):
        assert np.isfinite(stats[key]) and stats[key] >= 0.0


def test_negative_warmup_is_rejected():
    set_global_device("cpu")
    try:
        make_policy(True, warmup=-1)
    except ValueError as exc:
        assert "planning_warmup_calls" in str(exc)
    else:
        raise AssertionError("negative warmup was accepted")
