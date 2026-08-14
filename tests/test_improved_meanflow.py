"""Mathematical and integration tests for Improved MeanFlow."""

import copy

import pytest
import torch
from torch import nn

from diffuser.models import ImprovedMeanFlow, IntervalAdaLNPINTDenoiser


H, A, O = 3, 1, 4
D = A + O


class AnalyticAverage(nn.Module):
    def __init__(self, a=0.2, b=0.03, c=-0.04):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.c = nn.Parameter(torch.tensor(c))
        self.calls = 0

    def forward(self, x, cond, time, interval):
        self.calls += 1
        return (
            self.a * x
            + self.b * time[:, None, None]
            + self.c * interval[:, None, None]
        )


def make_wrapper(model=None, **kwargs):
    values = dict(
        model=AnalyticAverage() if model is None else model,
        horizon=H, observation_dim=O, action_dim=A, n_timesteps=4,
        loss_type="l2", action_weight=1.0, loss_discount=1.0,
        loss_weights=None, obs_only=False, action_only=False,
        time_scale=1.0, predict_epsilon=False, clip_denoised=False,
    )
    values.update(kwargs)
    return ImprovedMeanFlow(**values)


def inputs(batch=2, dtype=torch.float32):
    data = torch.randn(batch, H, D, dtype=dtype)
    cond = {0: torch.randn(batch, O, dtype=dtype),
            H - 1: torch.randn(batch, O, dtype=dtype)}
    return data, cond


def assert_conditioned(x, cond):
    for index, value in cond.items():
        torch.testing.assert_close(x[:, index, A:], value, rtol=0, atol=0)


def test_import_finite_loss_backward_and_finite_gradients():
    wrapper = make_wrapper()
    data, cond = inputs()
    loss, info = wrapper.loss(data, cond)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in info.values())
    loss.backward()
    gradients = [p.grad for p in wrapper.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_adaptive_l2_matches_official_per_sample_weighting():
    wrapper = make_wrapper(
        adaptive_weighting=True, adaptive_power=1.0, adaptive_epsilon=0.01
    )
    prediction = torch.tensor(
        [[[1.0] * D] * H, [[2.0] * D] * H], dtype=torch.float32
    )
    target = torch.zeros_like(prediction)
    reference = torch.zeros_like(prediction)
    loss, squared_error, mask = wrapper._meanflow_regression_loss(
        prediction, target, reference, {}
    )
    per_sample = squared_error.sum(dim=(1, 2))
    expected = (per_sample / (per_sample + 0.01).detach()).mean()
    torch.testing.assert_close(loss, expected)
    assert mask.all()


def test_adaptive_weighting_requires_l2_and_valid_parameters():
    with pytest.raises(ValueError, match="requires loss_type='l2'"):
        make_wrapper(loss_type="l1", adaptive_weighting=True)
    with pytest.raises(ValueError, match="adaptive_power"):
        make_wrapper(adaptive_power=0.0)
    with pytest.raises(ValueError, match="adaptive_epsilon"):
        make_wrapper(adaptive_epsilon=0.0)
    with pytest.raises(ValueError, match="adaptive_power"):
        make_wrapper(adaptive_power=float("nan"))
    with pytest.raises(TypeError, match="adaptive_weighting"):
        make_wrapper(adaptive_weighting=1)
    with pytest.raises(TypeError, match="adaptive_epsilon"):
        make_wrapper(adaptive_epsilon="0.01")
    with pytest.raises(TypeError, match="collect_diagnostics"):
        make_wrapper(collect_diagnostics=1)


@pytest.mark.parametrize(
    "batch_size, probability, expected",
    [(1, 0.5, 0), (2, 0.5, 1), (3, 0.5, 1), (32, 0.5, 16),
     (7, 0.0, 0), (7, 1.0, 7)],
)
def test_time_sampler_uses_exact_official_batch_proportion(
    batch_size, probability, expected
):
    wrapper = make_wrapper(boundary_probability=probability)
    r, t, boundary = wrapper._sample_times(
        batch_size, torch.device("cpu"), torch.float32
    )
    assert int(boundary.sum()) == expected
    torch.testing.assert_close(r[boundary], t[boundary])
    assert torch.all(r[~boundary] < t[~boundary])


def test_diagnostics_separate_boundary_interval_and_are_finite():
    wrapper = make_wrapper(collect_diagnostics=True)
    data, cond = inputs(batch=4)
    noise = torch.randn_like(data)
    r = torch.tensor([0.2, 0.4, 0.1, 0.7])
    t = torch.tensor([0.2, 0.8, 0.6, 0.7])
    _, info = wrapper._compute_meanflow_loss(
        data, cond, noise=noise, r=r, t=t
    )
    expected = {
        "boundary_raw_l2", "interval_raw_l2",
        "raw_l2_p50", "raw_l2_p90", "raw_l2_p99",
        "jvp_rms_p50", "jvp_rms_p90", "jvp_rms_p99",
    }
    assert expected <= set(info)
    assert all(torch.isfinite(info[key]) for key in expected)
    assert info["raw_l2_p50"] <= info["raw_l2_p90"] <= info["raw_l2_p99"]
    assert info["jvp_rms_p50"] <= info["jvp_rms_p90"] <= info["jvp_rms_p99"]


def test_known_compound_target_matches_equation_12():
    model = AnalyticAverage(a=0.2, b=0.03, c=-0.04)
    wrapper = make_wrapper(model)
    data = torch.zeros(1, H, D)
    noise = torch.full_like(data, 0.5)
    r = torch.tensor([0.2])
    t = torch.tensor([0.7])
    _, _, details = wrapper._compute_meanflow_loss(
        data, {}, noise=noise, r=r, t=t, return_details=True
    )
    z = 0.7 * noise
    marginal = 0.2 * z + 0.03 * t[:, None, None]
    average = 0.2 * z + 0.03 * t[:, None, None] - 0.04 * (t-r)[:, None, None]
    derivative = 0.2 * marginal + 0.03 - 0.04
    expected = average + (t-r)[:, None, None] * derivative
    torch.testing.assert_close(details["marginal_velocity"], marginal)
    torch.testing.assert_close(details["jvp"], derivative.expand_as(z))
    torch.testing.assert_close(details["compound_velocity"], expected)


def test_boundary_case_reduces_exactly_to_flow_prediction():
    wrapper = make_wrapper()
    data, cond = inputs()
    noise = torch.randn_like(data)
    time = torch.tensor([0.25, 0.75])
    _, _, details = wrapper._compute_meanflow_loss(
        data, cond, noise=noise, r=time, t=time, return_details=True
    )
    torch.testing.assert_close(
        details["compound_velocity"], details["marginal_velocity"]
    )


def test_jvp_is_stopped_but_average_branch_trains():
    wrapper = make_wrapper()
    data, cond = inputs()
    noise = torch.randn_like(data)
    r = torch.tensor([0.1, 0.2])
    t = torch.tensor([0.8, 0.9])
    _, _, details = wrapper._compute_meanflow_loss(
        data, cond, noise=noise, r=r, t=t, return_details=True
    )
    assert details["jvp"].requires_grad
    assert not details["jvp"].detach().requires_grad
    loss = details["compound_velocity"].sum()
    first = torch.autograd.grad(loss, tuple(wrapper.parameters()), create_graph=True)
    assert all(torch.isfinite(value).all() for value in first)


def test_loss_does_not_mutate_inputs_and_conditioning_is_masked():
    wrapper = make_wrapper()
    data, cond = inputs()
    data_before = data.clone()
    cond_before = {k: v.clone() for k, v in cond.items()}
    _, _, details = wrapper._compute_meanflow_loss(
        data, cond, return_details=True
    )
    torch.testing.assert_close(data, data_before)
    for key in cond:
        torch.testing.assert_close(cond[key], cond_before[key])
    for key in cond:
        assert not details["conditioning_mask"][:, key, A:].any()


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_sampling_shape_finiteness_conditioning_and_exact_call_count(steps):
    model = AnalyticAverage()
    wrapper = make_wrapper(model)
    _, cond = inputs(batch=1)
    before = model.calls
    sample = wrapper(cond, n_steps=steps, verbose=False)
    assert model.calls - before == steps
    assert sample.trajectories.shape == (1, H, D)
    assert sample.trajectories.device.type == "cpu"
    assert sample.trajectories.dtype == torch.float32
    assert torch.isfinite(sample.trajectories).all()
    assert_conditioned(sample.trajectories, cond)


def test_dtype_chain_and_checkpoint_roundtrip(tmp_path):
    wrapper = make_wrapper().double()
    _, cond = inputs(batch=1, dtype=torch.float64)
    torch.manual_seed(9)
    expected = wrapper(cond, n_steps=2, return_chain=True, verbose=False)
    path = tmp_path / "state.pt"
    torch.save({"model": wrapper.state_dict(), "ema": wrapper.state_dict(), "step": 7}, path)
    restored = make_wrapper().double()
    state = torch.load(path)
    restored.load_state_dict(state["model"])
    torch.manual_seed(9)
    actual = restored(cond, n_steps=2, return_chain=True, verbose=False)
    torch.testing.assert_close(actual.trajectories, expected.trajectories)
    assert actual.chains.shape == (1, 3, H, D)
    assert state["step"] == 7


def test_real_interval_pint_loss_and_sample():
    model = IntervalAdaLNPINTDenoiser(
        features_dim=2, action_dim=A, hidden_dim=16, projection_dim=16,
        n_head=2, n_layer=1, block_size=H, dropout=0.0,
        positional_bias=False, max_particles=None, multiview=False,
    )
    wrapper = make_wrapper(model)
    data, cond = inputs(batch=2)
    loss, _ = wrapper.loss(data, cond)
    loss.backward()
    gradients = [p.grad for p in wrapper.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    sample = wrapper(cond, n_steps=1, verbose=False)
    assert sample.trajectories.shape == data.shape
    assert torch.isfinite(sample.trajectories).all()
    assert_conditioned(sample.trajectories, cond)
