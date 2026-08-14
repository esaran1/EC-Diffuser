"""Mathematical and integration tests for auxiliary-head iMF."""

import pytest
import torch
from torch import nn

from diffuser.models import (
    AuxiliaryImprovedMeanFlow,
    AuxiliaryIntervalTemporalUnet,
    IntervalTemporalUnet,
)

H, A, O = 3, 1, 4
D = A + O


class AnalyticDual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.2))
        self.b = nn.Parameter(torch.tensor(0.03))
        self.c = nn.Parameter(torch.tensor(-0.04))
        self.d = nn.Parameter(torch.tensor(0.15))
        self.e = nn.Parameter(torch.tensor(-0.02))
        self.f = nn.Parameter(torch.tensor(0.05))
        self.forward_calls = 0
        self.aux_calls = 0

    def forward(self, x, cond, time, interval):
        self.forward_calls += 1
        return self.a*x + self.b*time[:, None, None] + self.c*interval[:, None, None]

    def forward_with_aux(self, x, cond, time, interval):
        self.aux_calls += 1
        u = self.a*x + self.b*time[:, None, None] + self.c*interval[:, None, None]
        v = self.d*x + self.e*time[:, None, None] + self.f*interval[:, None, None]
        return u, v


def make_wrapper(model=None, **kwargs):
    values = dict(
        model=AnalyticDual() if model is None else model,
        horizon=H, observation_dim=O, action_dim=A, n_timesteps=4,
        loss_type="l2", action_weight=1.0, loss_discount=1.0,
        loss_weights=None, obs_only=False, action_only=False,
        time_scale=1.0, predict_epsilon=False, clip_denoised=False,
    )
    values.update(kwargs)
    return AuxiliaryImprovedMeanFlow(**values)


def inputs(batch=2, dtype=torch.float32):
    data = torch.randn(batch, H, D, dtype=dtype)
    cond = {0: torch.randn(batch, O, dtype=dtype),
            H-1: torch.randn(batch, O, dtype=dtype)}
    return data, cond


def test_requires_explicit_auxiliary_model_interface():
    with pytest.raises(TypeError, match="forward_with_aux"):
        make_wrapper(nn.Linear(2, 2))


def test_exact_compound_and_two_losses_match_source_equations():
    model = AnalyticDual()
    wrapper = make_wrapper(model)
    data = torch.zeros(1, H, D)
    noise = torch.full_like(data, 0.5)
    r, t = torch.tensor([0.2]), torch.tensor([0.7])
    loss, info, details = wrapper._compute_meanflow_loss(
        data, {}, noise=noise, r=r, t=t, return_details=True)
    zt = 0.7 * noise
    tangent = model.d*zt + model.e*t[:, None, None]
    average = model.a*zt + model.b*t[:, None, None] + model.c*(t-r)[:, None, None]
    derivative = model.a*tangent + model.b + model.c
    auxiliary = model.d*zt + model.e*t[:, None, None] + model.f*(t-r)[:, None, None]
    compound = average + (t-r)[:, None, None] * derivative
    expected_u = (compound-noise).square().mean()
    expected_v = (auxiliary-noise).square().mean()
    torch.testing.assert_close(details["marginal_velocity"], tangent)
    torch.testing.assert_close(details["jvp"], derivative.expand_as(zt))
    torch.testing.assert_close(details["auxiliary_velocity"], auxiliary)
    torch.testing.assert_close(info["meanflow_u_loss"], expected_u)
    torch.testing.assert_close(info["meanflow_v_loss"], expected_v)
    torch.testing.assert_close(loss, expected_u + expected_v)


def test_finite_backward_reaches_both_heads_and_jvp_is_stopped():
    model = AnalyticDual()
    wrapper = make_wrapper(model, collect_diagnostics=True)
    data, cond = inputs(batch=4)
    loss, info, details = wrapper._compute_meanflow_loss(
        data, cond, return_details=True)
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in info.values())
    assert details["jvp"].requires_grad
    loss.backward()
    for name in ("a", "b", "c", "d", "e", "f"):
        gradient = getattr(model, name).grad
        assert gradient is not None and torch.isfinite(gradient) and gradient.abs() > 0


def test_adaptive_weighting_is_separate_and_summed():
    wrapper = make_wrapper(adaptive_weighting=True)
    data, cond = inputs(batch=4)
    loss, info = wrapper.loss(data, cond)
    torch.testing.assert_close(loss, info["meanflow_u_loss"] + info["meanflow_v_loss"])
    assert info["meanflow_u_loss"] < 1 and info["meanflow_v_loss"] < 1


def test_sampling_uses_only_u_head_and_preserves_conditions():
    model = AnalyticDual()
    wrapper = make_wrapper(model)
    _, cond = inputs(batch=1)
    sample = wrapper(cond, n_steps=4, verbose=False)
    assert model.forward_calls == 4 and model.aux_calls == 0
    assert sample.trajectories.shape == (1, H, D)
    assert torch.isfinite(sample.trajectories).all()
    for index, value in cond.items():
        torch.testing.assert_close(sample.trajectories[:, index, A:], value, rtol=0, atol=0)


def test_auxiliary_unet_preserves_u_path_and_branches_decoder():
    kwargs = dict(horizon=5, transition_dim=6, cond_dim=5,
                  dim=8, dim_mults=(1, 2), attention=False)
    torch.manual_seed(11)
    canonical = IntervalTemporalUnet(**kwargs)
    torch.manual_seed(11)
    auxiliary = AuxiliaryIntervalTemporalUnet(**kwargs)
    x, time, interval = torch.randn(2, 5, 6), torch.rand(2), torch.rand(2)
    expected = canonical(x, {}, time, interval=interval)
    average, velocity = auxiliary.forward_with_aux(x, {}, time, interval=interval)
    torch.testing.assert_close(auxiliary(x, {}, time, interval=interval), expected)
    torch.testing.assert_close(average, expected)
    assert not torch.equal(velocity, average)
    base_n = sum(p.numel() for p in canonical.parameters())
    aux_n = sum(p.numel() for p in auxiliary.parameters())
    assert base_n < aux_n < 2*base_n


def test_checkpoint_roundtrip_and_float64(tmp_path):
    wrapper = make_wrapper().double()
    data, cond = inputs(dtype=torch.float64)
    loss, _ = wrapper.loss(data, cond)
    loss.backward()
    path = tmp_path / "state.pt"
    torch.save({"model": wrapper.state_dict(), "step": 9}, path)
    restored = make_wrapper().double()
    state = torch.load(path)
    restored.load_state_dict(state["model"])
    actual, _ = restored.loss(data, cond)
    assert actual.dtype == torch.float64 and torch.isfinite(actual)
    assert state["step"] == 9
