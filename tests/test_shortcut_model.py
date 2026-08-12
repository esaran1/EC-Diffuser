"""Mathematical and integration tests for step-size-conditioned Shortcut Models."""

import pytest
import torch
from torch import nn

from diffuser.models import IntervalAdaLNPINTDenoiser, ShortcutModel


H, A, O = 3, 1, 4
D = A + O


class AnalyticShortcut(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.calls = 0

    def forward(self, x, cond, time, interval):
        self.calls += 1
        return self.scale * x + time[:, None, None] + 2.0 * interval[:, None, None]


def wrapper(model=None, **kwargs):
    values = dict(
        model=AnalyticShortcut() if model is None else model,
        horizon=H, observation_dim=O, action_dim=A, n_timesteps=4,
        loss_type="l2", action_weight=1.0, loss_discount=1.0,
        time_scale=1.0, predict_epsilon=False, clip_denoised=False,
        max_base_steps=128, flow_fraction=0.75,
    )
    values.update(kwargs)
    return ShortcutModel(**values)


def inputs(batch=2, dtype=torch.float32):
    data = torch.randn(batch, H, D, dtype=dtype)
    cond = {0: torch.randn(batch, O, dtype=dtype),
            H - 1: torch.randn(batch, O, dtype=dtype)}
    return data, cond


def assert_conditioned(x, cond):
    for index, value in cond.items():
        torch.testing.assert_close(x[:, index, A:], value, rtol=0, atol=0)


def test_import_finite_loss_backward_and_gradients():
    model = AnalyticShortcut()
    method = wrapper(model)
    data, cond = inputs(batch=4)
    t = torch.tensor([0.0, 0.25, 0.0, 0.5])
    d = torch.tensor([0.25, 0.125, 0.25, 0.125])
    flow = torch.tensor([True, False, True, False])
    loss, info = method._compute_shortcut_loss(
        data, cond, t=t, small_d=d, flow_mask=flow
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in info.values())
    loss.backward()
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)


def test_known_two_half_step_bootstrap_target():
    model = AnalyticShortcut()
    method = wrapper(model)
    data = torch.full((1, H, D), 0.5)
    noise = torch.full_like(data, -0.5)
    t = torch.tensor([0.25])
    d = torch.tensor([0.25])
    flow = torch.tensor([False])
    _, _, details = method._compute_shortcut_loss(
        data, {}, noise=noise, t=t, small_d=d,
        flow_mask=flow, return_details=True,
    )
    xt = (1.0 - t[:, None, None]) * noise + t[:, None, None] * data
    first = 0.2 * xt + t[:, None, None] + 2.0 * d[:, None, None]
    midpoint = xt + d[:, None, None] * first
    second = 0.2 * midpoint + (t+d)[:, None, None] + 2.0*d[:, None, None]
    expected = 0.5 * (first + second)
    torch.testing.assert_close(details["target"], expected)
    assert not details["target"].requires_grad
    torch.testing.assert_close(details["requested_d"], 2.0*d)


def test_flow_rows_use_empirical_velocity_and_zero_step_condition():
    method = wrapper()
    data = torch.full((1, H, D), 0.75)
    noise = torch.full_like(data, -0.25)
    _, _, details = method._compute_shortcut_loss(
        data, {}, noise=noise, t=torch.tensor([0.4]),
        small_d=torch.tensor([0.125]), flow_mask=torch.tensor([True]),
        return_details=True,
    )
    torch.testing.assert_close(details["target"], data-noise)
    torch.testing.assert_close(details["requested_d"], torch.zeros(1))


def test_loss_does_not_mutate_inputs_and_masks_conditions():
    method = wrapper()
    data, cond = inputs(batch=4)
    before = data.clone()
    cond_before = {key: value.clone() for key, value in cond.items()}
    _, _, details = method._compute_shortcut_loss(
        data, cond, return_details=True
    )
    torch.testing.assert_close(data, before)
    for key in cond:
        torch.testing.assert_close(cond[key], cond_before[key])
        assert not details["conditioning_mask"][:, key, A:].any()


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 16, 128])
def test_sampling_budget_equals_calls_and_preserves_conditions(steps):
    model = AnalyticShortcut()
    method = wrapper(model)
    _, cond = inputs(batch=1)
    before = model.calls
    sample = method(cond, n_steps=steps, verbose=False)
    assert model.calls-before == steps
    assert sample.trajectories.shape == (1, H, D)
    assert torch.isfinite(sample.trajectories).all()
    assert_conditioned(sample.trajectories, cond)


@pytest.mark.parametrize("steps", [3, 129])
def test_unsupported_sampling_budget_rejected(steps):
    method = wrapper()
    _, cond = inputs(batch=1)
    with pytest.raises(ValueError, match="power of two"):
        method(cond, n_steps=steps, verbose=False)


def test_dtype_chain_and_checkpoint_roundtrip(tmp_path):
    method = wrapper().double()
    _, cond = inputs(batch=1, dtype=torch.float64)
    torch.manual_seed(17)
    expected = method(cond, n_steps=4, return_chain=True, verbose=False)
    path = tmp_path / "shortcut.pt"
    torch.save({"model": method.state_dict(), "ema": method.state_dict(), "step": 11}, path)
    restored = wrapper().double()
    state = torch.load(path)
    restored.load_state_dict(state["model"])
    torch.manual_seed(17)
    actual = restored(cond, n_steps=4, return_chain=True, verbose=False)
    torch.testing.assert_close(actual.trajectories, expected.trajectories)
    assert actual.chains.shape == (1, 5, H, D)


def test_real_interval_pint_backward_and_sample():
    model = IntervalAdaLNPINTDenoiser(
        features_dim=2, action_dim=A, hidden_dim=16, projection_dim=16,
        n_head=2, n_layer=1, block_size=H, dropout=0.0,
        positional_bias=False, max_particles=None, multiview=False,
    )
    method = wrapper(model, time_scale=1000.0)
    data, cond = inputs(batch=4)
    loss, _ = method.loss(data, cond)
    loss.backward()
    gradients = [p.grad for p in method.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    sample = method(cond, n_steps=1, verbose=False)
    assert sample.trajectories.shape == data.shape
    assert torch.isfinite(sample.trajectories).all()
    assert_conditioned(sample.trajectories, cond)
