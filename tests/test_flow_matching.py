"""CPU-first coverage for conditional trajectory flow matching."""

import copy

import pytest
import torch
from torch import nn

from diffuser.device import get_available_device
from diffuser.models import AdaLNPINTDenoiser, ConditionalFlowMatching, GaussianDiffusion


HORIZON = 4
ACTION_DIM = 2
OBSERVATION_DIM = 6
TRANSITION_DIM = ACTION_DIM + OBSERVATION_DIM


class ToyVelocityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.forward_calls = 0
        self.received_times = []

    def forward(self, x, cond, time):
        self.forward_calls += 1
        self.received_times.append(time.detach().clone())
        return self.scale * x + self.bias


class ChannelErrorModel(nn.Module):
    def __init__(self, action_value=0.0, observation_value=0.0, active_timestep=None):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.action_value = action_value
        self.observation_value = observation_value
        self.active_timestep = active_timestep

    def forward(self, x, cond, time):
        output = torch.zeros_like(x) + self.anchor * 0.0
        if self.active_timestep is None:
            output[:, :, :ACTION_DIM] = self.action_value
            output[:, :, ACTION_DIM:] = self.observation_value
        else:
            output[:, self.active_timestep, :ACTION_DIM] = self.action_value
            output[:, self.active_timestep, ACTION_DIM:] = self.observation_value
        return output


class ConditionOnlyErrorModel(nn.Module):
    def __init__(self, error):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.error = error

    def forward(self, x, cond, time):
        output = torch.zeros_like(x) + self.anchor * 0.0
        for timestep in cond:
            output[:, timestep, ACTION_DIM:] = self.error
        return output


class WrongShapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, cond, time):
        return x[:, :, :-1]


class ParameterlessZeroModel(nn.Module):
    def forward(self, x, cond, time):
        return torch.zeros_like(x)


def make_flow(model=None, **kwargs):
    arguments = {
        "model": ToyVelocityModel() if model is None else model,
        "horizon": HORIZON,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "n_timesteps": 4,
        "loss_type": "l2",
        "clip_denoised": False,
        "predict_epsilon": False,
        "action_weight": 1.0,
        "loss_discount": 1.0,
        "loss_weights": None,
        "obs_only": False,
        "action_only": False,
    }
    arguments.update(kwargs)
    return ConditionalFlowMatching(**arguments)


def make_inputs(device="cpu", dtype=torch.float32, batch_size=3):
    x = torch.randn(batch_size, HORIZON, TRANSITION_DIM, device=device, dtype=dtype)
    cond = {
        0: torch.randn(batch_size, OBSERVATION_DIM, device=device, dtype=dtype),
        HORIZON - 1: torch.randn(batch_size, OBSERVATION_DIM, device=device, dtype=dtype),
    }
    return x, cond


def deterministic_loss(flow, model=None, cond=None):
    x1 = torch.zeros(2, HORIZON, TRANSITION_DIM)
    x0 = torch.zeros_like(x1)
    time = torch.tensor([0.25, 0.75])
    conditions = {} if cond is None else cond
    return flow._compute_flow_loss(x1, conditions, x0=x0, t=time, return_details=True)


def assert_conditioned(tensor, cond):
    for timestep, value in cond.items():
        torch.testing.assert_close(tensor[:, timestep, ACTION_DIM:], value, rtol=0, atol=0)


def test_public_imports_and_automatic_device_helper():
    assert ConditionalFlowMatching.__name__ == "ConditionalFlowMatching"
    assert GaussianDiffusion.__name__ == "GaussianDiffusion"
    assert AdaLNPINTDenoiser.__name__ == "AdaLNPINTDenoiser"
    assert get_available_device().type in ("cuda", "mps", "cpu")


def test_constructor_compatibility_and_attributes():
    flow = make_flow(n_solver_steps=8, n_diffusion_steps=3, n_timesteps=2, time_scale=500.0)
    assert flow.transition_dim == TRANSITION_DIM
    assert flow.n_timesteps == flow.n_solver_steps == flow.n_diffusion_steps == 8
    assert flow.time_scale == 500.0
    for name in ("model", "horizon", "observation_dim", "action_dim", "loss_type", "loss_weights", "device"):
        assert hasattr(flow, name)


@pytest.mark.parametrize("loss_type", ["l1", "l2"])
def test_finite_scalar_loss_and_detached_metrics(loss_type):
    flow = make_flow(loss_type=loss_type)
    x, cond = make_inputs()
    loss, info = flow.loss(x, cond)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert info["flow_loss"].ndim == 0
    assert all(torch.isfinite(value) and not value.requires_grad for value in info.values())


def test_backpropagation_has_finite_parameter_gradients():
    flow = make_flow()
    x, cond = make_inputs()
    loss, _ = flow.loss(x, cond)
    loss.backward()
    gradients = [parameter.grad for parameter in flow.model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_sample_shape_dtype_device_finiteness_and_default_chain():
    flow = make_flow()
    _, cond = make_inputs(dtype=torch.float64)
    flow = flow.double()
    sample = flow.conditional_sample(cond, verbose=False)
    assert sample.trajectories.shape == (3, HORIZON, TRANSITION_DIM)
    assert sample.values.shape == (3,)
    assert sample.trajectories.dtype == torch.float64
    assert sample.trajectories.device.type == "cpu"
    assert torch.isfinite(sample.trajectories).all()
    assert sample.chains is None


def test_initial_goal_and_multiple_conditions_are_exact():
    flow = make_flow()
    _, cond = make_inputs()
    cond[2] = torch.randn_like(cond[0])
    sample = flow(cond, verbose=False)
    assert_conditioned(sample.trajectories, cond)


def test_only_observation_channels_are_conditioned():
    flow = make_flow(model=ParameterlessZeroModel())
    _, cond = make_inputs()
    torch.manual_seed(91)
    sample = flow(cond, n_steps=1, verbose=False)
    torch.manual_seed(91)
    expected_noise = torch.randn_like(sample.trajectories)
    torch.testing.assert_close(sample.trajectories[:, :, :ACTION_DIM], expected_noise[:, :, :ACTION_DIM])
    assert_conditioned(sample.trajectories, cond)


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 16])
def test_solver_steps_equal_model_forward_calls(steps):
    model = ToyVelocityModel()
    flow = make_flow(model=model)
    _, cond = make_inputs(batch_size=1)
    flow(cond, n_steps=steps, verbose=False)
    assert model.forward_calls == steps


@pytest.mark.parametrize("steps", [1, 4, 8])
def test_chain_shape_conditioning_and_endpoints(steps):
    flow = make_flow(model=ParameterlessZeroModel())
    _, cond = make_inputs(batch_size=2)
    torch.manual_seed(4)
    sample = flow(cond, n_steps=steps, return_chain=True, verbose=False)
    assert sample.chains.shape == (2, steps + 1, HORIZON, TRANSITION_DIM)
    assert_conditioned(sample.chains.reshape(-1, HORIZON, TRANSITION_DIM), {
        timestep: value.repeat_interleave(steps + 1, dim=0) for timestep, value in cond.items()
    })
    torch.testing.assert_close(sample.chains[:, -1], sample.trajectories)
    torch.manual_seed(4)
    expected_initial = torch.randn_like(sample.trajectories)
    for timestep, value in cond.items():
        expected_initial[:, timestep, ACTION_DIM:] = value
    torch.testing.assert_close(sample.chains[:, 0], expected_initial)


def test_return_diffusion_alias_enables_chain():
    flow = make_flow()
    _, cond = make_inputs(batch_size=1)
    assert flow(cond, return_diffusion=True, n_steps=2, verbose=False).chains.shape[1] == 3


def test_sampling_is_seed_deterministic_and_different_seeds_differ():
    flow = make_flow()
    _, cond = make_inputs(batch_size=1)
    torch.manual_seed(12)
    first = flow(cond, verbose=False).trajectories
    torch.manual_seed(12)
    second = flow(cond, verbose=False).trajectories
    torch.manual_seed(13)
    third = flow(cond, verbose=False).trajectories
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, third)


def test_loss_does_not_mutate_trajectory_or_conditioning():
    flow = make_flow()
    x, cond = make_inputs()
    x_before = x.clone()
    cond_before = {key: value.clone() for key, value in cond.items()}
    keys_before = list(cond)
    flow.loss(x, cond)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    assert list(cond) == keys_before
    for key in cond:
        torch.testing.assert_close(cond[key], cond_before[key], rtol=0, atol=0)


@pytest.mark.parametrize("steps,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError), (False, TypeError)])
def test_invalid_solver_steps_raise(steps, error):
    flow = make_flow()
    _, cond = make_inputs(batch_size=1)
    with pytest.raises(error, match="n_steps"):
        flow(cond, n_steps=steps, verbose=False)


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"loss_type": "huber"}, ValueError, "loss_type"),
        ({"obs_only": True, "action_only": True}, ValueError, "cannot both"),
        ({"time_scale": 0.0}, ValueError, "time_scale"),
        ({"time_scale": float("nan")}, ValueError, "time_scale"),
        ({"n_solver_steps": True}, TypeError, "n_solver_steps"),
    ],
)
def test_invalid_constructor_arguments_raise(kwargs, error, match):
    with pytest.raises(error, match=match):
        make_flow(**kwargs)


def test_invalid_condition_timestep_shape_batch_dtype_and_device():
    flow = make_flow()
    x, cond = make_inputs()
    with pytest.raises(ValueError, match="outside active horizon"):
        flow.loss(x, {-1: cond[0]})
    with pytest.raises(ValueError, match="must have shape"):
        flow.loss(x, {0: cond[0][:, :-1]})
    with pytest.raises(ValueError, match="must have shape"):
        flow.loss(x, {0: cond[0][:-1]})
    with pytest.raises(ValueError, match="dtype"):
        flow.loss(x, {0: cond[0].double()})


def test_model_output_shape_mismatch_raises():
    flow = make_flow(model=WrongShapeModel())
    x, cond = make_inputs()
    with pytest.raises(ValueError, match="output shape"):
        flow.loss(x, cond)


def test_conditioned_prediction_errors_are_masked_but_unconditioned_errors_count():
    cond = {0: torch.zeros(2, OBSERVATION_DIM)}
    baseline = make_flow(model=ConditionOnlyErrorModel(0.0))
    conditioned_error = make_flow(model=ConditionOnlyErrorModel(100.0))
    unconditioned_error = make_flow(model=ChannelErrorModel(observation_value=1.0))
    base_loss = deterministic_loss(baseline, cond=cond)[0]
    fixed_loss = deterministic_loss(conditioned_error, cond=cond)[0]
    free_loss = deterministic_loss(unconditioned_error, cond=cond)[0]
    torch.testing.assert_close(base_loss, fixed_loss)
    assert free_loss > fixed_loss


def test_conditioned_target_velocity_is_zero_and_interpolation_is_exact():
    flow = make_flow()
    x1, cond = make_inputs(batch_size=2)
    x0 = torch.randn_like(x1)
    time = torch.tensor([0.2, 0.8])
    _, _, details = flow._compute_flow_loss(x1, cond, x0=x0, t=time, return_details=True)
    for timestep, value in cond.items():
        torch.testing.assert_close(details["x0"][:, timestep, ACTION_DIM:], value, rtol=0, atol=0)
        torch.testing.assert_close(details["x1"][:, timestep, ACTION_DIM:], value, rtol=0, atol=0)
        torch.testing.assert_close(details["xt"][:, timestep, ACTION_DIM:], value, rtol=0, atol=0)
        assert torch.count_nonzero(details["target_velocity"][:, timestep, ACTION_DIM:]) == 0


def test_obs_only_and_action_only_channel_selection():
    obs_only = make_flow(model=ChannelErrorModel(action_value=100.0, observation_value=1.0), obs_only=True)
    obs_reference = make_flow(model=ChannelErrorModel(action_value=0.0, observation_value=1.0), obs_only=True)
    action_only = make_flow(model=ChannelErrorModel(action_value=1.0, observation_value=100.0), action_only=True)
    action_reference = make_flow(model=ChannelErrorModel(action_value=1.0, observation_value=0.0), action_only=True)
    torch.testing.assert_close(deterministic_loss(obs_only)[0], deterministic_loss(obs_reference)[0])
    torch.testing.assert_close(deterministic_loss(action_only)[0], deterministic_loss(action_reference)[0])


def test_action_weight_changes_first_action_contribution():
    model = ChannelErrorModel(action_value=1.0, active_timestep=0)
    low = make_flow(model=copy.deepcopy(model), action_weight=1.0)
    high = make_flow(model=copy.deepcopy(model), action_weight=10.0)
    assert deterministic_loss(high)[0] > deterministic_loss(low)[0]


def test_loss_discount_changes_late_timestep_contribution():
    model = ChannelErrorModel(observation_value=1.0, active_timestep=HORIZON - 1)
    flat = make_flow(model=copy.deepcopy(model), loss_discount=1.0)
    discounted = make_flow(model=copy.deepcopy(model), loss_discount=0.5)
    assert deterministic_loss(discounted)[0] < deterministic_loss(flat)[0]


def test_custom_observation_loss_weights_change_contribution():
    x1 = torch.zeros(2, HORIZON, TRANSITION_DIM)
    x0 = torch.zeros_like(x1)
    time = torch.tensor([0.25, 0.75])

    class FirstObservationError(ChannelErrorModel):
        def forward(self, x, cond, model_time):
            output = torch.zeros_like(x) + self.anchor * 0.0
            output[:, :, ACTION_DIM] = 1.0
            return output

    normal = make_flow(model=FirstObservationError())
    weighted = make_flow(model=FirstObservationError(), loss_weights={0: 20.0})
    normal_loss = normal._compute_flow_loss(x1, {}, x0=x0, t=time)[0]
    weighted_loss = weighted._compute_flow_loss(x1, {}, x0=x0, t=time)[0]
    assert weighted_loss > normal_loss


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_weight": 4.0},
        {"loss_discount": 0.7},
        {"loss_weights": {0: 3.0, 4: 0.25}},
        {"action_only": True},
    ],
)
def test_loss_weight_matrix_matches_gaussian_diffusion(kwargs):
    model = ToyVelocityModel()
    flow = make_flow(model=model, **kwargs)
    gaussian = GaussianDiffusion(
        model=ToyVelocityModel(),
        horizon=HORIZON,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        n_timesteps=4,
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
        action_weight=kwargs.get("action_weight", 1.0),
        loss_discount=kwargs.get("loss_discount", 1.0),
        loss_weights=kwargs.get("loss_weights"),
        action_only=kwargs.get("action_only", False),
    )
    torch.testing.assert_close(flow.loss_weight_matrix, gaussian.loss_fn.weights, rtol=0, atol=0)


def test_masked_weighted_reduction_applies_weights_once_and_counts_active_elements():
    conditions = {0: torch.zeros(2, OBSERVATION_DIM)}
    flow = make_flow(
        model=ChannelErrorModel(action_value=2.0, observation_value=3.0),
        action_weight=5.0,
        loss_discount=0.8,
        loss_weights={0: 4.0},
    )
    loss, _, details = deterministic_loss(flow, cond=conditions)
    squared_error = torch.empty(2, HORIZON, TRANSITION_DIM)
    squared_error[:, :, :ACTION_DIM] = 4.0
    squared_error[:, :, ACTION_DIM:] = 9.0
    weights = flow.loss_weight_matrix.unsqueeze(0)
    mask = details["conditioning_mask"]
    expected = (squared_error * weights * mask).sum() / mask.sum()
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("action_only", [False, True])
def test_unmasked_weighted_reduction_matches_gaussian_mean(action_only):
    model = ChannelErrorModel(action_value=2.0, observation_value=3.0)
    flow = make_flow(
        model=model,
        action_weight=5.0,
        loss_discount=0.8,
        loss_weights={0: 4.0},
        action_only=action_only,
    )
    gaussian = GaussianDiffusion(
        model=ChannelErrorModel(action_value=2.0, observation_value=3.0),
        horizon=HORIZON,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        n_timesteps=4,
        loss_type="l2",
        clip_denoised=True,
        action_weight=5.0,
        loss_discount=0.8,
        loss_weights={0: 4.0},
        action_only=action_only,
    )
    trajectory = torch.zeros(2, HORIZON, TRANSITION_DIM)
    time = torch.tensor([0.25, 0.75])
    flow_loss = flow._compute_flow_loss(
        trajectory, {}, x0=torch.zeros_like(trajectory), t=time
    )[0]
    prediction = model(trajectory, {}, time)
    gaussian_loss = gaussian.loss_fn(prediction, torch.zeros_like(prediction))[0]
    torch.testing.assert_close(flow_loss, gaussian_loss)


def test_obs_only_weights_only_observations_and_fully_masked_loss_raises():
    obs_only = make_flow(obs_only=True)
    assert torch.count_nonzero(obs_only.loss_weight_matrix[:, :ACTION_DIM]) == 0
    assert torch.all(obs_only.loss_weight_matrix[:, ACTION_DIM:] > 0)

    horizon = 2
    flow = ConditionalFlowMatching(
        model=ParameterlessZeroModel(),
        horizon=horizon,
        observation_dim=3,
        action_dim=0,
        n_timesteps=2,
        loss_type="l2",
    )
    trajectory = torch.zeros(1, horizon, 3)
    conditions = {0: torch.zeros(1, 3), 1: torch.zeros(1, 3)}
    with pytest.raises(ValueError, match="no active elements"):
        flow._compute_flow_loss(
            trajectory,
            conditions,
            x0=torch.zeros_like(trajectory),
            t=torch.tensor([0.5]),
        )


def test_training_and_sampling_times_are_floating_and_scaled_consistently():
    model = ToyVelocityModel()
    flow = make_flow(model=model, time_scale=250.0)
    x, cond = make_inputs(batch_size=2)
    fixed_time = torch.tensor([0.2, 0.8])
    flow._compute_flow_loss(x, cond, x0=torch.randn_like(x), t=fixed_time)
    received_training_time = model.received_times[-1]
    assert received_training_time.is_floating_point()
    torch.testing.assert_close(received_training_time, fixed_time * 250.0)
    model.received_times.clear()
    flow(cond, n_steps=4, verbose=False)
    received = torch.stack(model.received_times)
    torch.testing.assert_close(received[:, 0], torch.tensor([0.0, 62.5, 125.0, 187.5]))
    model.received_times.clear()
    flow(cond, n_steps=2, verbose=False)
    torch.testing.assert_close(torch.stack(model.received_times)[:, 0], torch.tensor([0.0, 125.0]))


def test_parameterless_model_device_fallback_and_cpu_smoke():
    flow = make_flow(model=ParameterlessZeroModel())
    x, cond = make_inputs(batch_size=1)
    assert flow.device.type == "cpu"
    loss, _ = flow.loss(x, cond)
    sample = flow(cond, verbose=False)
    assert torch.isfinite(loss) and torch.isfinite(sample.trajectories).all()


def test_real_denoiser_accepts_continuous_float_time():
    model = AdaLNPINTDenoiser(
        features_dim=3,
        action_dim=2,
        hidden_dim=16,
        projection_dim=16,
        n_head=4,
        n_layer=1,
        block_size=3,
        dropout=0.0,
        positional_bias=False,
        max_particles=None,
        multiview=False,
    )
    x = torch.randn(2, 3, 8)
    output = model(x, {}, torch.tensor([125.5, 750.25]))
    assert output.shape == x.shape and torch.isfinite(output).all()


@pytest.mark.skipif(
    getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available(),
    reason="MPS backend is unavailable on this host",
)
def test_mps_loss_backward_sampling_and_conditioning():
    device = torch.device("mps")
    flow = make_flow().to(device)
    x, cond = make_inputs(device=device, batch_size=2)
    loss, _ = flow.loss(x, cond)
    loss.backward()
    sample = flow(cond, n_steps=2, verbose=False)
    assert sample.trajectories.device.type == "mps"
    assert torch.isfinite(loss) and torch.isfinite(sample.trajectories).all()
    assert_conditioned(sample.trajectories, cond)
