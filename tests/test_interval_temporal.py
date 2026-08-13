import pytest
import torch

from diffuser.models import (
    ConditionalFlowMatching,
    GaussianDiffusion,
    ImprovedMeanFlow,
    IntervalTemporalUnet,
    ShortcutModel,
    TemporalUnet,
)


def _model():
    return IntervalTemporalUnet(
        horizon=5,
        transition_dim=7,
        cond_dim=4,
        dim=8,
        dim_mults=(1, 2),
        attention=False,
    )


def _data(batch_size=3):
    trajectory = torch.randn(batch_size, 5, 7)
    conditions = {
        0: trajectory[:, 0, 3:].clone(),
        4: trajectory[:, 4, 3:].clone(),
    }
    return trajectory, conditions


def _wrapper(wrapper_class, model):
    kwargs = dict(
        model=model,
        horizon=5,
        observation_dim=4,
        action_dim=3,
        n_solver_steps=4,
        loss_type="l1",
        time_scale=1000.0,
    )
    if wrapper_class is ShortcutModel:
        kwargs["max_base_steps"] = 8
    return wrapper_class(**kwargs)


def test_interval_temporal_unet_zero_interval_is_default():
    torch.manual_seed(0)
    model = _model().eval()
    trajectory, conditions = _data()
    time = torch.tensor([0.0, 500.0, 1000.0])

    default = model(trajectory, conditions, time)
    explicit = model(trajectory, conditions, time, interval=torch.zeros_like(time))

    torch.testing.assert_close(default, explicit, rtol=0, atol=0)
    assert default.shape == trajectory.shape
    assert torch.isfinite(default).all()


def test_standard_temporal_unet_rejects_interval():
    model = TemporalUnet(
        horizon=5, transition_dim=7, cond_dim=4,
        dim=8, dim_mults=(1, 2), attention=False,
    )
    trajectory, conditions = _data()
    with pytest.raises(ValueError, match="interval"):
        model(
            trajectory,
            conditions,
            torch.ones(3),
            interval=torch.ones(3),
        )


@pytest.mark.parametrize(
    "wrapper_class",
    [ConditionalFlowMatching, ImprovedMeanFlow, ShortcutModel],
)
def test_flat_backbone_loss_backward_is_finite(wrapper_class):
    torch.manual_seed(1)
    model = _model()
    method = _wrapper(wrapper_class, model)
    trajectory, conditions = _data()

    loss, _ = method.loss(trajectory, conditions)
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in method.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_gaussian_flat_backbone_loss_backward_is_finite():
    torch.manual_seed(3)
    model = _model()
    method = GaussianDiffusion(
        model=model,
        horizon=5,
        observation_dim=4,
        action_dim=3,
        n_timesteps=8,
        loss_type="l1",
    )
    trajectory, conditions = _data()
    loss, _ = method.loss(trajectory, conditions)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in method.parameters()
    )


def test_interval_embedding_receives_gradient_and_changes_output():
    torch.manual_seed(2)
    model = _model().eval()
    trajectory, conditions = _data()
    time = torch.full((3,), 500.0)
    zero = model(trajectory, conditions, time, interval=torch.zeros_like(time))
    nonzero = model(trajectory, conditions, time, interval=torch.full_like(time, 250.0))
    nonzero.square().mean().backward()

    assert not torch.equal(zero, nonzero)
    gradients = [parameter.grad for parameter in model.interval_mlp.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_all_methods_use_identical_backbone_parameter_count():
    counts = []
    gaussian = GaussianDiffusion(
        model=_model(), horizon=5, observation_dim=4, action_dim=3, n_timesteps=8
    )
    counts.append(sum(parameter.numel() for parameter in gaussian.model.parameters()))
    for wrapper_class in (ConditionalFlowMatching, ImprovedMeanFlow, ShortcutModel):
        method = _wrapper(wrapper_class, _model())
        counts.append(sum(parameter.numel() for parameter in method.model.parameters()))
    assert len(set(counts)) == 1
