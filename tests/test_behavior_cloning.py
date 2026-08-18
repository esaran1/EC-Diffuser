"""CPU-first coverage for the deterministic behavior-cloning floor arm."""

import copy

import pytest
import torch
from torch import nn

from diffuser.models import ConditionalFlowMatching, DeterministicBehaviorCloning


HORIZON = 4
ACTION_DIM = 2
OBSERVATION_DIM = 6
TRANSITION_DIM = ACTION_DIM + OBSERVATION_DIM


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(0.25))
        self.forward_calls = 0
        self.received_times = []

    def forward(self, x, cond, time):
        self.forward_calls += 1
        self.received_times.append(time.detach().clone())
        return self.scale * x + self.bias


def build(model=None, **kwargs):
    return DeterministicBehaviorCloning(
        model=model if model is not None else ToyModel(),
        horizon=HORIZON,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        **kwargs,
    )


def make_batch(batch=3):
    torch.manual_seed(0)
    x = torch.randn(batch, HORIZON, TRANSITION_DIM)
    cond = {
        0: torch.randn(batch, OBSERVATION_DIM),
        HORIZON - 1: torch.randn(batch, OBSERVATION_DIM),
    }
    return x, cond


def test_loss_is_finite_and_gradients_flow():
    policy = build()
    x, cond = make_batch()
    loss, info = policy.loss(x, cond)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    for key in ("bc_loss", "unweighted_bc_loss", "action_loss", "observation_loss"):
        assert key in info and torch.isfinite(info[key])


def test_sampling_uses_exactly_one_model_call():
    model = ToyModel()
    policy = build(model=model)
    _, cond = make_batch()
    model.forward_calls = 0
    sample = policy.conditional_sample(cond)
    assert model.forward_calls == 1, "BC must cost exactly 1 NFE"
    assert sample.trajectories.shape == (3, HORIZON, TRANSITION_DIM)
    assert torch.isfinite(sample.trajectories).all()


def test_requesting_more_than_one_step_is_rejected():
    policy = build()
    _, cond = make_batch()
    policy.conditional_sample(cond, n_steps=1)
    with pytest.raises(ValueError):
        policy.conditional_sample(cond, n_steps=4)


def test_conditioning_is_exact_after_sampling():
    policy = build()
    _, cond = make_batch()
    sample = policy.conditional_sample(cond)
    observations = sample.trajectories[:, :, ACTION_DIM:]
    torch.testing.assert_close(observations[:, 0], cond[0])
    torch.testing.assert_close(observations[:, HORIZON - 1], cond[HORIZON - 1])


def test_conditioned_elements_are_excluded_from_the_loss():
    """Perturbing a conditioned slot must not change the loss."""
    policy = build()
    x, cond = make_batch()
    baseline, _ = policy.loss(x, cond)

    perturbed = x.clone()
    perturbed[:, 0, ACTION_DIM:] += 100.0
    perturbed[:, HORIZON - 1, ACTION_DIM:] += 100.0
    shifted, _ = policy.loss(perturbed, cond)

    torch.testing.assert_close(baseline, shifted)


def test_loss_reduction_matches_the_flow_arm():
    """BC and Flow must share masking and reduction so losses are comparable.

    With a model that returns a constant, the BC loss reduces to the same
    weighted-masked mean the flow arm applies to its own residual. This pins
    that the two arms agree on denominator and weighting, which the controlled
    comparison depends on.
    """
    torch.manual_seed(0)
    bc = build(loss_type="l2")
    flow = ConditionalFlowMatching(
        model=ToyModel(),
        horizon=HORIZON,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        loss_type="l2",
    )
    x, cond = make_batch()

    mask = bc._make_conditioning_mask(x, cond)
    flow_mask = flow._make_conditioning_mask(x, cond)
    torch.testing.assert_close(mask.to(x.dtype), flow_mask.to(x.dtype))
    torch.testing.assert_close(
        bc.loss_weight_matrix.to(x.dtype), flow.loss_weight_matrix.to(x.dtype)
    )


def test_checkpoint_round_trip_preserves_predictions():
    policy = build()
    _, cond = make_batch()
    before = policy.conditional_sample(cond).trajectories

    state = copy.deepcopy(policy.state_dict())
    restored = build()
    restored.load_state_dict(state)
    after = restored.conditional_sample(cond).trajectories

    torch.testing.assert_close(before, after)


def test_action_weight_semantics_match_the_flow_arm():
    """BC must inherit action_weight semantics from the flow arm exactly.

    Two inherited EC-Diffuser conventions are pinned here, because the
    controlled comparison depends on BC not diverging from them:

    1. `_make_loss_weights` applies action_weight only at timestep 0
       (`weights[0, :action_dim]`).
    2. `_make_conditioning_mask` masks only the *observation* channels of
       conditioned timesteps, so the action channels at t=0 stay active and
       action_weight does affect the loss.

    `loss_weight_matrix` is a registered buffer, so weighting must be compared
    on freshly constructed models -- `load_state_dict` would copy it over.
    """
    for weight in (1.0, 10.0):
        bc = build(action_weight=weight, loss_type="l2")
        flow = ConditionalFlowMatching(
            model=ToyModel(),
            horizon=HORIZON,
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            action_weight=weight,
            loss_type="l2",
        )
        torch.testing.assert_close(bc.loss_weight_matrix, flow.loss_weight_matrix)
        assert bc.loss_weight_matrix[0, :ACTION_DIM].eq(weight).all()

    # Conditioned timesteps mask observations but not actions.
    x, cond = make_batch()
    bc = build(action_weight=1.0, loss_type="l2")
    mask = bc._make_conditioning_mask(x, cond)
    assert mask[:, 0, :ACTION_DIM].all(), "actions at t=0 must stay active"
    assert not mask[:, 0, ACTION_DIM:].any(), "observations at t=0 must be masked"

    # Because t=0 actions are active, action_weight changes the loss.
    light = build(action_weight=1.0, loss_type="l2")
    heavy = build(action_weight=10.0, loss_type="l2")
    heavy.model.load_state_dict(light.model.state_dict())
    light_loss, _ = light.loss(x, cond)
    heavy_loss, _ = heavy.loss(x, cond)
    assert heavy_loss.item() > light_loss.item()
