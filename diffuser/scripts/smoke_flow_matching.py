#!/usr/bin/env python
"""Mac-safe ConditionalFlowMatching smoke test with the real PINT denoiser."""

import random

import numpy as np
import torch

from diffuser.device import get_available_device
from diffuser.models import AdaLNPINTDenoiser, ConditionalFlowMatching


def main():
    """Run loss, backward, Euler sampling, and invariance assertions."""
    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_available_device()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = mps_backend is not None and mps_backend.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    print("Selected device: {}".format(device))
    print("MPS available: {}".format(mps_available))

    batch_size = 2
    horizon = 4
    action_dim = 2
    features_dim = 3
    particles = 4
    observation_dim = features_dim * particles
    transition_dim = action_dim + observation_dim
    model = AdaLNPINTDenoiser(
        features_dim=features_dim,
        action_dim=action_dim,
        hidden_dim=32,
        projection_dim=32,
        n_head=4,
        n_layer=1,
        block_size=horizon,
        dropout=0.0,
        positional_bias=False,
        max_particles=None,
        multiview=False,
    ).to(device)
    flow = ConditionalFlowMatching(
        model=model,
        horizon=horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=4,
        loss_type="l2",
        action_weight=2.0,
        loss_discount=1.0,
        time_scale=1000.0,
    ).to(device)

    trajectory = torch.randn(batch_size, horizon, transition_dim, device=device)
    conditions = {
        0: torch.randn(batch_size, observation_dim, device=device),
        horizon - 1: torch.randn(batch_size, observation_dim, device=device),
    }
    trajectory_before = trajectory.clone()
    conditions_before = {key: value.clone() for key, value in conditions.items()}

    forward_calls = [0]

    def count_forward(module, inputs, output):
        forward_calls[0] += 1

    hook = model.register_forward_hook(count_forward)
    loss, info = flow.loss(trajectory, conditions)
    assert loss.ndim == 0 and torch.isfinite(loss), "loss must be finite and scalar"
    assert torch.isfinite(info["flow_loss"]), "reported flow loss must be finite"
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients, "real denoiser produced no gradients"
    assert all(torch.isfinite(gradient).all() for gradient in gradients), "gradient is non-finite"
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), "all gradients are zero"

    forward_calls[0] = 0
    sample = flow.conditional_sample(
        conditions,
        n_steps=4,
        return_chain=True,
        verbose=False,
        sort_by_value=False,
    )
    hook.remove()
    assert forward_calls[0] == 4, "four Euler steps must make exactly four denoiser calls"
    assert sample.trajectories.shape == (batch_size, horizon, transition_dim)
    expected_device = device
    if device.type == "cuda" and device.index is None:
        expected_device = torch.device("cuda", torch.cuda.current_device())
    assert sample.trajectories.device == expected_device
    assert sample.trajectories.dtype == trajectory.dtype
    assert torch.isfinite(sample.trajectories).all(), "sample contains non-finite values"
    assert sample.chains.shape == (batch_size, 5, horizon, transition_dim)
    for timestep, value in conditions.items():
        torch.testing.assert_close(
            sample.trajectories[:, timestep, action_dim:], value, rtol=0, atol=0
        )
        expected_chain = value[:, None, :].expand(-1, sample.chains.shape[1], -1)
        torch.testing.assert_close(
            sample.chains[:, :, timestep, action_dim:], expected_chain, rtol=0, atol=0
        )
    torch.testing.assert_close(sample.chains[:, -1], sample.trajectories)
    torch.testing.assert_close(trajectory, trajectory_before, rtol=0, atol=0)
    for key in conditions:
        torch.testing.assert_close(conditions[key], conditions_before[key], rtol=0, atol=0)

    gradient_norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients)).item()
    print("Model dimensions: horizon={}, action={}, observation={}, hidden=32, projection=32".format(
        horizon, action_dim, observation_dim
    ))
    print("Loss: {:.6f}".format(loss.item()))
    print("Gradient norm: {:.6f} (finite, nonzero)".format(gradient_norm))
    print("Sample shape: {}".format(tuple(sample.trajectories.shape)))
    print("Euler evaluations: {} | chain length: {}".format(forward_calls[0], sample.chains.shape[1]))
    print("Conditioning: exact | sample finiteness: PASS")
    print("REAL DENOISER SMOKE: PASS")


if __name__ == "__main__":
    main()
