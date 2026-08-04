#!/usr/bin/env python
"""Reproducible tiny learnability check for ConditionalFlowMatching."""

import random

import numpy as np
import torch
from torch import nn

from diffuser.device import get_available_device
from diffuser.models import ConditionalFlowMatching


class TinyVelocityModel(nn.Module):
    """Small pointwise MLP used to keep the optimization check Mac-friendly."""

    def __init__(self, transition_dim, time_scale):
        super().__init__()
        self.time_scale = time_scale
        self.network = nn.Sequential(
            nn.Linear(transition_dim + 1, 48),
            nn.SiLU(),
            nn.Linear(48, 48),
            nn.SiLU(),
            nn.Linear(48, transition_dim),
        )

    def forward(self, x, cond, time):
        normalized_time = (time / self.time_scale).view(-1, 1, 1)
        time_feature = normalized_time.expand(-1, x.shape[1], 1)
        return self.network(torch.cat([x, time_feature], dim=-1))


def finite_gradient_norm(model):
    """Return the global gradient norm, raising on missing or non-finite gradients."""
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise AssertionError("optimizer step produced no gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("optimizer step produced a non-finite gradient")
    return torch.sqrt(sum(gradient.square().sum() for gradient in gradients))


def main():
    """Fit a fixed straight-path validation problem and enforce a 50% reduction."""
    seed = 31415
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_available_device()
    print("Device: {}".format(device))

    batch_size = 8
    horizon = 4
    action_dim = 2
    observation_dim = 6
    transition_dim = action_dim + observation_dim
    time_scale = 1000.0
    training_steps = 300
    learning_rate = 3e-3
    model = TinyVelocityModel(transition_dim, time_scale).to(device)
    flow = ConditionalFlowMatching(
        model=model,
        horizon=horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_solver_steps=4,
        loss_type="l2",
        action_weight=1.0,
        loss_discount=1.0,
        time_scale=time_scale,
    ).to(device)

    clean_pattern = torch.linspace(-0.8, 0.8, horizon * transition_dim, device=device)
    clean = clean_pattern.view(1, horizon, transition_dim).repeat(batch_size, 1, 1)
    conditions = {
        0: clean[:, 0, action_dim:].clone(),
        horizon - 1: clean[:, -1, action_dim:].clone(),
    }
    torch.manual_seed(seed + 1)
    fixed_noise = torch.randn(batch_size, horizon, transition_dim, device=device)
    fixed_time = torch.linspace(0.1, 0.9, batch_size, device=device)

    def validation_loss():
        model.eval()
        with torch.no_grad():
            value, _ = flow._compute_flow_loss(clean, conditions, x0=fixed_noise, t=fixed_time)
        model.train()
        return value.item()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial_validation = validation_loss()
    best_validation = initial_validation
    recent_training = []
    gradient_norms = []
    for step in range(1, training_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = flow.loss(clean, conditions)
        if not torch.isfinite(loss):
            raise AssertionError("non-finite training loss at step {}".format(step))
        loss.backward()
        gradient_norm = finite_gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        recent_training.append(loss.item())
        recent_training = recent_training[-25:]
        gradient_norms.append(gradient_norm.item())
        if step % 50 == 0:
            current_validation = validation_loss()
            best_validation = min(best_validation, current_validation)
            print("step {:3d} | train_ma {:.6f} | validation {:.6f}".format(
                step, sum(recent_training) / len(recent_training), current_validation
            ))

    final_validation = validation_loss()
    best_validation = min(best_validation, final_validation)
    reduction = 100.0 * (initial_validation - final_validation) / initial_validation
    threshold = 50.0
    sample = flow(conditions, n_steps=4, return_chain=True, verbose=False)
    assert sample.trajectories.shape == (batch_size, horizon, transition_dim)
    assert torch.isfinite(sample.trajectories).all()
    for timestep, value in conditions.items():
        torch.testing.assert_close(sample.trajectories[:, timestep, action_dim:], value, rtol=0, atol=0)
        expected = value[:, None, :].expand(-1, sample.chains.shape[1], -1)
        torch.testing.assert_close(sample.chains[:, :, timestep, action_dim:], expected, rtol=0, atol=0)
    if reduction < threshold or final_validation >= initial_validation:
        raise AssertionError(
            "validation loss reduction {:.2f}% did not meet fixed {:.1f}% criterion".format(
                reduction, threshold
            )
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Model: TinyVelocityModel | parameters: {}".format(parameter_count))
    print("Optimization steps: {} | learning rate: {}".format(training_steps, learning_rate))
    print("Initial validation loss: {:.8f}".format(initial_validation))
    print("Final validation loss: {:.8f}".format(final_validation))
    print("Best validation loss: {:.8f}".format(best_validation))
    print("Reduction: {:.2f}% | required: {:.1f}%".format(reduction, threshold))
    print("Final training moving average: {:.8f}".format(sum(recent_training) / len(recent_training)))
    print("Gradient norm range: {:.6f} .. {:.6f}".format(min(gradient_norms), max(gradient_norms)))
    print("Final sample shape: {} | conditioning: exact".format(tuple(sample.trajectories.shape)))
    print("SYNTHETIC OVERFIT: PASS")


if __name__ == "__main__":
    main()
