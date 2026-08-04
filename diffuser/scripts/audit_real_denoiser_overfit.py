#!/usr/bin/env python
"""Short deterministic learnability audit using AdaLNPINTDenoiser itself."""

import torch

from diffuser.device import get_available_device
from diffuser.models import AdaLNPINTDenoiser, ConditionalFlowMatching


def main():
    """Require a predeclared 25% fixed-validation loss reduction in 100 steps."""
    seed = 20260804
    required_reduction = 25.0
    optimization_steps = 100
    torch.manual_seed(seed)
    device = get_available_device()

    batch_size = 4
    horizon = 3
    action_dim = 1
    features_dim = 2
    particles = 2
    observation_dim = features_dim * particles
    transition_dim = action_dim + observation_dim
    model = AdaLNPINTDenoiser(
        features_dim=features_dim,
        action_dim=action_dim,
        hidden_dim=16,
        projection_dim=16,
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
        n_solver_steps=3,
        loss_type="l2",
        time_scale=1000.0,
    ).to(device)

    clean = torch.linspace(
        -0.7, 0.7, horizon * transition_dim, device=device
    ).view(1, horizon, transition_dim).repeat(batch_size, 1, 1)
    conditions = {
        0: clean[:, 0, action_dim:].clone(),
        horizon - 1: clean[:, -1, action_dim:].clone(),
    }
    torch.manual_seed(seed + 1)
    validation_x0 = torch.randn_like(clean)
    validation_t = torch.linspace(0.15, 0.85, batch_size, device=device)

    def validation_loss():
        model.eval()
        with torch.no_grad():
            value, _ = flow._compute_flow_loss(
                clean, conditions, x0=validation_x0, t=validation_t
            )
        model.train()
        return value.item()

    initial_loss = validation_loss()
    best_loss = initial_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1, optimization_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = flow.loss(clean, conditions)
        assert torch.isfinite(loss), "non-finite loss at step {}".format(step)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if step % 20 == 0:
            current = validation_loss()
            best_loss = min(best_loss, current)
            print("step {:3d} | deterministic validation {:.8f}".format(step, current))

    final_loss = validation_loss()
    best_loss = min(best_loss, final_loss)
    reduction = 100.0 * (initial_loss - final_loss) / initial_loss
    sample = flow(conditions, n_steps=3, return_chain=True, verbose=False)
    assert torch.isfinite(sample.trajectories).all()
    for timestep, value in conditions.items():
        torch.testing.assert_close(sample.trajectories[:, timestep, action_dim:], value, rtol=0, atol=0)
        expected = value[:, None, :].expand(-1, sample.chains.shape[1], -1)
        torch.testing.assert_close(sample.chains[:, :, timestep, action_dim:], expected, rtol=0, atol=0)
    if reduction < required_reduction:
        raise AssertionError(
            "real denoiser reduction {:.2f}% is below predeclared {:.1f}% threshold".format(
                reduction, required_reduction
            )
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Device: {} | parameters: {} | optimization steps: {}".format(
        device, parameter_count, optimization_steps
    ))
    print("Initial deterministic loss: {:.8f}".format(initial_loss))
    print("Final deterministic loss: {:.8f}".format(final_loss))
    print("Best deterministic loss: {:.8f}".format(best_loss))
    print("Reduction: {:.2f}% | required: {:.1f}%".format(reduction, required_reduction))
    print("Conditioning after sampling: exact")
    print("REAL DENOISER OPTIMIZATION AUDIT: PASS")


if __name__ == "__main__":
    main()
