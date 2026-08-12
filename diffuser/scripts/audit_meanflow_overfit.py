#!/usr/bin/env python
"""Deterministic synthetic and real-PINT learnability audits for iMF."""

import torch
from torch import nn

from diffuser.device import get_available_device
from diffuser.models import ImprovedMeanFlow, IntervalAdaLNPINTDenoiser


class TinyAverageModel(nn.Module):
    def __init__(self, transition_dim, time_scale):
        super().__init__()
        self.time_scale = time_scale
        self.net = nn.Sequential(
            nn.Linear(transition_dim + 2, 48), nn.SiLU(),
            nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, transition_dim),
        )

    def forward(self, x, cond, time, interval):
        t = (time / self.time_scale).view(-1, 1, 1).expand(-1, x.shape[1], 1)
        d = (interval / self.time_scale).view(-1, 1, 1).expand(-1, x.shape[1], 1)
        return self.net(torch.cat([x, t, d], dim=-1))


def run_audit(name, model, wrapper, clean, conditions, steps, threshold):
    fixed_noise = torch.randn_like(clean)
    r = torch.linspace(0.05, 0.35, clean.shape[0], device=clean.device)
    t = torch.linspace(0.55, 0.95, clean.shape[0], device=clean.device)

    def validation():
        model.eval()
        with torch.no_grad():
            value, _ = wrapper._compute_meanflow_loss(
                clean, conditions, noise=fixed_noise, r=r, t=t
            )
        model.train()
        return value.item()

    initial = validation()
    best = initial
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = wrapper._compute_meanflow_loss(
            clean, conditions, noise=fixed_noise, r=r, t=t
        )
        if not torch.isfinite(loss):
            raise AssertionError("non-finite loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError("missing or non-finite gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % 25 == 0:
            current = validation()
            best = min(best, current)
            print("{} step {:3d} validation {:.8f}".format(name, step, current))
    final = validation()
    reduction = 100.0 * (initial - final) / initial
    if reduction < threshold:
        raise AssertionError("{} reduction {:.2f}% < {:.1f}%".format(
            name, reduction, threshold
        ))
    sample = wrapper(conditions, n_steps=1, verbose=False)
    if not torch.isfinite(sample.trajectories).all():
        raise AssertionError("non-finite sample")
    for index, value in conditions.items():
        torch.testing.assert_close(
            sample.trajectories[:, index, wrapper.action_dim:],
            value, rtol=0, atol=0,
        )
    print("{} initial {:.8f} final {:.8f} best {:.8f} reduction {:.2f}% PASS".format(
        name, initial, final, best, reduction
    ))


def main():
    seed = 20260812
    torch.manual_seed(seed)
    device = get_available_device()
    batch, horizon, action_dim, features_dim, particles = 4, 3, 1, 2, 2
    observation_dim = features_dim * particles
    transition_dim = action_dim + observation_dim
    time_scale = 1000.0
    clean = torch.linspace(
        -0.7, 0.7, horizon * transition_dim, device=device
    ).view(1, horizon, transition_dim).repeat(batch, 1, 1)
    conditions = {
        0: clean[:, 0, action_dim:].clone(),
        horizon - 1: clean[:, -1, action_dim:].clone(),
    }

    tiny = TinyAverageModel(transition_dim, time_scale).to(device)
    tiny_wrapper = ImprovedMeanFlow(
        model=tiny, horizon=horizon, observation_dim=observation_dim,
        action_dim=action_dim, n_solver_steps=1, loss_type="l2",
        time_scale=time_scale,
    ).to(device)
    run_audit("SYNTHETIC", tiny, tiny_wrapper, clean, conditions, 150, 60.0)

    torch.manual_seed(seed)
    pint = IntervalAdaLNPINTDenoiser(
        features_dim=features_dim, action_dim=action_dim,
        hidden_dim=16, projection_dim=16, n_head=4, n_layer=1,
        block_size=horizon, dropout=0.0, positional_bias=False,
        max_particles=None, multiview=False,
    ).to(device)
    pint_wrapper = ImprovedMeanFlow(
        model=pint, horizon=horizon, observation_dim=observation_dim,
        action_dim=action_dim, n_solver_steps=1, loss_type="l2",
        time_scale=time_scale,
    ).to(device)
    run_audit("REAL-PINT", pint, pint_wrapper, clean, conditions, 100, 25.0)
    print("Device: {}".format(device))
    print("IMPROVED MEANFLOW OVERFIT AUDIT: PASS")


if __name__ == "__main__":
    main()
