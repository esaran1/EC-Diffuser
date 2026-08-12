#!/usr/bin/env python
"""Deterministic synthetic and real-PINT learnability audits for Shortcut Models."""

import torch
from torch import nn

from diffuser.device import get_available_device
from diffuser.models import IntervalAdaLNPINTDenoiser, ShortcutModel


class TinyShortcut(nn.Module):
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


def run(name, model, method, clean, cond, steps, required):
    noise = torch.randn_like(clean)
    t = torch.tensor([0.0, 0.25, 0.0, 0.5], device=clean.device)
    d = torch.tensor([0.25, 0.125, 0.25, 0.125], device=clean.device)
    flow = torch.tensor([True, False, True, False], device=clean.device)

    def evaluate():
        model.eval()
        with torch.no_grad():
            value, _ = method._compute_shortcut_loss(
                clean, cond, noise=noise, t=t, small_d=d, flow_mask=flow
            )
        model.train()
        return value.item()

    initial = evaluate()
    best = initial
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = method._compute_shortcut_loss(
            clean, cond, noise=noise, t=t, small_d=d, flow_mask=flow
        )
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError("non-finite optimization")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % 25 == 0:
            current = evaluate()
            best = min(best, current)
            print("{} step {:3d} validation {:.8f}".format(name, step, current))
    final = evaluate()
    reduction = 100.0*(initial-final)/initial
    if reduction < required:
        raise AssertionError("{} reduction {:.2f}% < {:.1f}%".format(name, reduction, required))
    sample = method(cond, n_steps=1, verbose=False)
    if not torch.isfinite(sample.trajectories).all():
        raise AssertionError("non-finite sample")
    for index, value in cond.items():
        torch.testing.assert_close(sample.trajectories[:, index, method.action_dim:], value, rtol=0, atol=0)
    print("{} initial {:.8f} final {:.8f} best {:.8f} reduction {:.2f}% PASS".format(
        name, initial, final, best, reduction))


def main():
    torch.manual_seed(20260813)
    device = get_available_device()
    batch, horizon, action_dim, features, particles = 4, 3, 1, 2, 2
    observation_dim = features*particles
    transition_dim = action_dim+observation_dim
    clean = torch.linspace(-0.7, 0.7, horizon*transition_dim, device=device).view(
        1, horizon, transition_dim).repeat(batch, 1, 1)
    cond = {0: clean[:, 0, action_dim:].clone(),
            horizon-1: clean[:, -1, action_dim:].clone()}
    tiny = TinyShortcut(transition_dim, 1000.0).to(device)
    method = ShortcutModel(
        model=tiny, horizon=horizon, observation_dim=observation_dim,
        action_dim=action_dim, n_solver_steps=1, loss_type="l2",
        time_scale=1000.0).to(device)
    run("SYNTHETIC", tiny, method, clean, cond, 150, 60.0)

    torch.manual_seed(20260813)
    pint = IntervalAdaLNPINTDenoiser(
        features_dim=features, action_dim=action_dim, hidden_dim=16,
        projection_dim=16, n_head=4, n_layer=1, block_size=horizon,
        dropout=0.0, positional_bias=False, max_particles=None,
        multiview=False).to(device)
    method = ShortcutModel(
        model=pint, horizon=horizon, observation_dim=observation_dim,
        action_dim=action_dim, n_solver_steps=1, loss_type="l2",
        time_scale=1000.0).to(device)
    run("REAL-PINT", pint, method, clean, cond, 100, 25.0)
    print("Device: {}".format(device))
    print("SHORTCUT OVERFIT AUDIT: PASS")


if __name__ == "__main__":
    main()
