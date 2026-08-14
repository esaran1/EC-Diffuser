#!/usr/bin/env python3
"""Deterministic learnability audit for auxiliary-head Improved MeanFlow."""

import torch
from torch import nn

from diffuser.device import get_available_device
from diffuser.models import AuxiliaryImprovedMeanFlow, AuxiliaryIntervalTemporalUnet


class TinyDualModel(nn.Module):
    def __init__(self, transition_dim, time_scale):
        super().__init__()
        self.time_scale = time_scale
        self.shared = nn.Sequential(nn.Linear(transition_dim + 2, 32), nn.SiLU())
        self.u_head = nn.Linear(32, transition_dim)
        self.v_head = nn.Linear(32, transition_dim)

    def _features(self, x, time, interval):
        t = (time/self.time_scale).view(-1, 1, 1).expand(-1, x.shape[1], 1)
        d = (interval/self.time_scale).view(-1, 1, 1).expand(-1, x.shape[1], 1)
        return self.shared(torch.cat([x, t, d], dim=-1))

    def forward(self, x, cond, time, interval):
        return self.u_head(self._features(x, time, interval))

    def forward_with_aux(self, x, cond, time, interval):
        features = self._features(x, time, interval)
        return self.u_head(features), self.v_head(features)


def wrapper(model, horizon, observation_dim, action_dim, time_scale):
    return AuxiliaryImprovedMeanFlow(
        model=model, horizon=horizon, observation_dim=observation_dim,
        action_dim=action_dim, n_solver_steps=1, loss_type="l2",
        time_scale=time_scale,
    )


def run(name, model, method, clean, conditions, steps, required_reduction):
    noise = torch.randn_like(clean)
    r = torch.linspace(0.05, 0.35, clean.shape[0], device=clean.device)
    t = torch.linspace(0.55, 0.95, clean.shape[0], device=clean.device)

    def evaluate():
        model.eval()
        with torch.no_grad():
            value, _ = method._compute_meanflow_loss(
                clean, conditions, noise=noise, r=r, t=t)
        model.train()
        return value.item()

    initial = evaluate()
    best = initial
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = method._compute_meanflow_loss(
            clean, conditions, noise=noise, r=r, t=t)
        if not torch.isfinite(loss):
            raise AssertionError("non-finite loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError("missing or non-finite gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % 25 == 0:
            current = evaluate()
            best = min(best, current)
            print("{} step {:3d} validation {:.8f}".format(name, step, current))
    final = evaluate()
    reduction = 100.0 * (initial-final) / initial
    if reduction < required_reduction:
        raise AssertionError("{} reduction {:.2f}% < {:.1f}%".format(
            name, reduction, required_reduction))
    sample = method(conditions, n_steps=1, verbose=False)
    if not torch.isfinite(sample.trajectories).all():
        raise AssertionError("non-finite sample")
    print("{} initial {:.8f} final {:.8f} best {:.8f} reduction {:.2f}% PASS".format(
        name, initial, final, best, reduction))


def main():
    seed = 20260813
    torch.manual_seed(seed)
    device = get_available_device()
    batch, horizon, action_dim, observation_dim = 4, 3, 1, 4
    transition_dim, time_scale = action_dim + observation_dim, 1000.0
    clean = torch.linspace(-0.7, 0.7, horizon*transition_dim, device=device)
    clean = clean.view(1, horizon, transition_dim).repeat(batch, 1, 1)
    conditions = {0: clean[:, 0, action_dim:].clone(),
                  horizon-1: clean[:, -1, action_dim:].clone()}

    tiny = TinyDualModel(transition_dim, time_scale).to(device)
    run("SYNTHETIC", tiny, wrapper(tiny, horizon, observation_dim,
        action_dim, time_scale), clean, conditions, 150, 60.0)

    torch.manual_seed(seed)
    temporal = AuxiliaryIntervalTemporalUnet(
        horizon=horizon, transition_dim=transition_dim,
        cond_dim=observation_dim, dim=8, dim_mults=(1, 2), attention=False,
    ).to(device)
    run("REAL-TEMPORAL", temporal, wrapper(temporal, horizon,
        observation_dim, action_dim, time_scale), clean, conditions, 100, 20.0)
    print("Device: {}".format(device))
    print("AUXILIARY IMPROVED MEANFLOW OVERFIT AUDIT: PASS")


if __name__ == "__main__":
    main()
