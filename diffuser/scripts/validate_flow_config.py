#!/usr/bin/env python
"""Statically validate the PandaPush flow config without starting a simulator."""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from diffuser.configuration import diffusion_wrapper_kwargs, flow_sampling_kwargs
from diffuser.models import ConditionalFlowMatching, GaussianDiffusion


class ZeroVelocityModel(nn.Module):
    """Minimal model used to exercise exact config-produced wrapper arguments."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.forward_calls = 0

    def forward(self, x, cond, time):
        self.forward_calls += 1
        return self.bias.expand_as(x)


def main():
    """Import the config and validate its wrapper-facing fields."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "pandapush_flow.py"
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("pandapush_flow_validation", str(config_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load {}".format(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = set(sys.modules) - before
    forbidden = [name for name in loaded if any(token in name.lower() for token in ("isaac", "mujoco", "gym"))]
    assert not forbidden, "config import initialized simulator modules: {}".format(forbidden)

    training = module.base["diffusion"]
    planning = module.base["plan"]
    required = {
        "model", "diffusion", "horizon", "features_dim", "hidden_dim",
        "projection_dim", "n_heads", "n_layers", "dropout", "n_diffusion_steps",
        "loss_type", "action_weight", "loss_discount", "batch_size", "learning_rate",
        "normalizer", "renderer", "device", "time_scale",
    }
    missing = required - set(training)
    assert not missing, "flow config is missing keys: {}".format(sorted(missing))
    assert training["diffusion"] == "models.ConditionalFlowMatching"
    assert training["model"] == "models.AdaLNPINTDenoiser"
    assert training["n_diffusion_steps"] == 4
    assert planning["n_diffusion_steps"] == 4
    assert training["time_scale"] == 1000.0
    assert training["device"].startswith("cuda"), "canonical config must remain Linux/GPU-ready"
    signature = inspect.signature(ConditionalFlowMatching.__init__)
    assert "n_timesteps" in signature.parameters
    assert ConditionalFlowMatching.__module__ == "diffuser.models.flow_matching"

    parsed_args = SimpleNamespace(**training)
    wrapper_kwargs = diffusion_wrapper_kwargs(parsed_args, observation_dim=6, action_dim=2)
    expected_wrapper_keys = {
        "horizon", "observation_dim", "action_dim", "n_timesteps", "loss_type",
        "clip_denoised", "predict_epsilon", "action_weight", "loss_weights",
        "loss_discount", "obs_only", "action_only", "time_scale",
    }
    assert set(wrapper_kwargs) == expected_wrapper_keys
    assert wrapper_kwargs["n_timesteps"] == training["n_diffusion_steps"]
    assert wrapper_kwargs["time_scale"] == training["time_scale"]
    assert "n_solver_steps" not in wrapper_kwargs and "n_diffusion_steps" not in wrapper_kwargs
    wrapper = ConditionalFlowMatching(model=ZeroVelocityModel(), **wrapper_kwargs)
    assert wrapper.n_timesteps == wrapper.n_solver_steps == 4
    assert wrapper.time_scale == 1000.0
    for attribute in ("model", "horizon", "observation_dim", "action_dim", "transition_dim", "n_timesteps"):
        assert hasattr(wrapper, attribute)

    parsed_args.n_diffusion_steps = 7
    overridden_kwargs = diffusion_wrapper_kwargs(parsed_args, observation_dim=6, action_dim=2)
    overridden = ConditionalFlowMatching(model=ZeroVelocityModel(), **overridden_kwargs)
    assert overridden.n_solver_steps == 7, "training solver-step override did not reach wrapper"
    assert flow_sampling_kwargs(wrapper, 9) == {"n_steps": 9}
    conditions = {0: torch.zeros(1, 6), wrapper.horizon - 1: torch.ones(1, 6)}
    wrapper(conditions, verbose=False, **flow_sampling_kwargs(wrapper, 9))
    assert wrapper.model.forward_calls == 9, "planning override did not control Euler evaluations"

    gaussian = GaussianDiffusion(
        model=ZeroVelocityModel(), horizon=3, observation_dim=2, action_dim=1,
        n_timesteps=2, loss_type="l2", clip_denoised=True,
    )
    assert flow_sampling_kwargs(gaussian, 9) == {}, "Gaussian sampling must not receive flow-only n_steps"
    print("Config: {}".format(config_path))
    print("Resolved wrapper: {}.{}".format(
        ConditionalFlowMatching.__module__, ConditionalFlowMatching.__name__
    ))
    print("Default Euler steps: {}".format(training["n_diffusion_steps"]))
    print("Time scale: {}".format(wrapper.time_scale))
    print("Training override: 7 | planning override: 9")
    print("Simulator modules initialized: none")
    print("FLOW CONFIG VALIDATION: PASS")


if __name__ == "__main__":
    main()
