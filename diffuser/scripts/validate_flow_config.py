#!/usr/bin/env python
"""Statically validate the PandaPush flow config without starting a simulator."""

import importlib.util
import inspect
import sys
from pathlib import Path

from diffuser.models import ConditionalFlowMatching


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
        "normalizer", "renderer", "device",
    }
    missing = required - set(training)
    assert not missing, "flow config is missing keys: {}".format(sorted(missing))
    assert training["diffusion"] == "models.ConditionalFlowMatching"
    assert training["model"] == "models.AdaLNPINTDenoiser"
    assert training["n_diffusion_steps"] == 4
    assert planning["n_diffusion_steps"] == 4
    assert training["device"].startswith("cuda"), "canonical config must remain Linux/GPU-ready"
    signature = inspect.signature(ConditionalFlowMatching.__init__)
    assert "n_timesteps" in signature.parameters
    assert ConditionalFlowMatching.__module__ == "diffuser.models.flow_matching"
    print("Config: {}".format(config_path))
    print("Resolved wrapper: {}.{}".format(
        ConditionalFlowMatching.__module__, ConditionalFlowMatching.__name__
    ))
    print("Default Euler steps: {}".format(training["n_diffusion_steps"]))
    print("Simulator modules initialized: none")
    print("FLOW CONFIG VALIDATION: PASS")


if __name__ == "__main__":
    main()
