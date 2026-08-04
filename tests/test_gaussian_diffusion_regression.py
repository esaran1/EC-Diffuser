"""Behavioral regression coverage for the original GaussianDiffusion wrapper."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from diffuser.models import GaussianDiffusion
from diffuser.models.diffusion import Sample
from diffuser.models.progress import Progress, Silent


class ToyGaussianDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, cond, time):
        return self.scale * x


class RaisingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, cond, time):
        raise RuntimeError("intentional denoiser failure")


def make_gaussian(model=None):
    return GaussianDiffusion(
        model=ToyGaussianDenoiser() if model is None else model,
        horizon=4,
        observation_dim=3,
        action_dim=2,
        n_timesteps=4,
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
        action_weight=2.0,
        loss_discount=0.9,
        loss_weights={1: 1.5},
    )


def test_gaussian_construction_loss_backward_sampling_and_sample_contract():
    torch.manual_seed(17)
    gaussian = make_gaussian()
    trajectory = torch.randn(2, 4, 5)
    conditions = {
        0: torch.randn(2, 3),
        3: torch.randn(2, 3),
    }

    loss, info = gaussian.loss(trajectory, conditions)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert torch.isfinite(info["a0_loss"])
    loss.backward()
    assert gaussian.model.scale.grad is not None
    assert torch.isfinite(gaussian.model.scale.grad)

    sample = gaussian.conditional_sample(
        conditions,
        verbose=False,
        return_chain=True,
        sort_by_value=False,
    )
    assert isinstance(sample, Sample)
    assert sample._fields == ("trajectories", "values", "chains")
    assert sample.trajectories.shape == (2, 4, 5)
    assert sample.values.shape == (2,)
    assert sample.chains.shape == (2, gaussian.n_timesteps + 1, 4, 5)
    torch.testing.assert_close(sample.chains[:, -1], sample.trajectories)
    for timestep, value in conditions.items():
        torch.testing.assert_close(sample.trajectories[:, timestep, 2:], value, rtol=0, atol=0)
        expected = value[:, None, :].expand(-1, sample.chains.shape[1], -1)
        torch.testing.assert_close(sample.chains[:, :, timestep, 2:], expected, rtol=0, atol=0)

    without_chain = gaussian.conditional_sample(conditions, verbose=False, sort_by_value=False)
    assert without_chain.chains is None


def test_model_package_import_does_not_load_training_or_simulator_dependencies():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import sys
before = set(sys.modules)
from diffuser.models import AdaLNPINTDenoiser, ConditionalFlowMatching, GaussianDiffusion
loaded = set(sys.modules) - before
forbidden = [name for name in loaded if any(token in name.lower() for token in ('wandb', 'gym', 'isaac', 'mujoco'))]
assert not forbidden, forbidden
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "diffuser")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_model_progress_matches_original_gaussian_progress_contract():
    original_path = Path(__file__).resolve().parents[1] / "diffuser" / "diffuser" / "utils" / "progress.py"
    spec = importlib.util.spec_from_file_location("original_progress_for_test", str(original_path))
    assert spec is not None and spec.loader is not None
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)

    used_methods = ("update", "stamp", "close")
    for method in used_methods:
        assert callable(getattr(Progress, method))
        assert callable(getattr(original.Progress, method))
        assert getattr(Silent(), method)() is None
        assert getattr(original.Silent(), method)() is None
    assert not hasattr(Progress, "__enter__") and not hasattr(original.Progress, "__enter__")
    assert not hasattr(Progress, "__exit__") and not hasattr(original.Progress, "__exit__")


def test_gaussian_sampling_propagates_denoiser_exceptions():
    gaussian = make_gaussian(model=RaisingDenoiser())
    conditions = {0: torch.zeros(1, 3)}
    with pytest.raises(RuntimeError, match="intentional denoiser failure"):
        gaussian.conditional_sample(conditions, verbose=False)
