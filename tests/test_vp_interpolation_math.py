"""Correctness checks for the variance-preserving interpolation path.

These are the checks item 17 of the investigation requires before any VP arm
could be trained. They are deliberately narrow: the derivative, the endpoints,
the variance property, and the conditioning order.
"""

import numpy as np
import pytest
import torch


def s(t):
    return np.sqrt((1.0 - t) ** 2 + t**2)


def vp_interpolant(t, z, x):
    return ((1.0 - t) * z + t * x) / s(t)


def vp_velocity(t, z, x):
    """dy/dt for y(t) = ((1-t)z + tx)/s(t)."""
    n = (1.0 - t) * z + t * x
    st = s(t)
    return (x - z) / st - n * (2.0 * t - 1.0) / st**3


@pytest.mark.parametrize("t", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_analytic_velocity_matches_finite_differences(t):
    rng = np.random.RandomState(0)
    z, x = rng.randn(512), rng.randn(512)
    eps = 1e-6

    numeric = (vp_interpolant(t + eps, z, x) - vp_interpolant(t - eps, z, x)) / (2 * eps)

    np.testing.assert_allclose(numeric, vp_velocity(t, z, x), atol=1e-6)


@pytest.mark.parametrize("t", [0.1, 0.5, 0.9])
def test_analytic_velocity_matches_autograd(t):
    """An independent check that does not reuse the finite-difference stencil."""
    torch.manual_seed(0)
    z = torch.randn(32, dtype=torch.float64)
    x = torch.randn(32, dtype=torch.float64)

    t_torch = torch.tensor(t, dtype=torch.float64, requires_grad=True)
    scale = torch.sqrt((1 - t_torch) ** 2 + t_torch**2)
    y = ((1 - t_torch) * z + t_torch * x) / scale
    grad = torch.autograd.grad(y.sum(), t_torch)[0].item()

    assert grad == pytest.approx(vp_velocity(t, z.numpy(), x.numpy()).sum(), abs=1e-9)


def test_endpoints_are_exactly_noise_and_data():
    rng = np.random.RandomState(0)
    z, x = rng.randn(256), rng.randn(256)

    np.testing.assert_allclose(vp_interpolant(0.0, z, x), z, atol=0)
    np.testing.assert_allclose(vp_interpolant(1.0, z, x), x, atol=0)


def test_vp_holds_unit_variance_for_standardized_endpoints():
    """The defining property -- but only when the data is standardized."""
    rng = np.random.RandomState(0)
    x = rng.randn(100_000)
    z = rng.randn(100_000)

    for t in np.linspace(0, 1, 11):
        assert (vp_interpolant(t, z, x) ** 2).mean() == pytest.approx(1.0, abs=0.02)


def test_vp_does_not_preserve_variance_on_bounded_features():
    """Guards the naming rule: min-max features plus the denominator is NOT VP."""
    rng = np.random.RandomState(0)
    bounded = rng.uniform(-1.0, 1.0, size=100_000) * 0.35  # E[x^2] far below 1
    z = rng.randn(100_000)

    assert (vp_interpolant(1.0, z, bounded) ** 2).mean() < 0.2
    # And it is materially off unit variance in the interior too.
    assert (vp_interpolant(0.75, z, bounded) ** 2).mean() < 0.5


def test_conditions_must_be_reimposed_after_dividing():
    """Item 18: dividing by s(t) after conditioning would corrupt known values."""
    rng = np.random.RandomState(0)
    x, z = rng.randn(200), rng.randn(200)
    index = np.arange(0, 200, 4)
    known = x[index].copy()

    t = 0.5
    # Correct order: interpolate, divide, then re-impose.
    correct = vp_interpolant(t, z, x)
    correct[index] = known
    np.testing.assert_allclose(correct[index], known, atol=0)

    # Wrong order: impose into the numerator, then divide -> scaled by 1/s(t).
    numerator = (1 - t) * z + t * x
    numerator[index] = known
    wrong = numerator / s(t)
    assert np.abs(wrong[index] - known).max() > 0.1
