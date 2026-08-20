"""Pins the action-normalization facts established in the Isaac Gym audit.

These tests encode findings from experiments/isaacgym_debug_investigation.md so
that a future change to the normalizers cannot silently invalidate them.
"""

import numpy as np
import pytest

from diffuser.datasets.normalization import LimitsNormalizer, SafeLimitsNormalizer


def test_limits_normalizer_round_trip_is_exact():
    """Item 2's core claim: the inverse transform recovers the input."""
    rng = np.random.RandomState(0)
    data = rng.uniform(-1.0, 1.0, size=(5000, 3)).astype(np.float32)
    data[:, 2] = rng.uniform(-0.362, 0.062, size=5000)  # the real z range

    normalizer = SafeLimitsNormalizer(data)
    restored = normalizer.unnormalize(normalizer.normalize(data))

    np.testing.assert_allclose(restored, data, atol=1e-6)


def test_asymmetric_channel_maps_zero_away_from_zero():
    """Defect A: on an asymmetric range, a zero action is not a null action.

    The real 3C z action spans [-0.362, +0.062] -- the arm pushes down far
    further than it lifts. LimitsNormalizer maps that onto [-1, 1], so raw zero
    lands well above normalized zero, and normalized zero decodes to a downward
    push. Any tendency of a policy to emit near-zero output is therefore a press
    into the table rather than a no-op.
    """
    data = np.linspace(-0.362, 0.062, 1000, dtype=np.float32).reshape(-1, 1)
    normalizer = LimitsNormalizer(data)

    raw_zero_normed = normalizer.normalize(np.zeros((1, 1), dtype=np.float32))
    normed_zero_raw = normalizer.unnormalize(np.zeros((1, 1), dtype=np.float32))

    assert raw_zero_normed.item() == pytest.approx(0.706, abs=5e-3)
    assert normed_zero_raw.item() == pytest.approx(-0.150, abs=5e-3)
    assert normed_zero_raw.item() < 0.0, "normalized zero must decode to a downward push"


def test_safe_limits_widens_every_dimension_not_just_the_constant_one():
    """Defect B: a latent bug, documented rather than fixed.

    `SafeLimitsNormalizer` intends to widen only a constant dimension, but the
    body of its loop assigns to `self.mins` / `self.maxs` array-wide. One
    constant channel therefore rescales every other channel by eps.

    This does not fire on the 3C action data (no channel is constant), so no
    result in the investigation is affected. The test pins the real behavior so
    that a future fix is a deliberate, visible change rather than a silent one.
    """
    data = np.zeros((100, 2), dtype=np.float32)
    data[:, 0] = np.linspace(-1.0, 1.0, 100)  # varying
    data[:, 1] = 0.5  # constant -> triggers the widening branch

    normalizer = SafeLimitsNormalizer(data)

    # The varying dimension is widened too, even though it was never constant.
    assert normalizer.mins[0] == pytest.approx(-2.0)
    assert normalizer.maxs[0] == pytest.approx(2.0)


def test_variance_of_the_bounded_representation_is_far_below_one():
    """Item 6: the VP formula presumes E[x^2] == 1, which does not hold here."""
    rng = np.random.RandomState(0)
    data = rng.uniform(-1.0, 1.0, size=(20000, 8)).astype(np.float32) * 0.35

    normalizer = LimitsNormalizer(data)
    normed = normalizer.normalize(data)

    assert (normed**2).mean() < 0.5, "bounded features do not have unit second moment"


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_vp_path_is_exact_only_after_standardization(t):
    """Item 6's headline: standardization is what makes the VP formula true."""
    rng = np.random.RandomState(0)
    raw = (rng.uniform(-1.0, 1.0, size=(20000, 16)) * 0.35).astype(np.float32)
    noise = rng.randn(*raw.shape).astype(np.float32)
    scale = np.sqrt((1 - t) ** 2 + t**2)

    raw_vp = ((1 - t) * noise + t * raw) / scale

    standardized = (raw - raw.mean(0)) / (raw.std(0) + 1e-8)
    std_vp = ((1 - t) * noise + t * standardized) / scale

    assert (std_vp**2).mean() == pytest.approx(1.0, abs=0.02)
    if t > 0.5:
        assert (raw_vp**2).mean() < 0.5, "the un-standardized path collapses"
