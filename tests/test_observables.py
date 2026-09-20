"""Test sensor delay sampling and its use by observables."""

import numpy as np
import pytest

from robosuite.utils.observables import (
    Observable,
    create_deterministic_delayer,
    create_gaussian_sampled_delayer,
    sensor,
)


@pytest.mark.parametrize(
    "mean, std, sample, expected",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.04, 0.0, 0.0, 0.04),
        (0.75, 0.0, 0.0, 0.75),
        (1.25, 0.0, 0.0, 1.25),
        (0.04, 0.01, -1.0, 0.03),
        (0.04, 0.01, 1.0, 0.05),
        (0.04, 0.05, -1.0, 0.0),
    ],
)
def test_gaussian_delayer_preserves_seconds(monkeypatch, mean, std, sample, expected):
    """Keep fractional seconds while clipping negative samples to zero."""
    monkeypatch.setattr(np.random, "randn", lambda: sample)
    delayer = create_gaussian_sampled_delayer(mean, std)

    assert delayer() == pytest.approx(expected)


def test_zero_std_gaussian_matches_deterministic_observable():
    """A zero-variance Gaussian delay should behave like a fixed delay."""

    @sensor(modality="test")
    def constant_sensor(obs_cache):
        return np.array([1.0])

    gaussian = Observable(
        name="reading",
        sensor=constant_sensor,
        delayer=create_gaussian_sampled_delayer(0.04, 0.0),
        sampling_rate=20,
    )
    deterministic = Observable(
        name="reading",
        sensor=constant_sensor,
        delayer=create_deterministic_delayer(0.04),
        sampling_rate=20,
    )
    gaussian_cache, deterministic_cache = {}, {}

    for _ in range(5):
        gaussian.update(0.02, gaussian_cache)
        deterministic.update(0.02, deterministic_cache)
        np.testing.assert_array_equal(gaussian.obs, deterministic.obs)
        assert gaussian_cache.keys() == deterministic_cache.keys()
