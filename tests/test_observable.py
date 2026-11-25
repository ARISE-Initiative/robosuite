"""Regression tests for observation sampling across resets."""

import numpy as np
import pytest

from robosuite.utils.observables import Observable, sensor


@pytest.mark.parametrize("reenable", [False, True])
def test_observable_starts_new_sampling_period_after_reset(reenable):
    reading = np.array([1.0, 2.0])

    @sensor(modality="test")
    def read_sensor(obs_cache):
        return reading

    observable = Observable(name="pose", sensor=read_sensor, sampling_rate=10)
    observable.update(timestep=0.01, obs_cache={})
    np.testing.assert_array_equal(observable.obs, [1.0, 2.0])

    if reenable:
        observable.set_enabled(False)
        observable.set_enabled(True)
    else:
        observable.reset()

    reading = np.array([3.0, 4.0])
    obs_cache = {}
    observable.update(timestep=0.01, obs_cache=obs_cache)

    np.testing.assert_array_equal(observable.obs, reading)
    np.testing.assert_array_equal(obs_cache["pose"], reading)
