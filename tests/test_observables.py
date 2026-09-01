"""Tests for observable sampling timing and cache updates."""

import numpy as np
import pytest

from robosuite.utils.observables import Observable, create_deterministic_delayer, sensor

SAMPLING_TIMESTEP = 0.1
DELAY = 0.03
SAMPLE_TIME = SAMPLING_TIMESTEP - DELAY
EARLY_TIME = 0.01


def make_clock_observable(delay=0.0, delayer=None):
    clock = {"time": 0.0}
    sample_times = []

    @sensor(modality="test")
    def read_clock(_):
        sample_times.append(clock["time"])
        return np.array([clock["time"]])

    observable = Observable(
        name="clock",
        sensor=read_clock,
        delayer=delayer if delayer is not None else create_deterministic_delayer(delay),
        sampling_rate=1 / SAMPLING_TIMESTEP,
    )
    sample_times.clear()  # Ignore the sensor validity check during initialization.
    return observable, clock, sample_times


def test_zero_delay_samples_at_period_boundary():
    observable, clock, sample_times = make_clock_observable()
    cache = {}

    clock["time"] = EARLY_TIME
    observable.update(EARLY_TIME, cache)

    assert sample_times == []
    assert "clock" not in cache

    clock["time"] = SAMPLING_TIMESTEP
    observable.update(SAMPLING_TIMESTEP - EARLY_TIME, cache)

    assert sample_times == pytest.approx([SAMPLING_TIMESTEP])
    assert cache["clock"] == pytest.approx(SAMPLING_TIMESTEP)


def test_delay_samples_before_period_boundary():
    observable, clock, sample_times = make_clock_observable(DELAY)
    cache = {}

    clock["time"] = EARLY_TIME
    observable.update(EARLY_TIME, cache)

    assert sample_times == []
    assert "clock" not in cache

    clock["time"] = SAMPLE_TIME
    observable.update(SAMPLE_TIME - EARLY_TIME, cache)

    assert sample_times == pytest.approx([SAMPLE_TIME])
    assert observable.obs == pytest.approx(SAMPLE_TIME)
    assert cache["clock"] == pytest.approx(SAMPLE_TIME)


def test_force_update_does_not_consume_periodic_sample():
    delays = iter([DELAY, 0.0])
    sampled_delays = []

    def delayer():
        delay = next(delays)
        sampled_delays.append(delay)
        return delay

    observable, clock, sample_times = make_clock_observable(delayer=delayer)
    cache = {}

    observable.update(EARLY_TIME, cache, force=True)

    assert sample_times == pytest.approx([0.0])
    assert sampled_delays == pytest.approx([DELAY])
    assert observable._time_since_last_sample == 0.0
    assert observable._current_delay == pytest.approx(DELAY)
    assert observable._sampled is False
    assert cache["clock"] == pytest.approx(0.0)

    clock["time"] = EARLY_TIME
    observable.update(EARLY_TIME, cache)
    assert cache["clock"] == pytest.approx(0.0)
    clock["time"] = SAMPLE_TIME
    observable.update(SAMPLE_TIME - EARLY_TIME, cache)

    assert sample_times == pytest.approx([0.0, SAMPLE_TIME])
    assert sampled_delays == pytest.approx([DELAY, 0.0])
    assert cache["clock"] == pytest.approx(SAMPLE_TIME)


def test_reset_clears_sampled_state():
    observable, clock, sample_times = make_clock_observable(DELAY)
    cache = {}

    clock["time"] = SAMPLE_TIME
    observable.update(SAMPLE_TIME, cache)
    assert sample_times == pytest.approx([SAMPLE_TIME])

    observable.reset()
    sample_times.clear()
    cache.clear()

    clock["time"] = 2 * SAMPLE_TIME
    observable.update(SAMPLE_TIME, cache)

    assert sample_times == pytest.approx([2 * SAMPLE_TIME])
    assert cache["clock"] == pytest.approx(2 * SAMPLE_TIME)
