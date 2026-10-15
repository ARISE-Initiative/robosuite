"""Exercise GymWrapper seeding without loading MuJoCo or creating a renderer."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gym_wrapper(monkeypatch):
    """Load the real wrapper modules without robosuite's simulation imports."""
    root = Path(__file__).resolve().parents[2]
    wrappers = root / "robosuite" / "wrappers"

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    wrapper = load("_seed_test_wrapper", wrappers / "wrapper.py").Wrapper
    package = types.ModuleType("robosuite.wrappers")
    package.Wrapper = wrapper
    monkeypatch.setitem(sys.modules, "robosuite.wrappers", package)
    return load("_seed_test_gym_wrapper", wrappers / "gym_wrapper.py").GymWrapper


class RandomizedEnv:
    """Model reloads use env.rng; soft resets use a retained sampler reference."""

    reward_scale = 1.0
    use_object_obs = True
    use_camera_obs = False
    robots = [SimpleNamespace(robot_model=object())]
    action_spec = (-np.ones(1), np.ones(1))

    def __init__(self, hard_reset, seed=7, bit_generator=np.random.PCG64):
        self.seed = seed
        self.rng = np.random.Generator(bit_generator(seed))
        self.sampler_rng = self.rng
        self.hard_reset = hard_reset
        self.size = self.rng.uniform(0.02, 0.022, 3)

    def reset(self):
        if self.hard_reset:
            self.size = self.rng.uniform(0.02, 0.022, 3)
        placement = self.sampler_rng.uniform(-0.1, 0.1, 3)
        robot_noise = self.rng.normal(size=2)
        legacy_noise = np.random.normal(size=2)
        return {
            "object-state": np.concatenate((self.size, placement, robot_noise, legacy_noise)),
            "robot0_proprio-state": np.zeros(1),
        }


@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("flatten_obs", [False, True])
def test_seed_repeats_reset_and_preserves_rng_references(gym_wrapper, hard_reset, flatten_obs):
    env = RandomizedEnv(hard_reset)
    wrapped = gym_wrapper(env, flatten_obs=flatten_obs)
    rng = env.rng
    first, info = wrapped.reset(seed=123)
    second, _ = wrapped.reset(seed=123)
    if not flatten_obs:
        first, second = first["object-state"], second["object-state"]
    np.testing.assert_array_equal(first, second)
    assert info == {}
    assert env.rng is rng is env.sampler_rng


@pytest.mark.parametrize("hard_reset", [False, True])
def test_seed_changes_generator_and_none_continues(gym_wrapper, hard_reset):
    env = RandomizedEnv(hard_reset)
    wrapped = gym_wrapper(env)
    first, _ = wrapped.reset(seed=123)
    state = env.rng.bit_generator.state
    expected = np.random.default_rng()
    expected.bit_generator.state = state
    if hard_reset:
        expected.uniform(0.02, 0.022, 3)
    expected_placement = expected.uniform(-0.1, 0.1, 3)
    continued, _ = wrapped.reset(seed=None)
    np.testing.assert_array_equal(continued[3:6], expected_placement)
    assert not np.array_equal(first[3:6], continued[3:6])
    different, _ = wrapped.reset(seed=456)
    assert not np.array_equal(first[3:6], different[3:6])


@pytest.mark.parametrize("hard_reset", [False, True])
def test_independent_environments_share_seed(gym_wrapper, hard_reset):
    # Soft reset retains construction-time geometry; only hard reset rebuilds it.
    first = gym_wrapper(RandomizedEnv(hard_reset, seed=7))
    second = gym_wrapper(RandomizedEnv(hard_reset, seed=7 if not hard_reset else 8))
    np.testing.assert_array_equal(first.reset(seed=123)[0], second.reset(seed=123)[0])


def test_unseeded_wrapper_preserves_constructor_sequence(gym_wrapper):
    env = RandomizedEnv(False, seed=77)
    expected = np.random.default_rng()
    expected.bit_generator.state = env.rng.bit_generator.state
    expected.uniform(-0.1, 0.1, 3)  # Wrapper's observation-space initialization reset.
    expected.normal(size=2)
    wrapped = gym_wrapper(env)
    observed, _ = wrapped.reset()
    np.testing.assert_array_equal(observed[3:6], expected.uniform(-0.1, 0.1, 3))


def test_gym_rng_is_seeded_and_continues(gym_wrapper):
    wrapped = gym_wrapper(RandomizedEnv(False))
    wrapped.reset(seed=123)
    assert wrapped.np_random is wrapped.env.rng
    first = wrapped.np_random.random(4)
    if hasattr(wrapped, "np_random_seed"):
        assert wrapped.np_random_seed == 123
    wrapped.reset(seed=123)
    np.testing.assert_array_equal(wrapped.np_random.random(4), first)
    rng = wrapped.np_random
    wrapped.reset()
    assert wrapped.np_random is rng
    if hasattr(wrapped, "np_random_seed"):
        assert wrapped.np_random_seed == 123


def test_preserves_nondefault_bit_generator(gym_wrapper):
    env = RandomizedEnv(False, bit_generator=np.random.Philox)
    wrapped = gym_wrapper(env)
    rng = env.rng
    first, _ = wrapped.reset(seed=123)
    np.testing.assert_array_equal(first, wrapped.reset(seed=123)[0])
    assert env.rng is rng
    assert isinstance(env.rng.bit_generator, np.random.Philox)


@pytest.mark.parametrize("seed", [1.5, "123"])
def test_rejects_noninteger_seed(gym_wrapper, seed):
    wrapped = gym_wrapper(RandomizedEnv(False))
    with pytest.raises(TypeError, match="Seed must be an integer"):
        wrapped.reset(seed=seed)
