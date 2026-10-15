"""Verify GymWrapper reset seeding with real, CPU-only Panda simulations.

Run from the repository root with:
    python -m pytest tests/test_wrappers/test_gym_wrapper_seed_integration.py
"""

from copy import deepcopy

import numpy as np
import pytest

import robosuite as suite
from robosuite.wrappers import GymWrapper


def make_env(env_name, hard_reset, seed):
    return suite.make(
        env_name,
        robots="Panda",
        seed=seed,
        hard_reset=hard_reset,
        initialization_noise={"magnitude": 0.02, "type": "gaussian"},
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        renderer="mujoco",
    )


def snapshot(env, obs):
    assert env.viewer is None
    assert env.sim._render_context_offscreen is None
    return {
        **{key: value.copy() for key, value in obs.items()},
        "qpos": env.sim.data.qpos.copy(),
        "geom_size": env.sim.model.geom_size.copy(),
        "body_mass": env.sim.model.body_mass.copy(),
    }


def assert_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


@pytest.mark.parametrize("env_name", ["Lift", "NutAssembly"])
@pytest.mark.parametrize("hard_reset", [False, True])
def test_real_reset_seeding(env_name, hard_reset):
    env = make_env(env_name, hard_reset, seed=7)
    wrapped = GymWrapper(env, flatten_obs=False)
    try:
        rng = env.rng
        sampler = env.placement_initializer
        first = snapshot(env, wrapped.reset(seed=123)[0])
        assert_equal(first, snapshot(env, wrapped.reset(seed=123)[0]))
        assert env.rng is rng
        assert env.placement_initializer is sampler
        if env_name == "Lift":
            assert sampler.rng is rng
        else:
            assert all(child.rng is rng for child in sampler.samplers.values())
        assert wrapped.np_random is rng

        different = snapshot(env, wrapped.reset(seed=456)[0])
        assert not np.array_equal(first["qpos"], different["qpos"])
        if env_name == "Lift" and hard_reset:
            assert not np.array_equal(first["geom_size"], different["geom_size"])
            assert not np.array_equal(first["body_mass"], different["body_mass"])
        if not hard_reset:
            np.testing.assert_array_equal(first["geom_size"], different["geom_size"])

        wrapped.reset(seed=123)
        state = deepcopy(rng.bit_generator.state)
        continued = snapshot(env, wrapped.reset(seed=None)[0])
        assert state != rng.bit_generator.state
        assert not np.array_equal(first["qpos"], continued["qpos"])
        wrapped.reset(seed=123)
        assert_equal(continued, snapshot(env, wrapped.reset()[0]))
    finally:
        wrapped.close()


@pytest.mark.parametrize("hard_reset", [False, True])
def test_real_independent_environments(hard_reset):
    # Soft resets preserve the compiled model, so their construction seeds must match.
    envs = [make_env("Lift", hard_reset, seed=seed) for seed in (7, 8 if hard_reset else 7)]
    wrappers = [GymWrapper(env, flatten_obs=False) for env in envs]
    try:
        assert_equal(
            snapshot(envs[0], wrappers[0].reset(seed=123)[0]),
            snapshot(envs[1], wrappers[1].reset(seed=123)[0]),
        )
    finally:
        for wrapped in wrappers:
            wrapped.close()


@pytest.mark.parametrize("hard_reset", [False, True])
def test_real_constructor_seed_sequence_is_unchanged(hard_reset):
    wrapped_env = make_env("Lift", hard_reset, seed=77)
    raw_env = make_env("Lift", hard_reset, seed=77)
    wrapped = GymWrapper(wrapped_env, flatten_obs=False)
    try:
        raw_env.reset()  # Match GymWrapper's observation-space initialization reset.
        for _ in range(2):
            actual = snapshot(wrapped_env, wrapped.reset()[0])
            expected_obs = raw_env.reset()
            expected = snapshot(raw_env, {key: expected_obs[key] for key in wrapped.keys})
            assert_equal(actual, expected)
    finally:
        wrapped.close()
        raw_env.close()
