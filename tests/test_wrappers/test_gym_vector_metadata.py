"""Regression test for issue #847.

GymWrapper used to define the class attribute ``metadata = None`` which made
``gymnasium.vector.SyncVectorEnv`` raise a ``TypeError`` during construction
(when it writes ``self.metadata["autoreset_mode"]``), before any reset/step.
"""
import numpy as np
import pytest

import gymnasium as gym

import robosuite as suite
from robosuite.wrappers import GymWrapper

pytest.importorskip("gymnasium")


def _make_wrapped_env(flatten):
    base = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    if flatten:
        keys = list(base.observation_spec().keys())
        return GymWrapper(base, keys=keys, flatten_obs=True)
    return GymWrapper(base, flatten_obs=False)


@pytest.mark.parametrize("flatten", [True, False])
def test_sync_vector_env_construct_reset_step(flatten):
    venv = gym.vector.SyncVectorEnv([lambda f=flatten: _make_wrapped_env(f) for _ in range(2)])
    obs, _ = venv.reset()
    if flatten:
        assert obs.shape[0] == 2
    else:
        assert isinstance(obs, dict) and all(np.asarray(v).shape[0] == 2 for v in obs.values())
    venv.step(venv.action_space.sample())
    venv.close()
