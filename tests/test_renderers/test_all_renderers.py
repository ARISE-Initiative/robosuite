"""
Tests that all renderers are able to render properly.
"""

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def test_mujoco_renderer():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mujoco",
    )

    env.reset()

    low, high = env.action_spec

    # do visualization
    for i in range(10):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()


def test_default_renderer():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="default",
    )

    env.reset()

    low, high = env.action_spec

    # do visualization
    for i in range(10):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()


def test_offscreen_renderer():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
    )

    env.reset()

    low, high = env.action_spec

    # do visualization
    for i in range(10):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        assert obs["agentview_image"].shape == (256, 256, 3)
