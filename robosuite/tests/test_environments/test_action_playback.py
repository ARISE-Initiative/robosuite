"""
Test script for recording a sequence of random actions and playing them back
"""

import argparse
import json
import os
import random

import h5py
import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config


def test_playback():
    # set seeds
    random.seed(0)
    np.random.seed(0)

    env = robosuite.make(
        "Lift",
        robots=["Panda"],
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env.reset()

    # task instance
    task_xml = env.sim.model.get_xml()
    task_init_state = np.array(env.sim.get_state().flatten())

    # trick for ensuring that we can play MuJoCo demonstrations back
    # deterministically by using the recorded actions open loop
    env.reset_from_xml_string(task_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(task_init_state)
    env.sim.forward()

    # random actions to play
    n_actions = 100
    actions = 0.1 * np.random.uniform(low=-1.0, high=1.0, size=(n_actions, env.action_spec[0].shape[0]))

    # play actions
    print("playing random actions...")
    states = [task_init_state]
    for i in range(n_actions):
        env.step(actions[i])
        states.append(np.array(env.sim.get_state().flatten()))

    # try playback
    print("attempting playback...")
    env.reset()
    env.reset_from_xml_string(task_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(task_init_state)
    env.sim.forward()

    for i in range(n_actions):
        env.step(actions[i])
        state_playback = env.sim.get_state().flatten()
        assert np.all(np.equal(states[i + 1], state_playback))

    env.close()
    print("test passed!")


if __name__ == "__main__":

    test_playback()
