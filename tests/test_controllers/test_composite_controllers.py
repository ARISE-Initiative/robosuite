import sys
import traceback
from typing import List, Union

import numpy as np
from termcolor import colored

import robosuite as suite
from robosuite import ALL_ROBOTS
from robosuite.controllers import load_composite_controller_config


def create_and_run_test_env(controller_config: dict, env: str, robots: Union[str, List[str]]):

    config = {
        "env_name": "Lift",
        "robots": robots,
        "controller_configs": controller_config,
    }

    env = suite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env.reset()

    low, high = env.action_spec

    # Runs a few steps of the simulation as a sanity check
    for i in range(10):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)

    env.close()


def test_basic_controller_predefined_robots():
    """
    Tests the basic controller with all predefined robots
    (i.e., ALL_ROBOTS)
    """

    for robot in ALL_ROBOTS:

        # TODO: remove once DOF issue is fixed
        if robot == "Jaco":
            continue

        controller_config = load_composite_controller_config(
            controller="BASIC",
            robot=robot,
        )

        create_and_run_test_env(controller_config, robot)


def test_whole_body_ik_controller_predefined_robots():
    """
    Tests the whole body ik controller with all predefined robots
    (i.e., ALL_ROBOTS)
    """

    for robot in ALL_ROBOTS:

        # TODO: remove once DOF issue is fixed
        if robot == "Jaco":
            continue

        controller_config = load_composite_controller_config(
            controller="WHOLE_BODY_IK",
            robot=robot,
        )

        create_and_run_test_env(controller_config, robot)


if __name__ == "__main__":
    test_basic_controller_predefined_robots()
    test_whole_body_ik_controller_predefined_robots()
