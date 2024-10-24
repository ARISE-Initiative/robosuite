"""
Script to test composite robots:

$ pytest -s tests/test_robots/test_composite_robots.py
"""
import logging
from typing import Dict, List, Union

import numpy as np
import pytest

import robosuite as suite
import robosuite.utils.robot_composition_utils as cu
from robosuite.controllers import load_composite_controller_config
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.robots import is_robosuite_robot
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)

TEST_ROBOTS = ["Baxter", "IIWA", "Jaco", "Kinova3", "Panda", "Sawyer", "UR5e", "Tiago", "SpotArm", "GR1"]
TEST_BASES = [
    "RethinkMount",
    "RethinkMinimalMount",
    "NullMount",
    "OmronMobileBase",
    "NullMobileBase",
    "NoActuationBase",
    "Spot",
    "SpotFloating",
]


def create_and_test_env(env: str, robots: Union[str, List[str]], controller_config: Dict, render=True):

    config = {
        "env_name": env,
        "robots": robots,
        "controller_configs": controller_config,
    }

    env = suite.make(
        **config,
        has_renderer=render,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env.reset()
    low, high = env.action_spec
    low = np.clip(low, -1, 1)
    high = np.clip(high, -1, 1)

    # Runs a few steps of the simulation as a sanity check
    for i in range(200):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        if render:
            env.render()

    env.close()


@pytest.mark.parametrize("robot", TEST_ROBOTS)
@pytest.mark.parametrize("base", TEST_BASES)
def test_composite_robot_base_combinations(robot, base):
    if is_robosuite_robot(robot):
        if robot in ["Tiago", "GR1", "SpotArm"]:
            pytest.skip(f"Skipping {robot} for now since it we typically do not attach it to another base.")
        elif base in ["NullMobileBase", "NoActuationBase", "Spot", "SpotFloating"]:
            pytest.skip(f"Skipping {base} for now since comopsite robots do not use {base}.")
        else:
            cu.create_composite_robot(name="CompositeRobot", robot=robot, base=base, grippers="RethinkGripper")
            controller_config = load_composite_controller_config(controller="BASIC", robot="CompositeRobot")
            create_and_test_env(env="Lift", robots="CompositeRobot", controller_config=controller_config, render=False)


@pytest.mark.parametrize("robot", TEST_ROBOTS)
@pytest.mark.parametrize("gripper", GRIPPER_MAPPING.keys())
def test_composite_robot_gripper_combinations(robot, gripper):
    if is_robosuite_robot(robot):
        if robot in ["Tiago"]:
            base = "NullMobileBase"
        elif robot == "GR1":
            base = "NoActuationBase"
        else:
            base = "RethinkMount"

        cu.create_composite_robot(name="CompositeRobot", robot=robot, base=base, grippers=gripper)
        controller_config = load_composite_controller_config(controller="BASIC", robot="CompositeRobot")
        create_and_test_env(env="Lift", robots="CompositeRobot", controller_config=controller_config, render=False)
