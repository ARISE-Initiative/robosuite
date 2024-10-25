"""
Script to test composite controllers:

$ pytest -s tests/test_controllers/test_composite_controllers.py
"""

from typing import Dict, List, Union

import mujoco
import numpy as np
import pytest

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.robots import ROBOT_CLASS_MAPPING


def is_robosuite_robot(robot: str) -> bool:
    """
    robot is robosuite repo robot if can import robot class from robosuite.models.robots
    """
    try:
        module = __import__("robosuite.models.robots", fromlist=[robot])
        getattr(module, robot)
        return True
    except (ImportError, AttributeError):
        return False


def create_and_test_env(
    env: str,
    robots: Union[str, List[str]],
    controller_config: Dict,
):

    config = {
        "env_name": env,
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
    low = np.clip(low, -1, 1)
    high = np.clip(high, -1, 1)

    # Runs a few steps of the simulation as a sanity check
    for i in range(10):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)

    env.close()


@pytest.mark.parametrize("robot", ROBOT_CLASS_MAPPING.keys())
@pytest.mark.parametrize("controller", [None, "BASIC", "WHOLE_BODY_IK"])
def test_basic_controller_predefined_robots(robot, controller):
    """
    Tests the basic controller with all predefined robots
    (i.e., ALL_ROBOTS) and controller types.
    """
    if robot == "SpotArm" and mujoco.__version__ <= "3.1.2":
        pytest.skip(
            "Skipping test for SpotArm because the robot's mesh only works for Mujoco version 3.1.3 and above"
            "Spot arm xml and meshes were taken from: "
            "https://github.com/google-deepmind/mujoco_menagerie/tree/main/boston_dynamics_spot"
        )

    if controller is None and not is_robosuite_robot(robot):
        pytest.skip(f"Skipping test for non-robosuite robot {robot} with no specified controller.")

    # skip currently problematic robots
    if robot == "GR1":
        pytest.skip("Skipping GR1 for now due to error with the leg controller.")

    if robot == "Jaco":
        pytest.skip("Skipping Jaco for now due to error with action formatting.")

    controller_config = load_composite_controller_config(
        controller=controller,
        robot=robot,
    )

    create_and_test_env(env="Lift", robots=robot, controller_config=controller_config)
