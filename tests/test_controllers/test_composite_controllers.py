from typing import List, Union, Dict
import numpy as np
import pytest
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.robots import ROBOT_CLASS_MAPPING


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
def test_basic_controller_predefined_robots(robot):
    """
    Tests the basic controller with all predefined robots
    (i.e., ALL_ROBOTS)
    """

    controller_config = load_composite_controller_config(
        controller="BASIC",
        robot=robot,
    )

    create_and_test_env(env="Lift", robots=robot, controller_config=controller_config)