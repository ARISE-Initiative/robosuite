import numpy as np
import pytest

import robosuite as suite
from robosuite.controllers.composite.composite_controller import COMPOSITE_CONTROLLERS_DICT
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.controllers import load_composite_controller_config

# New test for different composite controller types
@pytest.mark.parametrize("controller_name", ["WHOLE_BODY_IK"])
@pytest.mark.parametrize("robot", ROBOT_CLASS_MAPPING.keys())
def test_composite_controllers(robot, controller_name):
    """
    Test to validate all composite controllers with predefined robots
    """
    controller_config = load_composite_controller_config(
        controller=controller_name,
        robot=robot,
    )

    config = {
        "env_name": "Lift",
        "robots": robot,
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