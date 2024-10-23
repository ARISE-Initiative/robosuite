from typing import Dict, List, Union

import numpy as np

import robosuite as suite


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
