import sys
import traceback
import numpy as np
from termcolor import colored

import robosuite as suite
from robosuite import ALL_ROBOTS
from robosuite.controllers import load_composite_controller_config

def test_basic_controller_predefined_robots():
    """
    Test to test the base controller with all predefined robots
    (i.e., ALL_ROBOTS)
    """

    success = True

    for robot in ALL_ROBOTS:

        try:
            controller_config = load_composite_controller_config(
                controller="BASIC",
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
    
        except Exception as e:
            success = False
            print("----------------", file=sys.stderr)
            print(f"An exception occurred when loading {robot} robot.", file=sys.stderr)
            traceback.print_exc()
            print("----------------", file=sys.stderr)

    assert success

if __name__ == "__main__":
    test_basic_controller_predefined_robots()