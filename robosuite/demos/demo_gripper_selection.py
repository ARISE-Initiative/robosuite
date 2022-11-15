"""
This script shows you how to select gripper for an environment.
This is controlled by gripper_type keyword argument.
"""
import numpy as np

import robosuite as suite
from robosuite import ALL_GRIPPERS

if __name__ == "__main__":

    for gripper in ALL_GRIPPERS:

        # Notify user which gripper we're currently using
        print("Using gripper {}...".format(gripper))

        # create environment with selected grippers
        env = suite.make(
            "Lift",
            robots="Panda",
            gripper_types=gripper,
            has_renderer=True,  # make sure we can render to the screen
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            use_camera_obs=False,  # do not use pixel observations
            control_freq=50,  # control should happen fast enough so that simulation looks smoother
            camera_names="frontview",
        )

        # Reset the env
        env.reset()

        # Get action limits
        low, high = env.action_spec

        # Run random policy
        for t in range(100):
            env.render()
            action = np.random.uniform(low, high)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        # close window
        env.close()
