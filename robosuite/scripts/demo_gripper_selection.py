"""
This script shows you how to select gripper for an environment.
This is controlled by gripper_type keyword argument
"""
import robosuite as suite
from robosuite.wrappers import GymWrapper

# TODO: Why are we wrapping this in a GymWrapper?


if __name__ == "__main__":

    grippers = ["SawyerGripper", "PR2Gripper", "RobotiqGripper", "RobotiqThreeFingerGripper", "PandaGripper"]

    for gripper in grippers:

        # create environment with selected grippers
        env = GymWrapper(
            suite.make(
                "PickPlace",
                robots="Sawyer",
                gripper_types=gripper,
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=True,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=100,  # control should happen fast enough so that simulation looks smooth
            )
        )

        # run a random policy
        observation = env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        # close window
        env.close()
