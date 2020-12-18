"""
Test all environments with random policies.

This runs some basic sanity checks on the environment, namely, checking that:
    - robot-state exists in the obs, and is a flat array
    - agentview_image exists and is of the correct shape
    - no object-obs in state, because we are only using image observations

Obviously, if an environment crashes during runtime, that is considered a failure as well.
"""
import numpy as np

import robosuite as suite


def test_all_environments():

    envs = sorted(suite.ALL_ENVIRONMENTS)
    for env_name in envs:
        # Create config dict
        env_config = {"env_name": env_name}
        for robot_name in ("Panda", "Sawyer", "Baxter"):
            # create an environment for learning on pixels
            config = None
            if "TwoArm" in env_name:
                if robot_name == "Baxter":
                    robots = robot_name
                    config = "bimanual"
                else:
                    robots = [robot_name, robot_name]
                    config = "single-arm-opposed"
                # compile configuration specs
                env_config["robots"] = robots
                env_config["env_configuration"] = config
            else:
                if robot_name == "Baxter":
                    continue
                env_config["robots"] = robot_name

            # Notify user of which test we are currently on
            print("Testing env: {} with robots {} with config {}...".format(env_name, env_config["robots"], config))

            # Create environment
            env = suite.make(
                **env_config,
                has_renderer=False,  # no on-screen renderer
                has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
                ignore_done=True,  # (optional) never terminates episode
                use_camera_obs=True,  # use camera observations
                camera_heights=84,  # set camera height
                camera_widths=84,  # set camera width
                camera_names="agentview",  # use "agentview" camera
                use_object_obs=False,  # no object feature when training on pixels
                reward_shaping=True,  # (optional) using a shaping reward
            )

            obs = env.reset()

            # get action range
            action_min, action_max = env.action_spec
            assert action_min.shape == action_max.shape

            # Get robot prefix
            pr = env.robots[0].robot_model.naming_prefix

            # run 10 random actions
            for _ in range(10):

                assert pr + "robot-state" in obs
                assert obs[pr + "robot-state"].ndim == 1

                assert "agentview_image" in obs
                assert obs["agentview_image"].shape == (84, 84, 3)

                assert "object-state" not in obs

                action = np.random.uniform(action_min, action_max)
                obs, reward, done, info = env.step(action)

    # Tests passed!
    print("All environment tests passed successfully!")


if __name__ == "__main__":

    test_all_environments()
