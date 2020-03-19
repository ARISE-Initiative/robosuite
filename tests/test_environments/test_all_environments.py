"""
Test all environments with random policies.
"""
import numpy as np

import robosuite as suite
# TODO: Figure out why PickPlace single with Sawyer is crashing on use_object_obs


def test_all_environments():

    envs = sorted(suite.environments.ALL_ENVS)

    for env_name in envs:
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
                env = suite.make(
                    env_name,
                    robots=robots,
                    env_configuration=config,
                    has_renderer=False,  # no on-screen renderer
                    has_offscreen_renderers=True,  # off-screen renderer is required for camera observations
                    ignore_done=True,  # (optional) never terminates episode
                    use_camera_obs=True,  # use camera observations
                    camera_heights=84,  # set camera height
                    camera_widths=84,  # set camera width
                    camera_names="agentview",  # use "agentview" camera
                    use_object_obs=False,  # no object feature when training on pixels
                    reward_shaping=True,  # (optional) using a shaping reward
                )
            else:
                if robot_name == "Baxter":
                    continue
                robots = robot_name
                env = suite.make(
                    env_name,
                    robots=robots,
                    has_renderer=False,  # no on-screen renderer
                    has_offscreen_renderers=True,  # off-screen renderer is required for camera observations
                    ignore_done=True,  # (optional) never terminates episode
                    use_camera_obs=True,  # use camera observations
                    camera_heights=84,  # set camera height
                    camera_widths=84,  # set camera width
                    camera_names="agentview",  # use "agentview" camera
                    use_object_obs=False,  # no object feature when training on pixels
                    reward_shaping=True,  # (optional) using a shaping reward
                )
            print("Testing env: {} with robots {} with config {}...".format(env_name, robots, config))

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

                assert "image" in obs
                assert obs["image"].shape == (84, 84, 3)

                assert "object-state" not in obs

                action = np.random.uniform(action_min, action_max)
                obs, reward, done, info = env.step(action)

    # Tests passed!
    print("All environment tests passed successfully!")


if __name__ == "__main__":

    test_all_environments()
