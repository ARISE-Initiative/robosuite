"""
Test all environments with random policies.
"""
import numpy as np

import robosuite as suite


def test_all_environments():

    envs = sorted(suite.environments.ALL_ENVS)

    for env_name in envs:

        # create an environment for learning on pixels
        env = suite.make(
            env_name,
            has_renderer=False,  # no on-screen renderer
            has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
            ignore_done=True,  # (optional) never terminates episode
            use_camera_obs=True,  # use camera observations
            camera_height=84,  # set camera height
            camera_width=84,  # set camera width
            camera_name="agentview",  # use "agentview" camera
            use_object_obs=False,  # no object feature when training on pixels
            reward_shaping=True,  # (optional) using a shaping reward
        )
        print("Testing env: {}...".format(env_name))

        obs = env.reset()

        # get action range
        action_min, action_max = env.action_spec
        assert action_min.shape == action_max.shape

        # run 10 random actions
        for _ in range(10):

            assert "robot-state" in obs
            assert obs["robot-state"].ndim == 1

            assert "image" in obs
            assert obs["image"].shape == (84, 84, 3)

            assert "object-state" not in obs

            action = np.random.uniform(action_min, action_max)
            obs, reward, done, info = env.step(action)


if __name__ == "__main__":

    test_all_environments()
