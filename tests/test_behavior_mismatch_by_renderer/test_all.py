import numpy as np
import unittest

import robosuite as suite
from pathlib import Path
import itertools
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

cwd = Path.cwd()


def get_envs(env_name: str, robot: str):
    env_config = {"env_name": env_name, "robots": robot}

    share_configs = {
        **env_config,
        "has_renderer": False,
        "use_object_obs": True,
        "reward_shaping": True,
        "use_camera_obs": False,
    }
    # Notify user of which test we are currently on
    print(f"Testing env: {env_name} with robot: {robot}...")
    env: ManipulationEnv = suite.make(
        **share_configs,
        has_offscreen_renderer=False,
    )
    render_env: ManipulationEnv = suite.make(
        **share_configs,
        has_offscreen_renderer=True,
    )

    return env, render_env


class TestAllCombinations(unittest.TestCase):

    def test_env_robot_combinations(self):
        envs = sorted(suite.ALL_ENVIRONMENTS)
        robots = [
            "Panda",
            "Sawyer",
            "Baxter",
            "GR1",
        ]

        # for env_name in envs:
        #     for robot_name in robots:
        for env_name, robot_name in itertools.product(envs, robots):
            with self.subTest(env=env_name, robot=robot_name):
                # Create config dict
                env, render_env = get_envs(env_name, robot_name)
                env.reset()
                render_env.reset()

                for i in range(env.horizon):
                    action = np.random.rand(*env.action_spec[0].shape)
                    action = env.action_spec[0] * action + env.action_spec[
                        1] * (1 - action)
                    # input(action)
                    obs, reward, *_, info = env.step(action)

                    obs1, reward2, *_, info2 = render_env.step(action)
                    assert reward == reward2, f"Inconsistant reward Step {i:04d} {reward:6e} {reward2:6e}"

            with self.subTest(env=env_name, robot=robot_name):
                # Create config dict
                env, render_env = get_envs(env_name, robot_name)
                obs = env.reset()
                obs1 = render_env.reset()
                join_keys = set(obs.keys()).intersection(obs1.keys())

                for i in range(env.horizon):
                    key_consistancy = True
                    keys = []
                    action = np.random.rand(*env.action_spec[0].shape)
                    action = env.action_spec[0] * action + env.action_spec[
                        1] * (1 - action)
                    # input(action)
                    obs, reward, *_, info = env.step(action)

                    obs1, reward2, *_, info2 = render_env.step(action)
                    for key in join_keys:
                        key_bool = np.all(obs[key] == obs1[key])
                        if not key_bool:
                            # print(
                            #     f"Inconsistant obs term {key}: {obs[key]} {obs1[key]}"
                            # )
                            keys.append(key)
                        key_consistancy = key_consistancy and key_bool
                    assert key_consistancy, f"Inconsistant obs term Step {i:4d} {keys}"


if __name__ == '__main__':
    unittest.main()
