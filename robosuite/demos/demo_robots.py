import argparse

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.wrappers import VisualizationWrapper


def bimanual_check(robot):
    bimanual_robots = ["Baxter", "Tiago", "GR1", "G1", "H1", "PR2", "Yumi", "Aloha"]
    for br in bimanual_robots:
        if br in robot:
            return True
    return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", type=str, nargs="+", default=ROBOT_CLASS_MAPPING.keys())
    parser.add_argument("--controller", type=str, default="BASIC", help="Choice of controller. Can be 'ik' or 'osc'")
    args = parser.parse_args()

    for robot in args.robots:
        print(f"{robot} demo...")

        # Check if we're using a multi-armed environment
        if "TwoArm" in args.environment and not bimanual_check(robot):
            robots = [robot, robot]
        else:
            robots = [robot]

        # Get controller config
        controller_config = load_composite_controller_config(
            controller=args.controller,
            robot=robot,
        )

        # Create argument configuration
        config = {
            "env_name": args.environment,
            "robots": robots,
            "controller_configs": controller_config,
        }

        # Create environment
        env = suite.make(
            **config,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=False,
        )

        # Wrap this environment in a visualization wrapper
        env = VisualizationWrapper(env, indicator_configs=None)

        env.reset()
        low, high = env.action_spec

        for i in range(200):
            action = np.random.uniform(low, high)
            env.step(action)
            env.render()

        env.close()
