import argparse

import numpy as np

import robosuite as suite
import robosuite.utils.test_utils as tu
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.robot_composition_utils import create_composite_robot

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--robot", type=str, required=True)
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--grippers", nargs="+", type=str, default=["PandaGripper"])
    parser.add_argument("--env", type=str, default="Lift")

    args = parser.parse_args()

    name = f"Custom{args.robot}"
    create_composite_robot(name, base=args.base, robot=args.robot, grippers=args.grippers)
    controller_config = load_composite_controller_config(controller="BASIC", robot=name)

    tu.create_and_test_env(env="Lift", robots=name, controller_config=controller_config)
