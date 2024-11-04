import argparse
import json
import time

import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.utils.input_utils import *

MAX_FR = 25  # max frame rate for running simluation


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def display_mjv_options():
    def print_command(char, info):
        char += " " * (30 - len(char))
        print("{}\t{}".format(char, info))

    print("")
    print("Quick list of some of the interactive keyboard options:")
    print("")
    print_command("Keyboard Input", "Functionality")
    print_command("Esc", "switch to free camera")
    print_command("]", "toggle between camera views")
    print_command("Shift + Tab", "visualize joints and control values")
    print_command("W", "visualize wireframe")
    print_command("C", "visualize contact points")
    print_command("F1", "basic GUI help")
    print_command("Tab", "view more toggleable options")
    print_command("Tab + Right Hold", "view keyboard shortcuts for more toggleable options")
    print("")
    print("")


if __name__ == "__main__":

    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    """

    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", type=str, default="mjviewer", help="Valid options include mujoco, and nvisii")

    args = parser.parse_args()
    renderer = args.renderer

    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))
    # If a humanoid environment has been chosen, choose humanoid robots
    elif "Humanoid" in options["env_name"]:
        options["robots"] = choose_robots(use_humanoids=True)
    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    env = suite.make(
        **options,
        has_renderer=True,  # no on-screen renderer
        has_offscreen_renderer=False,  # no off-screen renderer
        ignore_done=True,
        use_camera_obs=False,  # no camera observations
        control_freq=20,
        renderer=renderer,
    )

    env.reset()

    low, high = env.action_spec

    # do visualization
    for i in range(10000):
        start = time.time()

        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()

        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)

    env.close_renderer()
    print("Done.")
