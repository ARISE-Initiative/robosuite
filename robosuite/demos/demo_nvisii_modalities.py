"""
Dumps video of the modality specified from iGibson renderer.
"""

import argparse

import imageio
import matplotlib.cm
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.renderers import load_renderer_config
from robosuite.utils import macros
from robosuite.utils.input_utils import *

if __name__ == "__main__":

    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vision-modality",
        type=str,
        default="rgb",
        help="Modality to render. Could be set to `depth`, `normal`, `segmentation`, or `rgb`",
    )

    args = parser.parse_args()

    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

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

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

    # change renderer config
    config = load_renderer_config("nvisii")

    if args.vision_modality == "rgb":
        config["vision_modalities"] = None
    if args.vision_modality == "segmentation":
        config["vision_modalities"] = "segmentation"
    if args.vision_modality == "depth":
        config["vision_modalities"] = "depth"
    if args.vision_modality == "normal":
        config["vision_modalities"] = "normal"

    env = suite.make(
        **options,
        has_renderer=False,  # no on-screen renderer
        has_offscreen_renderer=False,  # no off-screen renderer
        ignore_done=True,
        use_camera_obs=False,  # no camera observations
        control_freq=20,
        renderer="nvisii",
        renderer_config=config,
        camera_segmentations="element" if config["vision_modalities"] == "segmentation" else None,
    )

    env.reset()

    low, high = env.action_spec

    timesteps = 300
    for i in range(timesteps):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)

        if i % 100 == 0:
            env.render()

    env.close_renderer()
    print("Done.")
