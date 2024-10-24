"""
Convenience script to make a video out of initial environment 
configurations. This can be a useful debugging tool to understand
what different sampled environment configurations look like.
"""

import argparse

import imageio
import numpy as np

import robosuite as suite
from robosuite.controllers import load_part_controller_config
from robosuite.utils.input_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # camera to use for generating frames
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
    )

    # number of frames in output video
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
    )

    # path to output video
    parser.add_argument(
        "--output",
        type=str,
        default="reset.mp4",
    )

    args = parser.parse_args()
    camera_name = args.camera
    num_frames = args.frames
    output_path = args.output

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
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

    # Load the controller
    options["controller_configs"] = load_part_controller_config(default_controller="OSC_POSE")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    # write a video
    video_writer = imageio.get_writer(output_path, fps=5)
    for i in range(num_frames):
        env.reset()
        video_img = env.sim.render(height=512, width=512, camera_name=camera_name)[::-1]
        env.step(np.zeros_like(env.action_spec[0]))
        video_writer.append_data(video_img)
    video_writer.close()
