"""
Dumps video of the modality specified from iGibson renderer.
"""

import argparse

import imageio
import matplotlib.cm
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.renderers import load_renderer_config
from robosuite.utils.input_utils import *

macros.IMAGE_CONVENTION = "opencv"


def segmentation_to_rgb(seg, max_classes):
    cmap = matplotlib.cm.get_cmap("jet")
    color_list = np.array([cmap(i / max_classes) for i in range(max_classes)])
    return (color_list[seg] * 255).astype(np.uint8)


def normalize_depth(depth):
    # min max normalize depth
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    return (depth * 255).astype(np.uint8)


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
        help="Modality to render. Could be set to `depth`, `normal`, `segmentation` or `rgb`",
    )
    parser.add_argument("--video-path", type=str, default="/tmp/video.mp4", help="Path to video file")
    parser.add_argument(
        "--segmentation-level",
        default=None,
        help="`instance`, `class`, or `element`. Can only be set when modality is `segmentation`",
    )
    args = parser.parse_args()

    # sanity check.
    if args.vision_modality != "segmentation" and args.segmentation_level is not None:
        raise ValueError("`segmentation-level` can only be set when `vision_modality` is `segmentation`")

    # default segmentation level is element
    if args.vision_modality == "segmentation":
        args.segmentation_level = "element"

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
    config = load_renderer_config("igibson")

    if args.vision_modality == "rgb":
        config["vision_modalities"] = ["rgb"]
    if args.vision_modality == "segmentation":
        config["vision_modalities"] = ["seg"]
        config["msaa"] = False
    if args.vision_modality == "depth":
        config["vision_modalities"] = ["3d"]
    if args.vision_modality == "normal":
        config["vision_modalities"] = ["normal"]

    config["camera_obs"] = True
    config["render_mode"] = "headless"

    # import pdb; pdb.set_trace();

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        camera_names="frontview",
        camera_segmentations=args.segmentation_level,
        renderer="igibson",
        renderer_config=config,
    )
    env.reset()

    video_writer = imageio.get_writer(args.video_path, fps=20)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(100):
        action = 0.5 * np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)

        if args.vision_modality == "rgb":
            video_img = obs[f"frontview_image"]
        if args.vision_modality == "depth":
            video_img = obs[f"frontview_depth"]
            video_img = normalize_depth(video_img)
        if args.vision_modality == "normal":
            video_img = obs[f"frontview_normal"]
        if args.vision_modality == "segmentation":
            video_img = obs[f"frontview_seg"]
            # max class count can change w.r.t segmentation type.
            if args.segmentation_level == "element":
                max_class_count = env.viewer.max_elements
            if args.segmentation_level == "class":
                max_class_count = env.viewer.max_elements
            if args.segmentation_level == "instance":
                max_class_count = env.viewer.max_elements
            video_img = segmentation_to_rgb(video_img, max_class_count)

        video_writer.append_data(video_img)

        if i % 5 == 0:
            print("Step #{} / 100".format(i))

    print("Done.")
    print(f"Dumped file at location {args.video_path}")
