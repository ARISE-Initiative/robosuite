"""
Play random actions in an environment and render a video that demonstrates segmentation.
"""
import imageio
import colorsys
import random
import numpy as np
import matplotlib.cm as cm

import robosuite as suite
from robosuite.controllers import load_controller_config


def random_colors(N, bright=True):
    """
    Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py#L59
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    rstate = np.random.RandomState(seed=0)
    rstate.shuffle(colors)
    return colors

def segmentation_to_rgb(seg_im):
    """
    Helper function to visualize segmentations as RGB frames.
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """
    # ensure all values lie within [0, 255]
    seg_im = np.mod(seg_im, 256)

    # colors = random_colors(N=256, bright=True)
    # return (255. * colors[seg_im]).astype(np.uint8)

    # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
    rstate = np.random.RandomState(seed=1)
    inds = np.arange(256)
    rstate.shuffle(inds)

    # use @inds to map each geom ID to a color
    return (255. * cm.rainbow(inds[seg_im], 3)).astype(np.uint8)[..., :3]

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment and add it to options
    options["env_name"] = "Lift"
    options["robots"] = ["Panda"]

    # Choose controller
    controller_name = "OSC_POSE"

    # Choose camera
    camera = "agentview"

    # Choose segmentation type
    segmentation_level = "instance"         # Options are {instance, class, element}

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=camera,
        camera_segmentations=segmentation_level,
    )
    env.reset()

    video_writer = imageio.get_writer("/tmp/video.mp4", fps=20)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(300):
        action = 0.1 * np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)

        if i % 5 == 0:
            video_img = obs[f"{camera}_segmentation_{segmentation_level}"].squeeze(-1)[::-1]
            np.savetxt("/tmp/seg_{}.txt".format(i), video_img, fmt="%.2f")
            video_img = segmentation_to_rgb(video_img)
            video_writer.append_data(video_img)
            import json
            print("geom_id2name")
            print(json.dumps(env.sim.model._geom_id2name, indent=4))
            from PIL import Image
            image = Image.fromarray(video_img)
            image.save("/tmp/seg_{}.png".format(i))
            break

    video_writer.close()


