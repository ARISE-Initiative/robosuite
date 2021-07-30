"""
Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

import argparse
import imageio
import numpy as np

import robosuite.utils.macros as macros
from robosuite import make

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Stack")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--camera", type=str, default="agentview", help="Name of camera to render")
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()

    # initialize an environment with offscreen renderer  
    env = make(
        args.environment,
        args.robots,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
    )

    obs = env.reset()
    ndim = env.action_dim

    # create a video writer with imageio
    writer = imageio.get_writer(args.video_path, fps=20)

    frames = []
    for i in range(args.timesteps):

        # run a uniformly random agent
        action = 0.5 * np.random.randn(ndim)
        obs, reward, done, info = env.step(action)

        # dump a frame from every K frames
        if i % args.skip_frame == 0:
            frame = obs[args.camera + "_image"]
            writer.append_data(frame)
            print("Saving frame #{}".format(i))

        if done:
            break

    writer.close()
