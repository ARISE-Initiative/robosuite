"""
Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

import argparse
import imageio
import numpy as np

from robosuite import make


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Stack")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
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
            frame = obs["image"][::-1]
            writer.append_data(frame)
            print("Saving frame #{}".format(i))

        if done:
            break

    writer.close()
