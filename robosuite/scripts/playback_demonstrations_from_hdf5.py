"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/SawyerPickPlace/
"""

import argparse
import json
import os
import random

import h5py
import numpy as np

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml

import imageio

import robosuite.utils.macros as macros
macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )

    parser.add_argument(
        "--write-video",
        action="store_true",
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    env = robosuite.make(
        **env_info,
        has_renderer=not args.write_video,
        has_offscreen_renderer=args.write_video,
        ignore_done=True,
        use_camera_obs=args.write_video,
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,  # set camera height
        camera_widths=128,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    video_images = []
    for ep in demos:
        # print("Playing back random episode... (press ESC to quit)")

        # # select an episode randomly
        # ep = random.choice(demos)

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        if not args.write_video:
            env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        if args.use_actions:

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                obs, _, _, _ = env.step(action)
                if not args.write_video:
                    env.render()
                else:
                    video_images.append(obs['agentview_image'])

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

        else:

            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                if not args.write_video:
                    env.render()

    if args.write_video:
        output_path = '../vis/kitchen.mp4'
        video_writer = imageio.get_writer(output_path, fps=5)
        for i_img in range(len(video_images)):
            if i_img % 3 == 0:
                video_writer.append_data(video_images[i_img])
        video_writer.close()

    f.close()
