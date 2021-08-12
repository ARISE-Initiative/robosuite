import os
import h5py
import argparse
import random
import numpy as np
import json

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml

import robosuite.utils.transform_utils as T

def rotate_camera(env, pos, quat, point=(0, 0, 0.8), axis=(0, 0, 1), angle=1.423):

    camera_pos = pos
    camera_rot = T.quat2mat(quat)

    rad = np.pi * angle / 180.0
    R = T.rotation_matrix(rad, axis, point=point)
    camera_pose = np.zeros((4, 4))
    camera_pose[:3, :3] = camera_rot
    camera_pose[:3, 3] = camera_pos
    camera_pose = R @ camera_pose

    pos, quat = camera_pose[:3, 3], T.mat2quat(camera_pose[:3, :3])
    return pos, quat

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
             "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'"
    ),
    parser.add_argument(
        "--use-actions", 
        action='store_true',
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")

    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_args = json.loads(f["data"].attrs["env_args"])

    env = robosuite.make(
        env_name,
        robots = env_args['env_kwargs']['robots'],
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        renderer="nvisii",
        width=500,
        height=500,
        spp=256,
        video_mode=True,
    )

    demos = list(f["data"].keys())

    env.reset()

    camera_pos = np.array([1.5, 0, 1.5])
    camera_quat = np.array([-1, 0, 0, 0])

    print("Playing back random episode... (press ESC to quit)")

    inc = 20
    i = 0
    image_count = 1

    ep = demos[4]

    # read the model xml, using the metadata stored in the attribute for this episode
    model_xml = f["data/{}".format(ep)].attrs["model_file"]

    env.reset()
    xml = postprocess_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()

    env.step(np.zeros(16))

    states = f["data/{}/states".format(ep)][()]

    if args.use_actions:
        # load the initial state
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        # load the actions and play them back open-loop
        actions = np.array(f["data/{}/actions".format(ep)][()])
        num_actions = actions.shape[0]

        for j, action in enumerate(actions):
            env.step(action)
            env.render()
            # print('rendered image... ' + str(image_count))
            image_count += 1

            i += 1

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
            env.step(np.zeros(16))

            env.render()
            # print('rendered image... ' + str(image_count))
            image_count += 1
            i += 1

            camera_pos, camera_quat = rotate_camera(env, camera_pos, camera_quat)
            env.set_camera_pos_quat(camera_pos, camera_quat)

    print('done')
    env.close()
