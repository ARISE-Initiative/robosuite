import os
import h5py
import argparse
import random
import numpy as np
import json
import cv2

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite.utils.camera_utils import *

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

def to_str(arr):
    s = ''
    for i in arr:
        s += str(i) + ' '

    return s[:-1]

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
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    demos = list(f["data"].keys())

    env.reset()

    print("Playing back random episode... (press ESC to quit)")

    ep = demos[4]

    # # read the model xml, using the metadata stored in the attribute for this episode

    cam_tree = ET.Element("camera", attrib={"name": "frontview"})
    CAMERA_NAME = cam_tree.get("name")

    env.reset()

    model_xml = f["data/{}".format(ep)].attrs["model_file"]
    xml = postprocess_model_xml(model_xml)
    
    env.reset_from_xml_string(xml)

    camera_mover = CameraMover(
        env=env,
        camera=CAMERA_NAME,
    )

    # Make sure we're using the camera that we're modifying
    camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
    
    env.sim.reset()

    env.step(np.zeros(16))

    states = f["data/{}/states".format(ep)][()]

    video = cv2.VideoWriter('mujoco_rendering.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (1280, 720))

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
            obs, _, _, _ = env.step(np.zeros(16))

            img = env.sim.render(height=720, width=1280, camera_name="frontview")[::-1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.imwrite('images/image0.png', img)

            pos, quat = camera_mover.get_camera_pose()

            camera_mover.rotate_camera(point=(0, 0, -2), axis=(0., 1., 0), angle=1.423)

            video.write(cv2.imread('images/image0.png'))

            # xml, camera_pos, camera_quat = postprocess_model_xml(xml, {"frontview": {"pos": to_str(camera_pos), "quat": to_str(camera_quat)}})
            # env.reset_from_xml_string(xml)

            # print(camera_pos, camera_quat)
            # env.set_camera_pos_quat(camera_pos, camera_quat)

    print('done')
    env.close()