import os
import h5py
import argparse
import random
import numpy as np
import json
from PIL import Image as im

import sys
sys.path.append('../renderers/nvisii/')

from nvisii_render_wrapper import NViSIIWrapper

import robosuite as suite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite.controllers import load_controller_config

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
    env_info = json.loads(f["data"].attrs["env_info"])

    options = {}
    options["env_name"] = "Stack"
    options["robots"] = "Panda"
    options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

    env = NViSIIWrapper(
        env = suite.make(
                **options,
                reward_shaping=True,
                has_renderer=False,           # no on-screen renderer
                has_offscreen_renderer=False, # no off-screen renderer
                ignore_done=True,
                use_object_obs=True,          # use object-centric feature
                use_camera_obs=False,         # no camera observations
                control_freq=10, 
            ),
        img_path='images',
        spp=128,          # samples per pixel for images
        use_noise=False,  # add noise to the images
        debug_mode=False, # interactive setting
        video_mode=True,
        video_directory='videos/',
        video_name='robosuite_video_1.mp4',
        video_fps=60,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    env.reset()

    inc = 20
    i = 0
    image_count = 1

    print("Playing back random episode... (press ESC to quit)")

    ep = random.choice(demos)

    # read the model xml, using the metadata stored in the attribute for this episode
    model_xml = f["data/{}".format(ep)].attrs["model_file"]
    
    env.reset()
    xml = postprocess_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    # env.viewer.set_camera(0)

    env.step([0, 0, 0, 0, 0, 0, 0])

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
            env.step(action)

            # if i % inc == 0:
            env.render()
            print('rendered image... ' + str(image_count))
            image_count += 1

            i+=1

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
            env.step([0, 0, 0, 0, 0, 0, 0])
            
            # if i % inc == 0:
            env.render()
            print('rendered image... ' + str(image_count))
            image_count += 1
            
            # img_arr = env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
            # data = im.fromarray(img_arr)
            # data.save("images/image_" + str(image_count) + ".png")
                
            i+=1

    print('done')

    env.close()