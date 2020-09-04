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

import os
import h5py
import argparse
import random
import numpy as np
import json

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default=os.path.join(
            robosuite.models.assets_root, "demonstrations/1592855346_302028"
        ),
    )
    parser.add_argument(
        "--use-actions", 
        action='store_true',
    )
    parser.add_argument(
        "--visualize-gripper",
        action='store_true',
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    env = robosuite.make(
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualizations=args.visualize_gripper,
        reward_shaping=True,
        control_freq=100,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    while True:
        print("Playing back random episode... (press ESC to quit)")

        # # select an episode randomly
        ep = random.choice(demos)

        # read the model xml, using the metadata stored in the attribute for this episode
        model_file = f["data/{}".format(ep)].attrs["model_file"]
        model_path = os.path.join(demo_path, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()

        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)].value

        if args.use_actions:

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            if not args.visualize_gripper:
                # We make the gripper site invisible
                robot = env.robots[0]
                env.sim.model.site_rgba[robot.eef_site_id] = np.zeros(4)
                env.sim.model.site_rgba[robot.eef_cylinder_id] = np.zeros(4)
            env.sim.forward()

            # load the actions and play them back open-loop
            joint_torques = f["data/{}/joint_torques".format(ep)].value
            actions = np.array(f["data/{}/actions".format(ep)].value)
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                env.step(action)
                env.render()

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    assert(np.all(np.equal(states[j + 1], state_playback)))

        else:

            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                if not args.visualize_gripper:
                    # We make the gripper site invisible
                    robot = env.robots[0]
                    env.sim.model.site_rgba[robot.eef_site_id] = np.zeros(4)
                    env.sim.model.site_rgba[robot.eef_cylinder_id] = np.zeros(4)

                env.sim.forward()
                env.render()

    f.close()
