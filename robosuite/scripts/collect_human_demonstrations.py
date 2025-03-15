"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np
import sys
sys.path.append('')

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
import robosuite.utils.transform_utils as T

from robosuite.utils.env_utils import get_eef_quat, get_obs, get_target_quat, get_axisangle_error, get_eef_pos
from robosuite.utils.primitive_utils import inverse_scale_action, scale_action

def collect_human_trajectory(env, device, arm, env_configuration, only_yaw):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    step_cnt = 0
    while True:
        time.sleep(0.01)
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        )

        # If action is none, then this a reset so we should break
        if action is None:
            break
        assert not (only_yaw and args.controller == 'OSC_POSITION_YAW')
        if only_yaw:
            action_copy = action.copy()
            action[3:5] = np.array([0, 0])
            scaled_action = scale_action(env, action)
            obs = get_obs(env)
            cur_quat = get_eef_quat(obs)
            target_quat = get_target_quat(cur_quat, scaled_action[3:6])
            target_euler = T.mat2euler(T.quat2mat(target_quat), axes='rxyz')
            target_euler = np.concatenate(([np.pi, 0], target_euler[-1:]))
            target_quat = T.mat2quat(T.euler2mat(target_euler))
            axisangle_error = get_axisangle_error(cur_quat, target_quat)
            scaled_action = np.concatenate((scaled_action[:3], axisangle_error, scaled_action[-1:]))
            action = inverse_scale_action(env, scaled_action)
            try:
                assert np.linalg.norm(action[:3] - action_copy[:3]) < 1e-4
            except:
                print(np.linalg.norm(action[:3] - action_copy[:3]))

        if args.controller == 'OSC_POSITION_YAW':
            action = np.concatenate([action[:3], action[5:]])

        # TODO: limit xyz bound

        obs = get_obs(env)
        eef_pos = get_eef_pos(obs)
        info = env._get_skill_info()
        for pos_i in range(3):
            if eef_pos[pos_i] < env.data_eef_bounds[0][pos_i] and action[pos_i] < 0:
                action[pos_i] = 0
            elif eef_pos[pos_i] > env.data_eef_bounds[1][pos_i] and action[pos_i] > 0:
                action[pos_i] = 0
        # Run environment step
        obs, _, _, _ = env.step(action)
        step_cnt += 1
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 20 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    print("traj len", step_cnt)
    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, only_success):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point
    demo_cnt = 0
    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        is_success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            if dic["success"]:
                is_success = True
            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
        if not is_success and only_success:
            continue

        if len(states) == 0:
            continue
        demo_cnt += 1
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)


        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()


        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    try:
        # write dataset attributes (metadata)
        now = datetime.datetime.now()
        grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info
    except:
        assert len(os.listdir(directory)) == 1 and (not is_success and is_success)

    f.close()
    print("collected ndemo", demo_cnt)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument('--only-yaw', action='store_true', default=False)
    parser.add_argument('--only-success', action='store_true', default=False)
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    # t1, t2 = str(time.time()).split(".")
    # new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    new_dir = os.path.join(args.directory, "{}".format(args.environment))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.config, args.only_yaw)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, args.only_success)
