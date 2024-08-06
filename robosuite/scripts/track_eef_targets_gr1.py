"""
A script test whole body control GR1 eef tracking.
python robosuite/scripts/track_eef_targets.py --robot GR1FixedLowerBody --controller IK_POSE --arm arms_body --camera frontview --device keyboard --use-whole-body-controller --input-file recordings/gr1_eef_targets_gr1_w_hands_from_avp_preprocessor.pkl
"""

import argparse
import datetime
import json
import os
import time
from glob import glob
from typing import Dict, List

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.transform_utils import quat2axisangle


def collect_human_trajectory(env, device, arm, env_configuration, end_effector: str = "right",
    history: Dict[str, List] = None
):
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
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()
    env.robots[0].print_action_info_dict()
    # Loop until we get a reset from the input or the task completes

    count = 0
    while True:
        # Set active robot
        active_robot = env.robots[0]  # if env_configuration == "bimanual" else env.robots[arm == "left"]

        # # Get the newest action
        # input_action, grasp = input2action(
        #     device=device, robot=active_robot, active_arm=arm, active_end_effector=end_effector, env_configuration=env_configuration
        # )

        # # If action is none, then this a reset so we should break
        # if input_action is None:
        #     break

        # Run environment step
        arm_actions = None
        if env.robots[0].is_mobile:
            if int(count) >= len(history['right_eef_pos']):
                print(f"Finished looping through the history. Count: {count}")
                break
            # count = 0
            left_target_pos = history['left_eef_pos'][int(count)]
            right_target_pos = history['right_eef_pos'][int(count)]
            left_target_ori = history['left_eef_quat_wxyz'][int(count)]
            right_target_ori = history['right_eef_quat_wxyz'][int(count)]

            # convert to axis angle using quat2axisangle
            left_target_aa = quat2axisangle(np.roll(left_target_ori, -1))
            right_target_aa = quat2axisangle(np.roll(right_target_ori, -1))

            arm_actions = np.concatenate([right_target_pos, right_target_aa, left_target_pos, left_target_aa])

            input_action = np.zeros(8)
            # add 
            base_action = input_action[-5:-2]
            torso_action = input_action[-2:-1]

            right_action = [0.0] * 5
            right_action[0] = 0.0

            action_dict =  {
                "gripper0_left_grip_site_pos": left_target_pos,
                "gripper0_left_grip_site_axis_angle": left_target_aa,
                "gripper0_right_grip_site_pos": right_target_pos,
                "gripper0_right_grip_site_axis_angle": right_target_aa,
                "left_gripper": np.repeat(input_action[6:7], env.robots[0].gripper[end_effector].dof),
                "right_gripper": np.repeat(input_action[7:8], env.robots[0].gripper[end_effector].dof),
            }
            action = env.robots[0].create_action_vector(action_dict)

            mode_action = input_action[-1]

            if mode_action > 0:
                env.robots[0].enable_parts(base=True, right=True, left=True, torso=True)
            else:
                env.robots[0].enable_parts(base=False, right=True, left=True, torso=True)
        else:
            arm_actions = input_action
            action = env.robots[0].create_action_vector({arm: arm_actions[:-1], f"{end_effector}_gripper": arm_actions[-1:]})

        env.step(action)
        env.render()

        count += 1

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
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

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
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
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


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
        "--config", type=str, default="default", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument(
        "--use-whole-body-controller", action="store_true", help="Use the whole body controller for the arms and body"
    )
    # pkl file
    parser.add_argument("--input-file", type=str, default=None, help="Path to the pkl file containing the controller config")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument(
        "--renderer",
        type=str,
        default="mujoco",
        help="Use the Nvisii viewer (Nvisii), OpenCV viewer (mujoco), or Mujoco's builtin interactive viewer (mjviewer)",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    with open("robosuite/controllers/config/default_gr1.json") as f:
        gr1_controller_config = json.load(f)

    # naming of type is weird
    composite_controller_config = {
        "type": "WHOLE_BODY",
        # so it's more whole body controller specific configs, rather
        # than composite controller specific configs ...
        "composite_controller_specific_configs": {
            "ref_name": ["gripper0_right_grip_site", "gripper0_left_grip_site"],
            "interpolation": None,
            "robot_name": args.robots[0],
            "individual_part_names": ["torso", "head", "right", "left"],
            "max_dq": 4,
            "nullspace_joint_weights": {
                "robot0_torso_waist_yaw": 100.0,
                "robot0_torso_waist_pitch": 100.0,
                "robot0_torso_waist_roll": 500.0,
                "robot0_l_shoulder_pitch": 4.0,
                "robot0_r_shoulder_pitch": 4.0,
                "robot0_l_shoulder_roll": 3.0,
                "robot0_r_shoulder_roll": 3.0,
                "robot0_l_shoulder_yaw": 2.0,
                "robot0_r_shoulder_yaw": 2.0,
            },
            "ik_pseudo_inverse_damping": 5e-2,
            "ik_integration_dt": 1e-1,
            "ik_max_dq": 4.0,
            "ik_max_dq_torso": 0.2,
            "ik_input_rotation_repr": "axis_angle",
            "ik_debug": True,
        },
        "default_controller_configs_part_names": ["right_gripper", "left_gripper"],
        "body_parts": {
            "right": gr1_controller_config,
            "left": gr1_controller_config,
        }
    }
    if not args.use_whole_body_controller:
        composite_controller_config = None

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
        "composite_controller_configs": composite_controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
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
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    import pickle
    with open(args.input_file, 'rb') as f:
        history = pickle.load(f)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.config, history=history)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)