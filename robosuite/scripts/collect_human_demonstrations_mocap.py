"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob
from typing import List

import h5py
import numpy as np

import mujoco
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils import transform_utils

"""
Command for running GR1 + mocap (only works for GR1 b/c of site_names assumption)

python robosuite/scripts/collect_human_demonstrations.py --robot GR1FixedLowerBody --camera frontview --device keyboard --composite-controller WHOLE_BODY_IK --renderer mjviewer --use-mocap

Need to Esc; Tab; Scroll to Group Enable, then press Group 2. Then, double click on the mocap cube and Ctrl+left or right click.

"""


def set_target(sim, target_pos=None, target_mat=None, mocap_name: str = "target"):
    mocap_id = sim.model.body(mocap_name).mocapid[0]
    if target_pos is not None:
        sim.data.mocap_pos[mocap_id] = target_pos
    if target_mat is not None:
        # convert mat to quat
        target_quat = np.empty(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.reshape(9, 1))
        sim.data.mocap_quat[mocap_id] = target_quat

def get_target(sim, mocap_name: str = "target"):
    mocap_id = sim.model.body(mocap_name).mocapid[0]
    target_pos = np.copy(sim.data.mocap_pos[mocap_id])
    target_quat = np.copy(sim.data.mocap_quat[mocap_id])
    target_mat = np.empty(9)
    mujoco.mju_quat2Mat(target_mat, target_quat)
    target_mat = target_mat.reshape(3, 3)
    return target_pos, target_mat

def collect_human_trajectory(env, device, arm, env_configuration, end_effector: str = "right", use_mocap: bool = False):
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

    if use_mocap:
        site_names: List[str] = env.robots[0].composite_controller.joint_action_policy.site_names
        right_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site_names[0])]
        right_mat = env.sim.data.site_xmat[env.sim.model.site_name2id(site_names[0])]
        left_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site_names[1])]
        left_mat = env.sim.data.site_xmat[env.sim.model.site_name2id(site_names[1])]

        # # Add mocap bodies if they don't exist
        # if "right_eef_target" not in env.sim.model.body_names:
        #     add_mocap_body(env.sim.model, "right_eef_target", right_pos)
        # if "left_eef_target" not in env.sim.model.body_names:
        #     add_mocap_body(env.sim.model, "left_eef_target", left_pos)

        set_target(env.sim, right_pos, right_mat, "right_eef_target")
        set_target(env.sim, left_pos, left_mat, "left_eef_target")

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()
    env.robots[0].print_action_info_dict()
    # Loop until we get a reset from the input or the task completes

    while True:
        # Set active robot
        active_robot = env.robots[0]  # if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        input_action, grasp = input2action(
            device=device, robot=active_robot, active_arm=arm, active_end_effector=end_effector, env_configuration=env_configuration
        )

        # If action is none, then this a reset so we should break
        if input_action is None:
            break

        # Run environment step
        if env.robots[0].is_mobile:
            arm_actions = input_action[:12].copy() if "bimanual" in env.robots[0].name else input_action[:6].copy()
            if "GR1" in env.robots[0].name:
                # "relative" actions by default for now
                action_dict = {
                    'gripper0_left_grip_site_pos': input_action[:3] * 0.1, 
                    'gripper0_left_grip_site_axis_angle': input_action[3:6], 
                    'gripper0_right_grip_site_pos': np.zeros(3), 
                    'gripper0_right_grip_site_axis_angle': np.zeros(3), 
                    'left_gripper': np.array([0., 0., 0., 0., 0., 0.]), 
                    'right_gripper': np.array([0., 0., 0., 0., 0., 0.])
                }

                if use_mocap:
                    right_pos, right_mat = get_target(env.sim, "right_eef_target")
                    left_pos, left_mat = get_target(env.sim, "left_eef_target")
                    # convert mat to quat wxyz
                    right_quat_wxyz = np.empty(4)
                    left_quat_wxyz = np.empty(4)
                    mujoco.mju_mat2Quat(right_quat_wxyz, right_mat.reshape(9, 1))
                    mujoco.mju_mat2Quat(left_quat_wxyz, left_mat.reshape(9, 1))

                    # convert to quat xyzw
                    right_quat_xyzw = np.roll(right_quat_wxyz, -1)
                    left_quat_xyzw = np.roll(left_quat_wxyz, -1)
                    # convert to axis angle
                    right_axis_angle = transform_utils.quat2axisangle(right_quat_xyzw)
                    left_axis_angle = transform_utils.quat2axisangle(left_quat_xyzw)

                    action_dict['gripper0_left_grip_site_pos'] = left_pos
                    action_dict['gripper0_right_grip_site_pos'] = right_pos
                    action_dict['gripper0_left_grip_site_axis_angle'] = left_axis_angle
                    action_dict['gripper0_right_grip_site_axis_angle'] = right_axis_angle

            elif "Tiago" in env.robots[0].name:
                action_dict = {
                    'right_gripper': np.array([0.]), 
                    'left_gripper': np.array([0.]), 
                    'gripper0_left_grip_site_pos': np.array([-0.4189254 ,  0.22745755,  1.0597]) + input_action[:3] * 0.05, 
                    'gripper0_left_grip_site_axis_angle': np.array([-2.1356914 ,  2.50323857, -2.45929076]), 
                    'gripper0_right_grip_site_pos': np.array([-0.41931295, -0.22706004,  1.0566]), 
                    'gripper0_right_grip_site_axis_angle': np.array([-1.26839518,  1.15421975,  0.99332174]),
                }
            else:
                action_dict = {}
                base_action = input_action[-5:-2]
                torso_action = input_action[-2:-1]

                right_action = [0.0] * 5
                right_action[0] = 0.0
                action_dict = {
                    arm: arm_actions,
                    f"{end_effector}_gripper": np.repeat(input_action[6:7], env.robots[0].gripper[end_effector].dof),
                    env.robots[0].base: base_action,
                    # env.robots[0].head: base_action,
                    # env.robots[0].torso: base_action
                    # env.robots[0].torso: torso_action
                }

            action = env.robots[0].create_action_vector(action_dict)
            mode_action = input_action[-1]

            if mode_action > 0:
                env.robots[0].enable_parts(base=True, right=True, left=True, torso=True)
            else:
                if "GR1FixedLowerBody" in env.robots[0].name or "Tiago" in env.robots[0].name:
                    env.robots[0].enable_parts(base=False, right=True, left=True, torso=True)
                else:
                    env.robots[0].enable_parts(base=False, right=True, left=True, torso=False)
        else:
            arm_actions = input_action
            action = env.robots[0].create_action_vector({arm: arm_actions[:-1], f"{end_effector}_gripper": arm_actions[-1:]})

        env.step(action)
        env.render()

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
        "--composite-controller", type=str, default=None, help="Choice of composite controller. Can be 'NONE' or 'WHOLE_BODY_IK'"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument(
        "--renderer",
        type=str,
        default="mujoco",
        help="Use the Nvisii viewer (Nvisii), OpenCV viewer (mujoco), or Mujoco's builtin interactive viewer (mjviewer)",
    )
    # add use mocap option
    parser.add_argument(
        "--use-mocap", action="store_true"
    )
    args = parser.parse_args()
    if args.use_mocap:
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)
    composite_controller_config = load_composite_controller_config(default_controller=args.composite_controller, robot=args.robots[0])

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

        device = Keyboard(env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.config, end_effector="right", use_mocap=args.use_mocap)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)