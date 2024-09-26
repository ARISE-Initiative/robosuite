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
import mujoco
import numpy as np

import robosuite as suite
import robosuite.examples.third_party_controller.mink_controller
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5, get_target, set_target
from robosuite.utils import transform_utils
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

"""
Command for running GR1 + mocap (only works for GR1 b/c of site_names assumption)

python robosuite/examples/third_party_controller/collect_human_demonstrations_mocap.py --environment Lift --robots GR1FixedLowerBody --device keyboard --camera frontview --custom-controller-config robosuite/examples/third_party_controller/default_mink_ik_gr1.json

Need to Esc; Tab; Scroll to Group Enable, then press Group 2. Then, double click on the mocap cube and Ctrl+left or right click.

"""


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
            device=device,
            robot=active_robot,
            active_arm=arm,
            active_end_effector=end_effector,
            env_configuration=env_configuration,
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
                    "robot0_l_eef_site_pos": input_action[:3] * 0.1,
                    "robot0_l_eef_site_axis_angle": input_action[3:6],
                    "robot0_r_eef_site_pos": np.zeros(3),
                    "robot0_r_eef_site_axis_angle": np.zeros(3),
                    "left_gripper": np.array([0.0] * env.robots[0].gripper["left"].dof),
                    "right_gripper": np.array([0.0] * env.robots[0].gripper["right"].dof),
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

                    action_dict["robot0_l_eef_site_pos"] = left_pos
                    action_dict["robot0_r_eef_site_pos"] = right_pos
                    action_dict["robot0_l_eef_site_axis_angle"] = left_axis_angle
                    action_dict["robot0_r_eef_site_axis_angle"] = right_axis_angle
                    action_dict["gripper0_left_grip_site_pos"] = left_pos
                    action_dict["gripper0_right_grip_site_pos"] = right_pos
                    action_dict["gripper0_left_grip_site_axis_angle"] = left_axis_angle
                    action_dict["gripper0_right_grip_site_axis_angle"] = right_axis_angle

            elif "Tiago" in env.robots[0].name:
                action_dict = {
                    "right_gripper": np.array([0.0]),
                    "left_gripper": np.array([0.0]),
                    "robot0_l_eef_site_pos": np.array([-0.4189254, 0.22745755, 1.0597]) + input_action[:3] * 0.05,
                    "robot0_l_eef_site_axis_angle": np.array([-2.1356914, 2.50323857, -2.45929076]),
                    "robot0_r_eef_site_pos": np.array([-0.41931295, -0.22706004, 1.0566]),
                    "robot0_r_eef_site_axis_angle": np.array([-1.26839518, 1.15421975, 0.99332174]),
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
            action = env.robots[0].create_action_vector(
                {arm: arm_actions[:-1], f"{end_effector}_gripper": arm_actions[-1:]}
            )

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
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use the Nvisii viewer (Nvisii), OpenCV viewer (mujoco), or Mujoco's builtin interactive viewer (mjviewer)",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(controller=args.controller, robot=args.robots[0])

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

        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)

    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.config, use_mocap=True)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
