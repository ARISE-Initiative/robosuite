"""
A script to teleop a robot using mink and mj-viewer GUI + mocap + mouse:

python robosuite/examples/third_party_controller/teleop_mink.py --controller robosuite/examples/third_party_controller/default_mink_ik_gr1.json --robots GR1FixedLowerBody --device mjgui
"""

import argparse
import os
import time
from typing import List, Tuple

import mujoco
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

# mink-related imports
from robosuite.devices.mjgui import MJGUI
from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK
from robosuite.wrappers import DataCollectionWrapper
from robosuite.devices.keyboard import Keyboard


def set_target_pose(sim, target_pos=None, target_mat=None, mocap_name: str = "target"):
    mocap_id = sim.model.body(mocap_name).mocapid[0]
    if target_pos is not None:
        sim.data.mocap_pos[mocap_id] = target_pos
    if target_mat is not None:
        # convert mat to quat
        target_quat = np.empty(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.reshape(9, 1))
        sim.data.mocap_quat[mocap_id] = target_quat


def get_target_pose(sim, mocap_name: str = "target") -> Tuple[np.ndarray]:
    mocap_id = sim.model.body(mocap_name).mocapid[0]
    target_pos = np.copy(sim.data.mocap_pos[mocap_id])
    target_quat = np.copy(sim.data.mocap_quat[mocap_id])
    target_mat = np.empty(9)
    mujoco.mju_quat2Mat(target_mat, target_quat)
    target_mat = target_mat.reshape(3, 3)
    return target_pos, target_mat


def collect_human_trajectory(env, device, arm, env_configuration, end_effector: str = "right"):
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

    site_names: List[str] = env.robots[0].composite_controller.joint_action_policy.site_names
    right_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site_names[0])]
    right_mat = env.sim.data.site_xmat[env.sim.model.site_name2id(site_names[0])]
    left_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site_names[1])]
    left_mat = env.sim.data.site_xmat[env.sim.model.site_name2id(site_names[1])]

    set_target_pose(env.sim, right_pos, right_mat, "right_eef_target")
    set_target_pose(env.sim, left_pos, left_mat, "left_eef_target")

    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[device.active_robot]
        prev_gripper_actions = all_prev_gripper_actions[device.active_robot]

        arm = device.active_arm
        # Check if we have gripper actions for the active arm
        arm_using_gripper = f"{arm}_gripper" in all_prev_gripper_actions[device.active_robot]
        # Get the newest action
        action_dict = device.input2action()

        for gripper_name, gripper in active_robot.gripper.items():
            action_dict[f"{gripper_name}_gripper"] = np.zeros(gripper.dof)  # what's the 'do nothing' action for all grippers?

        if arm_using_gripper:
            prev_gripper_actions[f"{arm}_gripper"] = action_dict[f"{arm}_gripper"]

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)

        env.step(env_action)
        env.render()

    # cleanup for end of data collection episodes
    env.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
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
        help="Choice of controller json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

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
        renderer="mjviewer",
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # wrap the environment with data collection wrapper
    tmp_directory = "teleop-mink-data/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)  # need this wrapper's reset for UI mocap dragging to work
    if args.device == "keyboard":
        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "mjgui":
        device = MJGUI(env=env)
    while True:
        collect_human_trajectory(env, device, args.arm, args.config)