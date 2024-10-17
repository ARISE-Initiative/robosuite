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

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


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

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
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
    count = 0
    while True:
        # Set active robot
        active_robot = env.robots[device.active_robot]
        prev_gripper_actions = all_prev_gripper_actions[device.active_robot]

        arm = device.active_arm
        # Check if we have gripper actions for the active arm
        arm_using_gripper = f"{arm}_gripper" in all_prev_gripper_actions[device.active_robot]
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
        action_dict = prev_gripper_actions.copy()
        arm_actions = input_action[:6]
        if active_robot.is_mobile:
            # arm_actions = np.concatenate([arm_actions, ])

            # Decide if it's one arm or two arms. The number of arms can be decided based on the attribute `arms` of the robot.
            arm_actions = input_action[: 6 * len(env.robots[0].arms)].copy()

            if "GR1" in env.robots[0].name:
                # "relative" actions by default for now
                action_dict = {
                    "gripper0_left_grip_site_pos": input_action[:3] * 0.1,
                    "gripper0_left_grip_site_axis_angle": input_action[3:6],
                    "gripper0_right_grip_site_pos": np.zeros(3),
                    "gripper0_right_grip_site_axis_angle": np.zeros(3),
                    "left_gripper": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    "right_gripper": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                }
            elif "Tiago" in env.robots[0].name and args.composite_controller == "WHOLE_BODY_IK":
                action_dict = {
                    "right_gripper": np.array([0.0]),
                    "left_gripper": np.array([0.0]),
                    "gripper0_left_grip_site_pos": np.array([-0.4189254, 0.22745755, 1.0597]) + input_action[:3] * 0.05,
                    "gripper0_left_grip_site_axis_angle": np.array([-2.1356914, 2.50323857, -2.45929076]),
                    "gripper0_right_grip_site_pos": np.array([-0.41931295, -0.22706004, 1.0566]),
                    "gripper0_right_grip_site_axis_angle": np.array([-1.26839518, 1.15421975, 0.99332174]),
                }
            else:
                action_dict = {}
                base_action = input_action[-5:-2]
                torso_action = input_action[-2:-1]

                right_action = [0.0] * 5
                right_action[0] = 0.0

                action_dict.update(
                    {
                        arm: arm_actions,
                        active_robot.base: base_action,
                        # active_robot.head: base_action,
                        # active_robot.torso: base_action
                        # active_robot.torso: torso_action
                    }
                )
            if arm_using_gripper:
                action_dict[f"{arm}_gripper"] = np.repeat(input_action[6:7], active_robot.gripper[arm].dof)
                prev_gripper_actions[f"{arm}_gripper"] = np.repeat(input_action[6:7], active_robot.gripper[arm].dof)
            action = active_robot.create_action_vector(action_dict)
            mode_action = input_action[-1]

            if mode_action > 0:
                active_robot.enable_parts(base=True, right=True, left=True, torso=True)
            else:
                active_robot.enable_parts(base=False, right=True, left=True, torso=False)
        else:
            action_dict.update({arm: arm_actions})
            if arm_using_gripper:
                action_dict[f"{arm}_gripper"] = np.repeat(input_action[6:7], active_robot.gripper[arm].dof)
                prev_gripper_actions[f"{arm}_gripper"] = np.repeat(input_action[6:7], active_robot.gripper[arm].dof)
            action = active_robot.create_action_vector(action_dict)

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = action
        env_action = np.concatenate(env_action)
        env.step(env_action)
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
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_IK') or json file (see robosuite/controllers/config for examples)",
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
    composite_controller_config = load_composite_controller_config(controller=args.controller, robot=args.robots[0])

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": composite_controller_config,
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
        collect_human_trajectory(env, device, args.arm, args.config)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
