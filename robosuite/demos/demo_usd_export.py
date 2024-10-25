""" Exports a USD file corresponding to the collected trajectory.

The USD (Universal Scene Description) file format allows users to save
trajectories such that they can be rendered in external renderers such
as Omniverse or Blender, offering higher quality rendering. To view the
USD file, open your renderer (must support USD) and import the USD file.
Start the animation in your renderer to view the full trajectory.

***IMPORTANT***: If you are using mujoco version 3.1.1, please make sure
that you also have numpy < 2 installed in your environment. Failure to do
so may result in incorrect renderings. 
"""
import argparse

import mujoco
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
from robosuite.utils.usd import exporter
from robosuite.wrappers import VisualizationWrapper

if mujoco.__version__ == "3.1.1" and np.__version__[0] == "2":
    ROBOSUITE_DEFAULT_LOGGER.warning("If using mujoco==3.1.1, please use numpy < 2 for rendering with USD.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="default", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="BASIC", help="Choice of controller. Can be 'ik' or 'osc'")
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
    else:
        args.config = None

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
        hard_reset=False,
    )

    model = env.sim.model._model
    data = env.sim.data._data

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup = [0, 1, 0, 0, 0, 0]

    exp = exporter.USDExporter(model=model, camera_names=["frontview"])

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    env.reset()
    cam_id = 0
    num_cam = len(env.sim.model.camera_names)
    env.render()

    # Initialize variables that should the maintained between resets
    last_grasp = 0

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
        input_action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            active_end_effector=args.arm,
            env_configuration=args.config,
        )

        # If action is none, then this a reset so we should break
        if input_action is None:
            break

        # Run environment step
        action_dict = prev_gripper_actions.copy()
        arm_actions = input_action[:6].copy()
        if active_robot.is_mobile:
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
            elif "Tiago" in env.robots[0].name and args.controller == "WHOLE_BODY_IK":
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
                active_robot.enable_parts(base=True, right=True, left=True, torso=True)
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
        exp.update_scene(data, scene_option=scene_option)

    exp.add_light(pos=[0, 0, 0], intensity=2000, obj_name="dome_light", light_type="dome")

    exp.save_scene(filetype="usd")

    # cleanup for end of data collection episodes
    env.close()
