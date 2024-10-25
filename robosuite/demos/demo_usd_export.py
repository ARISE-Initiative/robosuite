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
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
from robosuite.controllers.composite.composite_controller import WholeBody
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

        # Get the newest action
        input_ac_dict = device.input2action()

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)

        env.step(env_action)
        env.render()
        exp.update_scene(data, scene_option=scene_option)

    exp.add_light(pos=[0, 0, 0], intensity=2000, obj_name="dome_light", light_type="dome")

    exp.save_scene(filetype="usd")

    # cleanup for end of data collection episodes
    env.close()
