"""Sensor Corruption Demo.

This script provides an example of using the Observables functionality to implement a corrupted sensor
(corruption + delay).
Images will be rendered in a delayed fashion, such that the user will have seemingly delayed actions

This is a modified version of the demo_device_control teleoperation script.

Example:
    $ python demo_sensor_corruption.py --environment Stack --robots Panda --delay 0.05 --corruption 5.0 --toggle-corruption-on-grasp
"""

import argparse
import sys

import cv2
import numpy as np

import robosuite as suite
from robosuite.utils.input_utils import input2action
from robosuite.utils.observables import Observable, create_gaussian_noise_corrupter, create_uniform_sampled_delayer
from robosuite.wrappers import VisualizationWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="default", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument(
        "--toggle-corruption-on-grasp", action="store_true", help="Toggle corruption ON / OFF on gripper action"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--delay", type=float, default=0.04, help="average delay to use (sec)")
    parser.add_argument("--corruption", type=float, default=20.0, help="Scale of corruption to use (std dev)")
    parser.add_argument("--camera", type=str, default="agentview", help="Name of camera to render")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    env = suite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        use_camera_obs=True,
        use_object_obs=True,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Set shared settings
    attributes = ["corrupter", "delayer", "sampling_rate"]
    corruption_mode = 1  # 1 is corruption = ON, 0 is corruption = OFF
    obs_settings = {}

    # Function to easily modify observable on the fly
    def modify_obs(obs_name, attrs, mods):
        for attr, mod in zip(attrs, mods):
            env.modify_observable(
                observable_name=obs_name,
                attribute=attr,
                modifier=mod,
            )

    # Add image corruption and delay
    image_sampling_rate = 10.0
    image_obs_name = f"{args.camera}_image"
    image_corrupter = create_gaussian_noise_corrupter(mean=0.0, std=args.corruption, low=0, high=255)
    image_delayer = create_uniform_sampled_delayer(min_delay=max(0, args.delay - 0.025), max_delay=args.delay + 0.025)
    image_modifiers = [image_corrupter, image_delayer, image_sampling_rate]

    # Initialize settings
    modify_obs(obs_name=image_obs_name, attrs=attributes, mods=image_modifiers)

    # Add entry for the corruption / delay settings in dict
    obs_settings[image_obs_name] = {
        "attrs": attributes[:2],
        "mods": lambda: image_modifiers[:2] if corruption_mode else [None, None],
    }

    # Add proprioception corruption and delay
    proprio_sampling_rate = 20.0
    proprio_obs_name = f"{env.robots[0].robot_model.naming_prefix}joint_pos"
    joint_limits = env.sim.model.jnt_range[env.robots[0]._ref_joint_indexes]
    joint_range = joint_limits[:, 1] - joint_limits[:, 0]
    proprio_corrupter = create_gaussian_noise_corrupter(mean=0.0, std=joint_range / 50.0)
    curr_proprio_delay = 0.0
    tmp_delayer = create_uniform_sampled_delayer(
        min_delay=max(0, (args.delay - 0.025) / 2), max_delay=(args.delay + 0.025) / 2
    )

    # Define delayer to synchronize delay between ground truth and corrupted sensors
    def proprio_delayer():
        global curr_proprio_delay
        curr_proprio_delay = tmp_delayer()
        return curr_proprio_delay

    # Define function to convert raw delay time to actual sampling delay (in discrete timesteps)
    def calculate_proprio_delay():
        base = env.model_timestep
        return base * round(curr_proprio_delay / base) if corruption_mode else 0.0

    proprio_modifiers = [proprio_corrupter, proprio_delayer, proprio_sampling_rate]

    # We will create a separate "ground truth" delayed proprio observable to track exactly
    # how much corruption we're getting in real time
    proprio_sensor = env._observables[proprio_obs_name]._sensor
    proprio_ground_truth_obs_name = f"{proprio_obs_name}_ground_truth"
    observable = Observable(
        name=proprio_ground_truth_obs_name,
        sensor=proprio_sensor,
        delayer=lambda: curr_proprio_delay,
        sampling_rate=proprio_sampling_rate,
    )

    # Add this observable
    env.add_observable(observable)

    # We also need to set the normal joint pos observable to be active (not active by default)
    env.modify_observable(observable_name=proprio_obs_name, attribute="active", modifier=True)

    # Initialize settings
    modify_obs(obs_name=proprio_obs_name, attrs=attributes, mods=proprio_modifiers)

    # Add entry for the corruption / delay settings in dict
    obs_settings[proprio_obs_name] = {
        "attrs": attributes[:2],
        "mods": lambda: proprio_modifiers[:2] if corruption_mode else [None, None],
    }
    obs_settings[proprio_ground_truth_obs_name] = {
        "attrs": [attributes[1]],
        "mods": lambda: [lambda: curr_proprio_delay] if corruption_mode else [None],
    }

    # Setup printing options for numbers
    np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    while True:
        # Reset the environment
        obs = env.reset()

        # Reset corruption mode
        corruption_mode = 1

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        while True:
            # Set active robot
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

            # Get the newest action
            action, grasp = input2action(
                device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config
            )

            # If action is none, then this a reset so we should break
            if action is None:
                break

            # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
            # toggle arm control and / or corruption if requested
            if last_grasp < 0 < grasp:
                if args.switch_on_grasp:
                    args.arm = "left" if args.arm == "right" else "right"
                if args.toggle_corruption_on_grasp:
                    # Toggle corruption and update observable
                    corruption_mode = 1 - corruption_mode
                    for obs_name, settings in obs_settings.items():
                        modify_obs(obs_name=obs_name, attrs=settings["attrs"], mods=settings["mods"]())
            # Update last grasp
            last_grasp = grasp

            # Fill out the rest of the action space if necessary
            rem_action_dim = env.action_dim - action.size
            if rem_action_dim > 0:
                # Initialize remaining action space
                rem_action = np.zeros(rem_action_dim)
                # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
                if args.arm == "right":
                    action = np.concatenate([action, rem_action])
                elif args.arm == "left":
                    action = np.concatenate([rem_action, action])
                else:
                    # Only right and left arms supported
                    print(
                        "Error: Unsupported arm specified -- "
                        "must be either 'right' or 'left'! Got: {}".format(args.arm)
                    )
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space, so trim the action space to be the action dim
                action = action[: env.action_dim]

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)

            # Calculate and print out stats for proprio observation
            observed_value = obs[proprio_obs_name]
            ground_truth_delayed_value = obs[proprio_ground_truth_obs_name]
            print(
                f"Observed joint pos: {observed_value}, "
                f"Corruption: {observed_value - ground_truth_delayed_value}, "
                f"Delay: {calculate_proprio_delay():.3f} sec"
            )

            # read camera observation
            im = np.flip(obs[args.camera + "_image"][..., ::-1], 0).astype(np.uint8)

            cv2.imshow("offscreen render", im)
            cv2.waitKey(1)
