"""sensor corruption demo.

This script provides an example of using the Observables functionality to implement a corrupted sensor
(corruption + delay).
Images will be rendered in a delayed fashion, such that the user will have seemingly delayed actions

This is a modified version of the demo_device_control teleoperation script.

Example:
    $ python demo_sensor_corruption.py --environment Stack --robots Panda --delay 0.05 --corruption 5.0 --toggle-corruption-on-grasp
"""

import sys
import numpy as np
import pygame
import argparse

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.utils.observables import create_uniform_sampled_delayer, create_gaussian_noise_corrupter
from robosuite.wrappers import VisualizationWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right",
                        help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-corruption-on-grasp", action="store_true",
                        help="Toggle corruption ON / OFF on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.5, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.5, help="How much to scale rotation user inputs")
    parser.add_argument("--delay", type=float, default=0.04, help="average delay to use (sec)")
    parser.add_argument("--corruption", type=float, default=20.0, help="Scale of corruption to use (std dev)")
    parser.add_argument("--camera", type=str, default="agentview", help="Name of camera to render")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()

    screen = pygame.display.set_mode((args.width, args.height))

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == 'ik':
        controller_name = 'IK_POSE'
    elif args.controller == 'osc':
        controller_name = 'OSC_POSE'
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

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

    # Add delay
    corrupter = create_gaussian_noise_corrupter(mean=0.0, std=args.corruption, low=0, high=255)
    delayer = create_uniform_sampled_delayer(min_delay=max(0, args.delay - 0.025), max_delay=args.delay + 0.025)
    sampling_rate = 10.
    attributes = ["corrupter", "delayer", "sampling_rate"]
    modifiers = [corrupter, delayer, sampling_rate]

    def modify_image_obs(attrs, mods):
        for attr, mod in zip(attrs, mods):
            env.modify_observable(
                observable_name=f"{args.camera}_image",
                attribute=attr,
                modifier=mod,
            )

    # Modify image observable
    modify_image_obs(attrs=attributes, mods=modifiers)

    # Setup corruption mapping for toggling ON / OFF
    corrupter_mappings = [
        [None, None],           # 0 maps to no corruption / delay
        modifiers[:2],          # 1 maps to corruption + delay
    ]
    corrupter_names = attributes[:2]

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        # Define wrapper method for keyboard callback that only uses the key
        pygame.key.set_repeat(20)
        on_press = lambda key: device.on_press(None, ord(chr(key).capitalize()), None, None, None)
        on_release = lambda key: device.on_release(None, ord(chr(key).capitalize()), None, None, None)

    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    while True:
        # Reset the environment
        obs = env.reset()

        # Setup corruption toggling (1 is ON, 0 is OFF)
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
                device=device,
                robot=active_robot,
                active_arm=args.arm,
                env_configuration=args.config
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
                    modify_image_obs(attrs=corrupter_names, mods=corrupter_mappings[corruption_mode])
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
                    print("Error: Unsupported arm specified -- "
                          "must be either 'right' or 'left'! Got: {}".format(args.arm))
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space, so trim the action space to be the action dim
                action = action[:env.action_dim]

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                # Handle keyboard events if appropriate
                if args.device == "keyboard":
                    if event.type == pygame.KEYDOWN:
                        on_press(event.key)
                    elif event.type == pygame.KEYUP:
                        on_release(event.key)

            # read camera observation
            im = np.flip(obs[args.camera + "_image"].transpose((1, 0, 2)), 1).astype(np.int)
            pygame.pixelcopy.array_to_surface(screen, im)
            pygame.display.update()
