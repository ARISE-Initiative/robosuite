"""Teleoperate robot with keyboard or SpaceMouse.

***Choose user input option with the --device argument***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with Mac OS X, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

    SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
    We used the SpaceMouse Wireless in our experiments. The paper below used the same device
    to collect human demonstrations for imitation learning.

    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018

    Note:
        This current implementation only supports Mac OS X (Linux support can be added).
        Download and install the driver before running the script:
            https://www.3dconnexion.com/service/drivers.html

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).

    Notes:
        OSC also tends to be more efficient since IK relies on backend pybullet sim
        IK maintains initial orientation of robot env while OSC automatically initializes with gripper facing downwards


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "TwoArmPegInHole", "NutAssembly", etc.

    --robots: Robot(s) with which to perform the task. Can be any in {"Panda", "Sawyer", "Baxter"}. Note that the
        environments include sanity checks, such that a "TwoArm..." environment will only accept either a 2-tuple of
        robot names or a single bimanual robot name, according to the specified configuration (see below), and all
        other environments will only accept a single single-armed robot name

    --config: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies the robot
        configuration desired for the task. Options are {"bimanual", "single-arm-parallel", and "single-arm-opposed"}

            -"bimanual": Sets up the environment for a single bimanual robot. Expects a single bimanual robot name to
                be specified in the --robots argument

            -"single-arm-parallel": Sets up the environment such that two single-armed robots are stationed next to
                each other facing the same direction. Expects a 2-tuple of single-armed robot names to be specified
                in the --robots argument.

            -"single-arm-opposed": Sets up the environment such that two single-armed robots are stationed opposed from
                each other, facing each other from opposite directions. Expects a 2-tuple of single-armed robot names
                to be specified in the --robots argument.

    --arm: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies which of the
        multiple arm eef's to control. The other (passive) arm will remain stationary. Options are {"right", "left"}
        (from the point of view of the robot(s) facing against the viewer direction)

    --switch-on-click: Exclusively applicable and only should be specified for "TwoArm..." environments. If enabled,
        will switch the current arm being controlled every time the gripper input is pressed

    --toggle-camera-on-click: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc

    For two-arm bimanual environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Baxter --config bimanual --arm left --controller osc

    For two-arm multi single-arm robot environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config single-arm-parallel --controller osc


"""

import argparse
import numpy as np
import os
import json

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.models.robots import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="", help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-click", action="store_true", help="Switch gripper control on gripper click")
    parser.add_argument("--toggle-camera-on-click", action="store_true", help="Switch camera angle on gripper click")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.5, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.5, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Import controller config for EE IK or OSC (pos/ori)
    controller_config = None
    controller_path = None
    if args.controller == 'ik':
        controller_path = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/ee_ik.json')
    elif args.controller == 'osc':
        controller_path = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/ee_pos_ori.json')
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError
    try:
        with open(controller_path) as f:
            controller_config = json.load(f)
            if args.controller == 'osc':
                controller_config["max_action"] = 1
                controller_config["min_action"] = -1
                controller_config["control_delta"] = True
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
              "Please check filepath and try again.".format(controller_path))

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    if args.config != "":
        config["env_configuration"] = args.config

    # Create environment
    env = robosuite.make(
        **config,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualizations=True,
        reward_shaping=True,
        control_freq=20,
    )

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

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
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    while True:
        obs = env.reset()
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        last_grasp = 0

        device.start_control()
        while True:
            state = device.get_controller_state()
            # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
            #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
            #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
            dpos, rotation, raw_drotation, grasp, reset = (
                state["dpos"],
                state["rotation"],
                state["raw_drotation"],
                state["grasp"],
                state["reset"],
            )
            if reset:
                break

            # Set active robot
            active_robot = env.robots[0] if not args.config or args.config == "bimanual" else \
                env.robots[args.arm == "left"]

            # If the current grasp is active and last grasp is not (i.e.: grasping input just pressed),
            # toggle arm control and / or camera viewing angle if requested
            if grasp and not last_grasp:
                if args.switch_on_click:
                    args.arm = "left" if args.arm == "right" else "right"
                if args.toggle_camera_on_click:
                    cam_id = (cam_id + 1) % num_cam
                    env.viewer.set_camera(camera_id=cam_id)
            # Update last grasp
            last_grasp = grasp

            # First process the raw drotation
            drotation = raw_drotation[[1,0,2]]
            if args.controller == 'ik':
                # If this is panda, want to flip y
                if isinstance(active_robot.robot_model, Panda):
                    drotation[1] = -drotation[1]
                else:
                    # Flip x
                    drotation[0] = -drotation[0]
                # Scale rotation for teleoperation (tuned for IK)
                drotation *= 10
                dpos *= 5
                # relative rotation of desired from current eef orientation
                # IK expects quat, so also convert to quat
                drotation = T.mat2quat(T.euler2mat(drotation))

                # If we're using a non-forward facing configuration, need to adjust relative position / orientation
                if hasattr(env, "env_configuration"):
                    if env.env_configuration == "single-arm-opposed":
                        # Swap x and y for pos and flip x,y signs for ori
                        dpos = dpos[[1,0,2]]
                        drotation[0] = -drotation[0]
                        drotation[1] = -drotation[1]
                        if args.arm == "left":
                            # x pos needs to be flipped
                            dpos[0] = -dpos[0]
                        else:
                            # y pos needs to be flipped
                            dpos[1] = -dpos[1]

            elif args.controller == 'osc':
                # Flip z
                drotation[2] = -drotation[2]
                # Scale rotation for teleoperation (tuned for OSC)
                drotation *= 75
                dpos *= 200
            else:
                # No other controllers currently supported
                print("Error: Unsupported controller specified -- must be either ik or osc!")

            # map 0 to -1 (open) and map 1 to 1 (closed)
            grasp = [1] if grasp else [-1]

            # Check to make sure robot actually has a gripper and clear grasp action if it doesn't
            if hasattr(env, "env_configuration"):
                # This is a multi-arm robot
                if env.env_configuration == "bimanual":
                    # We should check the correct arm to see if it has a gripper
                    if not active_robot.has_gripper[args.arm]:
                        grasp = []
                else:
                    # We should check the correct robot to see if it has a gripper (assumes 0 = right, 1 = left)
                    if not active_robot.has_gripper:
                        grasp = []
            else:
                # This is a single-arm robot, simply check to see if it has a gripper
                if not active_robot.has_gripper:
                    grasp = []

            # Create action based on action space of individual robot
            action = np.concatenate([dpos, drotation, grasp])

            # Fill out the rest of the action space if necessary
            rem_action_dim = env.action_dim - action.size
            rem_action = np.zeros(rem_action_dim)
            # Make sure ik input isn't degenerate
            if rem_action_dim > 0 and args.controller == 'ik':
                rem_action[6] = 1
            if rem_action_dim > 0:
                # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
                if args.arm == "right":
                    action = np.concatenate([action, rem_action])
                elif args.arm == "left":
                    action = np.concatenate([rem_action, action])
                else:
                    # Only right and left arms supported
                    print("Error: Unsupported arm specified -- "
                          "must be either 'right' or 'left'! Got: {}".format(args.arm))

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            env.render()


