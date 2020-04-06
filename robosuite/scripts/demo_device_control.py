"""Teleoperate robot with keyboard or SpaceMouse.

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

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).

    Notes:
        OSC also tends to be more efficient since IK relies on backend pybullet sim
        IK maintains initial orientation of robot env while OSC automatically initializes with gripper facing downwards

Example:
    $ python demo_device_control.py --environment SawyerPickPlaceCan --controller ik

"""

import argparse
import numpy as np
import os
import json

import robosuite
import robosuite.utils.transform_utils as T


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="SawyerLift")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (if bimanual) 'right' or 'left'")
    parser.add_argument("--controller", type=str, default="ik", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
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

    env = robosuite.make(
        args.environment,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=20,
        controller_config=controller_config
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
        env.viewer.set_camera(camera_id=2)
        env.render()

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

            # First process the raw drotation
            drotation = raw_drotation[[1,0,2]]
            if args.controller == 'ik':
                # If this is panda, want to flip y
                if env.mujoco_robot.name == 'panda':
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
            grasp = 1 if grasp else -1

            # Create action based on action space of individual robot
            if env.mujoco_robot.name == 'baxter':
                # Baxter takes double the action length
                nonactive = np.zeros(6)
                if args.controller == 'ik':
                    nonactive = np.concatenate([nonactive, [1]])
                if args.arm == 'right':
                    # Right control
                    action = np.concatenate([dpos, drotation, nonactive, [grasp], [-1]])
                elif args.arm == 'left':
                    # Left control
                    action = np.concatenate([nonactive, dpos, drotation, [-1], [grasp]])
                else:
                    # Only right and left arms supported
                    print("Error: Unsupported arm specified -- must be either 'right' or 'left'!")

            else:
                action = np.concatenate([dpos, drotation, [grasp]])

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            env.render()
