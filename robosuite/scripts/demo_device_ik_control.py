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

Example:
    $ python demo_device_ik_control.py --environment SawyerPickPlaceCan

"""

import argparse
import numpy as np
import sys

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="SawyerPickPlaceCan")
    parser.add_argument("--device", type=str, default="keyboard")
    args = parser.parse_args()

    env = robosuite.make(
        args.environment,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=100
    )

    # enable controlling the end effector directly instead of using joint velocities
    env = IKWrapper(env)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard()
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse()
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    while True:
        obs = env.reset()
        env.viewer.set_camera(camera_id=2)
        env.render()

        # rotate the gripper so we can see it easily
        if env.mujoco_robot.name == 'sawyer':
            env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        elif env.mujoco_robot.name == 'panda':
            env.set_robot_joint_positions([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])
        else:
            print("Error: Script supported for Sawyer and Panda robots only!")
            sys.exit()

        device.start_control()
        while True:
            state = device.get_controller_state()
            dpos, rotation, grasp, reset = (
                state["dpos"],
                state["rotation"],
                state["grasp"],
                state["reset"],
            )
            if reset:
                break

            # convert into a suitable end effector action for the environment
            current = env._right_hand_orn

            # relative rotation of desired from current
            drotation = current.T.dot(rotation)
            dquat = T.mat2quat(drotation)

            # map 0 to -1 (open) and 1 to 0 (closed halfway)
            grasp = grasp - 1.

            action = np.concatenate([dpos, dquat, [grasp]])
            obs, reward, done, info = env.step(action)
            env.render()