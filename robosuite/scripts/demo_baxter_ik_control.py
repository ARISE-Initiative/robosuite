"""End-effector control for bimanual Baxter robot.

This script shows how to use inverse kinematics solver from Bullet
to command the end-effectors of two arms of the Baxter robot.
"""

import os
import numpy as np

import robosuite
from robosuite.wrappers import IKWrapper


if __name__ == "__main__":

    # initialize a Baxter environment
    env = robosuite.make(
        "BaxterLift",
        ignore_done=True,
        has_renderer=True,
        gripper_visualization=True,
        use_camera_obs=False,
    )
    env = IKWrapper(env)

    obs = env.reset()

    # rotate the gripper so we can see it easily
    env.set_robot_joint_positions([
        0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00,
        0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00,
    ])

    bullet_data_path = os.path.join(robosuite.models.assets_root, "bullet_data")

    def robot_jpos_getter():
        return np.array(env._joint_positions)

    for t in range(100000):
        omega = 2 * np.pi / 1000.
        A = 5e-4
        dpos_right = np.array([A * np.cos(omega * t), 0, A * np.sin(omega * t)])
        dpos_left = np.array([A * np.sin(omega * t), A * np.cos(omega * t), 0])
        dquat = np.array([0, 0, 0, 1])
        grasp = 0.
        action = np.concatenate([dpos_right, dquat, dpos_left, dquat, [grasp, grasp]])

        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            break
