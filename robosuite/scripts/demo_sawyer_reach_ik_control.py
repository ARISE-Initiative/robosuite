"""
End-effector control Sawyer robot.
"""

import os
import numpy as np

import robosuite
from robosuite.wrappers import IKWrapper


if __name__ == "__main__":

    # initialize a Baxter environment
    env = robosuite.make(
        "SawyerLift",
        ignore_done=True,
        has_renderer=False,
        gripper_visualization=False,
        use_camera_obs=False,
    )
    env = IKWrapper(env)

    obs = env.reset()

    # rotate the gripper so we can see it easily
    env.set_robot_joint_positions([
        0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00,
    ])

    bullet_data_path = os.path.join(robosuite.models.assets_root, "bullet_data")

    def robot_jpos_getter():
        return np.array(env._joint_positions)

    robot_states = []
    omega = 2 * np.pi / 1000.
    A = 3e-3

    for t in range(100):
        dpos = np.array([0, A * np.cos(omega * t), A * np.sin(omega * t)])
        dquat = np.array([0, 0, 0, 1])
        grasp = 0.
        action = np.concatenate([dpos, dquat, [grasp]])

        obs, reward, done, info = env.step(action)
        robot_states.append(obs['robot-state'])
        # env.render()

        if done:
            break

    np.save('real_robot_states.npy', robot_states)

