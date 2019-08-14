"""
This file implements a wrapper for controlling the robot through end effector
movements instead of joint velocities. This is useful in learning pipelines
that want to output actions in end effector space instead of joint space.
"""

import os
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import Wrapper


class IKWrapper(Wrapper):
    env = None

    def __init__(self, env, action_repeat=1):
        """
        Initializes the inverse kinematics wrapper.
        This wrapper allows for controlling the robot through end effector
        movements instead of joint velocities.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            action_repeat (int): Determines the number of times low-level joint
                control actions will be commanded per high-level end effector
                action. Higher values will allow for more precise control of
                the end effector to the commanded targets.
        """
        super().__init__(env)
        if self.env.mujoco_robot.name == "sawyer":
            from robosuite.controllers import SawyerIKController

            self.controller = SawyerIKController(
                bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        elif self.env.mujoco_robot.name == "baxter":
            from robosuite.controllers import BaxterIKController

            self.controller = BaxterIKController(
                bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        else:
            raise Exception(
                "Only Sawyer and Baxter robot environments are supported for IK "
                "control currently."
            )

        self.action_repeat = action_repeat
        self.action_spec = self.env.action_spec
        self.init_quat = np.array([-0.02224347, -0.99944383, 0.01488818, -0.01988979])

    def set_robot_joint_positions(self, positions):
        """
        Overrides the function to set the joint positions directly, since we need to notify
        the IK controller of the change.
        """
        self.env.set_robot_joint_positions(positions)
        self.controller.sync_state()

    def _robot_jpos_getter(self):
        """
        Helper function to pass to the ik controller for access to the
        current robot joint positions.
        """
        return np.array(self.env._joint_positions)

    def reset(self):
        ret = super().reset()
        ret['robot-state'] = np.concatenate([ret['robot-state'], self.controller.ik_robot_target_pos])
        self.controller.sync_state()
        return ret

    def step(self, action):
        """
        Move the end effector(s) according to the input control.

        Args:
            action (numpy array): The array should have the corresponding elements.
                0-2: The desired change in end effector position in x, y, and z.
                3-6: The desired change in orientation, expressed as a (x, y, z, w) quaternion.
                    Note that this quaternion encodes a relative rotation with respect to the
                    current gripper orientation. If the current rotation is r, this corresponds
                    to a quaternion d such that r * d will be the new rotation.
                *: Controls for gripper actuation.

                Note: When wrapping around a Baxter environment, the indices 0-6 inidicate the
                right hand. Indices 7-13 indicate the left hand, and the rest (*) are the gripper
                inputs (first right, then left).
        """
        # Make change in orientation none
        action = np.hstack([action[:3], np.array([0, 0, 0, 1]), action[3]])

        input_1 = self._make_input(action[:7], self.init_quat)
        if self.env.mujoco_robot.name == "sawyer":
            velocities = self.controller.get_control(**input_1)
            low_action = np.concatenate([velocities, action[7:]])
        elif self.env.mujoco_robot.name == "baxter":
            input_2 = self._make_input(action[7:14], self.env._left_hand_quat)
            velocities = self.controller.get_control(input_1, input_2)
            low_action = np.concatenate([velocities, action[14:]])
        else:
            raise Exception(
                "Only Sawyer and Baxter robot environments are supported for IK "
                "control currently."
            )

        # keep trying to reach the target in a closed-loop
        for i in range(self.action_repeat):
            ret = self.env.step(low_action)
            velocities = self.controller.get_control()
            if self.env.mujoco_robot.name == "sawyer":
                low_action = np.concatenate([velocities, action[7:]])
            else:
                low_action = np.concatenate([velocities, action[14:]])
            
        # This is to append target position into obs space
        ret[0]['robot-state'] = np.concatenate([ret[0]['robot-state'], self.controller.ik_robot_target_pos])
        return ret

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat.
        """
        action = np.hstack([action[:3], [0, 0, 0, 1], action[3]])
        return {
            "dpos": action[:3],
            # IK controller takes an absolute orientation in robot base frame
            "rotation": T.quat2mat(T.quat_multiply(old_quat, action[3:7])),
        }

    @property
    def dof(self):
        """
        Returns size of (x, y, z) delta + gripper
        """
        return 4

