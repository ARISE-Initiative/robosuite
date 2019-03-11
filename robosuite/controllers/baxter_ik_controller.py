"""
NOTE: requires pybullet module.

Run `pip install pybullet==1.9.5`.
"""

import os
import numpy as np

try:
    import pybullet as p
except ImportError:
    raise Exception(
        "Please make sure pybullet is installed. Run `pip install pybullet==1.9.5`"
    )

import robosuite.utils.transform_utils as T
from robosuite.controllers import Controller


class BaxterIKController(Controller):
    """
    Inverse kinematics for the Baxter robot, using Pybullet and the urdf description
    files.
    """

    def __init__(self, bullet_data_path, robot_jpos_getter):
        """
        Args:
            bullet_data_path (str): base path to bullet data.

            robot_jpos_getter (function): function that returns the joint positions of
                the robot to be controlled as a numpy array. 
        """
        # Set up inverse kinematics
        self.robot_jpos_getter = robot_jpos_getter

        path = os.path.join(bullet_data_path, "baxter_description/urdf/baxter_mod.urdf")
        self.setup_inverse_kinematics(path)

        self.rest_joints = np.zeros(14)
        self.commanded_joint_positions = robot_jpos_getter()

        self.sync_state()

    def get_control(self, right=None, left=None):
        """
        Returns joint velocities to control the robot after the target end effector 
        positions and orientations are updated from arguments @left and @right.
        If no arguments are provided, joint velocities will be computed based
        on the previously recorded target.

        Args:
            left (dict): A dictionary to control the left end effector with these keys.

                dpos (numpy array): a 3 dimensional array corresponding to the desired
                    change in x, y, and z left end effector position.

                rotation (numpy array): a rotation matrix of shape (3, 3) corresponding
                    to the desired orientation of the left end effector.

            right (dict): A dictionary to control the left end effector with these keys.

                dpos (numpy array): a 3 dimensional array corresponding to the desired
                    change in x, y, and z right end effector position.

                rotation (numpy array): a rotation matrix of shape (3, 3) corresponding
                    to the desired orientation of the right end effector.

        Returns:
            velocities (numpy array): a flat array of joint velocity commands to apply
                to try and achieve the desired input control.


        """
        # Sync joint positions for IK.
        self.sync_ik_robot(self.robot_jpos_getter())

        # Compute new target joint positions if arguments are provided
        if (right is not None) and (left is not None):
            self.commanded_joint_positions = self.joint_positions_for_eef_command(
                right, left
            )

        # P controller from joint positions (from IK) to velocities
        velocities = np.zeros(14)
        deltas = self._get_current_error(
            self.robot_jpos_getter(), self.commanded_joint_positions
        )

        for i, delta in enumerate(deltas):
            velocities[i] = -2 * delta
        velocities = self.clip_joint_velocities(velocities)

        self.commanded_joint_velocities = velocities
        return velocities

        # For debugging purposes: set joint positions directly
        # robot.set_joint_positions(self.commanded_joint_positions)

    def sync_state(self):
        """
        Syncs the internal Pybullet robot state to the joint positions of the
        robot being controlled.
        """

        # sync IK robot state to the current robot joint positions
        self.sync_ik_robot(self.robot_jpos_getter())

        # make sure target pose is up to date
        pos_r, orn_r, pos_l, orn_l = self.ik_robot_eef_joint_cartesian_pose()

        self.ik_robot_target_pos_right = pos_r
        self.ik_robot_target_orn_right = orn_r
        self.ik_robot_target_pos_left = pos_l
        self.ik_robot_target_orn_left = orn_l

    def setup_inverse_kinematics(self, urdf_path):
        """
        This function is responsible for doing any setup for inverse kinematics.
        Inverse Kinematics maps end effector (EEF) poses to joint angles that
        are necessary to achieve those poses. 
        """

        # These indices come from the urdf file we're using
        self.effector_right = 27
        self.effector_left = 45

        # Use PyBullet to handle inverse kinematics.
        # Set up a connection to the PyBullet simulator.
        p.connect(p.DIRECT)
        p.resetSimulation()

        self.ik_robot = p.loadURDF(urdf_path, (0, 0, 0), useFixedBase=1)

        # Relevant joints we care about. Many of the joints are fixed and don't count, so
        # we need this second map to use the right ones.
        self.actual = [13, 14, 15, 16, 17, 19, 20, 31, 32, 33, 34, 35, 37, 38]

        self.num_joints = p.getNumJoints(self.ik_robot)
        n = p.getNumJoints(self.ik_robot)
        self.rest = []
        self.lower = []
        self.upper = []
        self.ranges = []
        for i in range(n):
            info = p.getJointInfo(self.ik_robot, i)
            # Retrieve lower and upper ranges for each relevant joint
            if info[3] > -1:
                self.rest.append(p.getJointState(self.ik_robot, i)[0])
                self.lower.append(info[8])
                self.upper.append(info[9])
                self.ranges.append(info[9] - info[8])

        # Simulation will update as fast as it can in real time, instead of waiting for
        # step commands like in the non-realtime case.
        p.setRealTimeSimulation(1)

    def sync_ik_robot(self, joint_positions, simulate=False, sync_last=True):
        """
        Force the internal robot model to match the provided joint angles.

        Args:
            joint_positions (list): a list or flat numpy array of joint positions.
            simulate (bool): If True, actually use physics simulation, else 
                write to physics state directly.
            sync_last (bool): If False, don't sync the last joint angle. This
                is useful for directly controlling the roll at the end effector.
        """
        num_joints = len(joint_positions)
        if not sync_last:
            num_joints -= 1
        for i in range(num_joints):
            if simulate:
                p.setJointMotorControl2(
                    self.ik_robot,
                    self.actual[i],
                    p.POSITION_CONTROL,
                    targetVelocity=0,
                    targetPosition=joint_positions[i],
                    force=500,
                    positionGain=0.5,
                    velocityGain=1.,
                )
            else:
                # Note that we use self.actual[i], and not i
                p.resetJointState(self.ik_robot, self.actual[i], joint_positions[i])

    def ik_robot_eef_joint_cartesian_pose(self):
        """
        Returns the current cartesian pose of the last joint of the ik robot with respect
        to the base frame as a (pos, orn) tuple where orn is a x-y-z-w quaternion.
        """
        out = []
        for eff in [self.effector_right, self.effector_left]:
            eef_pos_in_world = np.array(p.getLinkState(self.ik_robot, eff)[0])
            eef_orn_in_world = np.array(p.getLinkState(self.ik_robot, eff)[1])
            eef_pose_in_world = T.pose2mat((eef_pos_in_world, eef_orn_in_world))

            base_pos_in_world = np.array(
                p.getBasePositionAndOrientation(self.ik_robot)[0]
            )
            base_orn_in_world = np.array(
                p.getBasePositionAndOrientation(self.ik_robot)[1]
            )
            base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))
            world_pose_in_base = T.pose_inv(base_pose_in_world)

            eef_pose_in_base = T.pose_in_A_to_pose_in_B(
                pose_A=eef_pose_in_world, pose_A_in_B=world_pose_in_base
            )
            out.extend(T.mat2pose(eef_pose_in_base))

        return out

    def inverse_kinematics(
        self,
        target_position_right,
        target_orientation_right,
        target_position_left,
        target_orientation_left,
        rest_poses,
    ):
        """
        Helper function to do inverse kinematics for a given target position and
        orientation in the PyBullet world frame.

        Args:
            target_position_{right, left}: A tuple, list, or numpy array of size 3 for position.
            target_orientation_{right, left}: A tuple, list, or numpy array of size 4 for
                a orientation quaternion.
            rest_poses: A list of size @num_joints to favor ik solutions close by.

        Returns:
            A list of size @num_joints corresponding to the joint angle solution.
        """

        ndof = 48

        ik_solution = list(
            p.calculateInverseKinematics(
                self.ik_robot,
                self.effector_right,
                target_position_right,
                targetOrientation=target_orientation_right,
                restPoses=rest_poses[:7],
                lowerLimits=self.lower,
                upperLimits=self.upper,
                jointRanges=self.ranges,
                jointDamping=[0.7] * ndof,
            )
        )
        ik_solution2 = list(
            p.calculateInverseKinematics(
                self.ik_robot,
                self.effector_left,
                target_position_left,
                targetOrientation=target_orientation_left,
                restPoses=rest_poses[7:],
                lowerLimits=self.lower,
                upperLimits=self.upper,
                jointRanges=self.ranges,
                jointDamping=[0.7] * ndof,
            )
        )
        for i in range(8, 15):
            ik_solution[i] = ik_solution2[i]

        return ik_solution[1:]

    def bullet_base_pose_to_world_pose(self, pose_in_base):
        """
        Convert a pose in the base frame to a pose in the world frame.

        Args:
            pose_in_base: a (pos, orn) tuple.

        Returns:
            pose_in world: a (pos, orn) tuple.
        """
        pose_in_base = T.pose2mat(pose_in_base)

        base_pos_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[0])
        base_orn_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[1])
        base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))

        pose_in_world = T.pose_in_A_to_pose_in_B(
            pose_A=pose_in_base, pose_A_in_B=base_pose_in_world
        )
        return T.mat2pose(pose_in_world)

    def joint_positions_for_eef_command(self, right, left):
        """
        This function runs inverse kinematics to back out target joint positions
        from the provided end effector command.

        Same arguments as @get_control.

        Returns:
            A list of size @num_joints corresponding to the target joint angles.
        """

        dpos_right = right["dpos"]
        dpos_left = left["dpos"]
        self.target_pos_right = self.ik_robot_target_pos_right + np.array([0, 0, 0.913])
        self.target_pos_left = self.ik_robot_target_pos_left + np.array([0, 0, 0.913])
        self.ik_robot_target_pos_right += dpos_right
        self.ik_robot_target_pos_left += dpos_left

        rotation_right = right["rotation"]
        rotation_left = left["rotation"]
        self.ik_robot_target_orn_right = T.mat2quat(rotation_right)
        self.ik_robot_target_orn_left = T.mat2quat(rotation_left)

        # convert from target pose in base frame to target pose in bullet world frame
        world_targets_right = self.bullet_base_pose_to_world_pose(
            (self.ik_robot_target_pos_right, self.ik_robot_target_orn_right)
        )
        world_targets_left = self.bullet_base_pose_to_world_pose(
            (self.ik_robot_target_pos_left, self.ik_robot_target_orn_left)
        )

        # Empirically, more iterations aren't needed, and it's faster
        for _ in range(5):
            arm_joint_pos = self.inverse_kinematics(
                world_targets_right[0],
                world_targets_right[1],
                world_targets_left[0],
                world_targets_left[1],
                rest_poses=self.robot_jpos_getter(),
            )
            self.sync_ik_robot(arm_joint_pos, sync_last=True)

        return arm_joint_pos

    def _get_current_error(self, current, set_point):
        """
        Returns an array of differences between the desired joint positions and current
        joint positions. Useful for PID control.

        Args:
            current: the current joint positions.
            set_point: the joint positions that are desired as a numpy array.

        Returns:
            the current error in the joint positions.
        """
        error = current - set_point
        return error

    def clip_joint_velocities(self, velocities):
        """
        Clips joint velocities into a valid range.
        """
        for i in range(len(velocities)):
            if velocities[i] >= 1.0:
                velocities[i] = 1.0
            elif velocities[i] <= -1.0:
                velocities[i] = -1.0
        return velocities
