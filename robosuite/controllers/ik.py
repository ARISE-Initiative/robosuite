"""
***********************************************************************************

NOTE: requires pybullet module.

Run `pip install "pybullet-svl>=3.1.6.4"`.


NOTE: IK is only supported for the following robots:

:Baxter:
:Sawyer:
:Panda:

Attempting to run IK with any other robot will raise an error!

***********************************************************************************
"""
try:
    import pybullet as p
except ImportError:
    raise Exception("""Please make sure pybullet is installed. Run `pip install "pybullet-svl>=3.1.6.4"`""")
import os
from os.path import join as pjoin

import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers.joint_vel import JointVelocityController
from robosuite.utils.control_utils import *

# Dict of supported ik robots
SUPPORTED_IK_ROBOTS = {"Baxter", "Sawyer", "Panda"}


class PyBulletServer(object):
    """
    Helper class to encapsulate an alias for a single pybullet server
    """

    def __init__(self):
        # Attributes
        self.server_id = None
        self.is_active = False

        # Bodies: Dict of <bullet_robot_id : robot_name> active in pybullet simulation
        self.bodies = {}

        # Automatically setup this pybullet server
        self.connect()

    def connect(self):
        """
        Global function to (re-)connect to pybullet server instance if it's not currently active
        """
        if not self.is_active:
            self.server_id = p.connect(p.DIRECT)

            # Reset simulation (Assumes pre-existing connection to the PyBullet simulator)
            p.resetSimulation(physicsClientId=self.server_id)
            self.is_active = True

    def disconnect(self):
        """
        Function to disconnect and shut down this pybullet server instance.

        Should be called externally before resetting / instantiating a new controller
        """
        if self.is_active:
            p.disconnect(physicsClientId=self.server_id)
            self.bodies = {}
            self.is_active = False


class InverseKinematicsController(JointVelocityController):
    """
    Controller for controlling robot arm via inverse kinematics. Allows position and orientation control of the
    robot's end effector.

    Inverse kinematics solving is handled by pybullet.

    NOTE: Control input actions are assumed to be relative to the current position / orientation of the end effector
    and are taken as the array (x_dpos, y_dpos, z_dpos, x_rot, y_rot, z_rot).

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        robot_name (str): Name of robot being controlled. Can be {"Sawyer", "Panda", or "Baxter"}

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        eef_rot_offset (4-array): Quaternion (x,y,z,w) representing rotational offset between the final
            robot arm link coordinate system and the end effector coordinate system (i.e: the gripper)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        ik_pos_limit (float): Limit (meters) above which the magnitude of a given action's
            positional inputs will be clipped

        ik_ori_limit (float): Limit (radians) above which the magnitude of a given action's
            orientation inputs will be clipped

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current state to
            the goal state during each timestep between inputted actions

        converge_steps (int): How many iterations to run the pybullet inverse kinematics solver to converge to a
            solution

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Unsupported robot]
    """

    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        robot_name,
        actuator_range,
        eef_rot_offset,
        bullet_server_id=0,
        policy_freq=20,
        load_urdf=True,
        ik_pos_limit=None,
        ik_ori_limit=None,
        interpolator_pos=None,
        interpolator_ori=None,
        converge_steps=5,
        **kwargs,
    ):

        # Run sueprclass inits
        super().__init__(
            sim=sim,
            eef_name=eef_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            input_max=1,
            input_min=-1,
            output_max=1,
            output_min=-1,
            kv=0.25,
            policy_freq=policy_freq,
            velocity_limits=[-1, 1],
            **kwargs,
        )

        # Verify robot is supported by IK
        assert robot_name in SUPPORTED_IK_ROBOTS, (
            "Error: Tried to instantiate IK controller for unsupported robot! "
            "Inputted robot: {}, Supported robots: {}".format(robot_name, SUPPORTED_IK_ROBOTS)
        )

        # Initialize ik-specific attributes
        self.robot_name = robot_name  # Name of robot (e.g.: "Panda", "Sawyer", etc.)

        # Override underlying control dim
        self.control_dim = 6

        # Rotation offsets (for mujoco eef -> pybullet eef) and rest poses
        self.eef_rot_offset = eef_rot_offset
        self.rotation_offset = None
        self.rest_poses = None

        # Set the reference robot target pos / orientation (to prevent drift / weird ik numerical behavior over time)
        self.reference_target_pos = self.ee_pos
        self.reference_target_orn = T.mat2quat(self.ee_ori_mat)

        # Bullet server id
        self.bullet_server_id = bullet_server_id

        # Interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # Interpolator-related attributes
        self.ori_ref = None
        self.relative_ori = None

        # Values for initializing pybullet env
        self.ik_robot = None
        self.robot_urdf = None
        self.num_bullet_joints = None
        self.bullet_ee_idx = None
        self.bullet_joint_indexes = None  # Useful for splitting right and left hand indexes when controlling bimanual
        self.ik_command_indexes = None  # Relevant indices from ik loop; useful for splitting bimanual left / right
        self.ik_robot_target_pos_offset = None
        self.base_orn_offset_inv = None  # inverse orientation offset from pybullet base to world
        self.converge_steps = converge_steps

        # Set ik limits and override internal min / max
        self.ik_pos_limit = ik_pos_limit
        self.ik_ori_limit = ik_ori_limit

        # Target pos and ori
        self.ik_robot_target_pos = None
        self.ik_robot_target_orn = None  # note: this currently isn't being used at all

        # Commanded pos and resulting commanded vel
        self.commanded_joint_positions = None
        self.commanded_joint_velocities = None

        # Should be in (0, 1], smaller values mean less sensitivity.
        self.user_sensitivity = 0.3

        # Setup inverse kinematics
        self.setup_inverse_kinematics(load_urdf)

        # Lastly, sync pybullet state to mujoco state
        self.sync_state()

    def setup_inverse_kinematics(self, load_urdf=True):
        """
        This function is responsible for doing any setup for inverse kinematics.

        Inverse Kinematics maps end effector (EEF) poses to joint angles that are necessary to achieve those poses.

        Args:
            load_urdf (bool): specifies whether the robot urdf should be loaded into the sim. Useful flag that
                should be cleared in the case of multi-armed robots which might have multiple IK controller instances
                but should all reference the same (single) robot urdf within the bullet sim

        Raises:
            ValueError: [Invalid eef id]
        """

        # get paths to urdfs
        self.robot_urdf = pjoin(
            os.path.join(robosuite.models.assets_root, "bullet_data"),
            "{}_description/urdf/{}_arm.urdf".format(self.robot_name.lower(), self.robot_name.lower()),
        )

        # import reference to the global pybullet server and load the urdfs
        from robosuite.controllers import get_pybullet_server

        if load_urdf:
            self.ik_robot = p.loadURDF(fileName=self.robot_urdf, useFixedBase=1, physicsClientId=self.bullet_server_id)
            # Add this to the pybullet server
            get_pybullet_server().bodies[self.ik_robot] = self.robot_name
        else:
            # We'll simply assume the most recent robot (robot with highest pybullet id) is the relevant robot and
            # mark this controller as belonging to that robot body
            self.ik_robot = max(get_pybullet_server().bodies)

        # load the number of joints from the bullet data
        self.num_bullet_joints = p.getNumJoints(self.ik_robot, physicsClientId=self.bullet_server_id)

        # Disable collisions between all the joints
        for joint in range(self.num_bullet_joints):
            p.setCollisionFilterGroupMask(
                bodyUniqueId=self.ik_robot,
                linkIndexA=joint,
                collisionFilterGroup=0,
                collisionFilterMask=0,
                physicsClientId=self.bullet_server_id,
            )

        # TODO: Very ugly initialization - any way to automate this? Maybe move the hardcoded magic numbers to the robot model files?
        # TODO: Rotations for non-default grippers are not all supported -- e.g.: Robotiq140 Gripper whose coordinate frame
        #   is fully flipped about its x axis -- resulting in mirrored rotational behavior when trying to execute IK control

        # For now, hard code baxter bullet eef idx
        if self.robot_name == "Baxter":
            if "right" in self.eef_name:
                self.bullet_ee_idx = 27
                self.bullet_joint_indexes = [13, 14, 15, 16, 17, 19, 20]
                self.ik_command_indexes = np.arange(1, self.joint_dim + 1)
            elif "left" in self.eef_name:
                self.bullet_ee_idx = 45
                self.bullet_joint_indexes = [31, 32, 33, 34, 35, 37, 38]
                self.ik_command_indexes = np.arange(self.joint_dim + 1, self.joint_dim * 2 + 1)
            else:
                # Error with inputted id
                raise ValueError("Error loading ik controller for Baxter -- arm id's must contain 'right' or 'left'!")
        else:
            # Default assumes pybullet has same number of joints compared to mujoco sim
            self.bullet_ee_idx = self.num_bullet_joints - 1
            self.bullet_joint_indexes = np.arange(self.joint_dim)
            self.ik_command_indexes = np.arange(self.joint_dim)

        # Set rotation offsets (for mujoco eef -> pybullet eef) and rest poses
        self.rest_poses = list(self.initial_joint)
        eef_offset = np.eye(4)
        eef_offset[:3, :3] = T.quat2mat(T.quat_inverse(self.eef_rot_offset))

        self.rotation_offset = eef_offset

        # Simulation will update as fast as it can in real time, instead of waiting for
        # step commands like in the non-realtime case.
        p.setRealTimeSimulation(1, physicsClientId=self.bullet_server_id)

    def sync_state(self):
        """
        Syncs the internal Pybullet robot state to the joint positions of the
        robot being controlled.
        """

        # update model (force update)
        self.update(force=True)

        # sync IK robot state to the current robot joint positions
        self.sync_ik_robot()

        # make sure target pose is up to date
        self.ik_robot_target_pos, self.ik_robot_target_orn = self.ik_robot_eef_joint_cartesian_pose()

        # Store initial offset for mapping pose between mujoco and pybullet (pose_pybullet = offset + pose_mujoco)
        self.ik_robot_target_pos_offset = self.ik_robot_target_pos - self.ee_pos

    def sync_ik_robot(self, joint_positions=None, simulate=False, sync_last=True):
        """
        Force the internal robot model to match the provided joint angles.

        Args:
            joint_positions (Iterable): Array of joint positions. Default automatically updates to
                current mujoco joint pos state
            simulate (bool): If True, actually use physics simulation, else
                write to physics state directly.
            sync_last (bool): If False, don't sync the last joint angle. This
                is useful for directly controlling the roll at the end effector.
        """
        if not joint_positions:
            joint_positions = self.joint_pos
        num_joints = self.joint_dim
        if not sync_last and self.robot_name != "Baxter":
            num_joints -= 1
        for i in range(num_joints):
            if simulate:
                p.setJointMotorControl2(
                    bodyUniqueId=self.ik_robot,
                    jointIndex=self.bullet_joint_indexes[i],
                    controlMode=p.POSITION_CONTROL,
                    targetVelocity=0,
                    targetPosition=joint_positions[i],
                    force=500,
                    positionGain=0.5,
                    velocityGain=1.0,
                    physicsClientId=self.bullet_server_id,
                )
            else:
                p.resetJointState(
                    bodyUniqueId=self.ik_robot,
                    jointIndex=self.bullet_joint_indexes[i],
                    targetValue=joint_positions[i],
                    targetVelocity=0,
                    physicsClientId=self.bullet_server_id,
                )

    def ik_robot_eef_joint_cartesian_pose(self):
        """
        Calculates the current cartesian pose of the last joint of the ik robot with respect to the base frame as
        a (pos, orn) tuple where orn is a x-y-z-w quaternion

        Returns:
            2-tuple:

                - (np.array) position
                - (np.array) orientation
        """
        eef_pos_in_world = np.array(
            p.getLinkState(self.ik_robot, self.bullet_ee_idx, physicsClientId=self.bullet_server_id)[0]
        )
        eef_orn_in_world = np.array(
            p.getLinkState(self.ik_robot, self.bullet_ee_idx, physicsClientId=self.bullet_server_id)[1]
        )
        eef_pose_in_world = T.pose2mat((eef_pos_in_world, eef_orn_in_world))

        base_pos_in_world = np.array(
            p.getBasePositionAndOrientation(self.ik_robot, physicsClientId=self.bullet_server_id)[0]
        )
        base_orn_in_world = np.array(
            p.getBasePositionAndOrientation(self.ik_robot, physicsClientId=self.bullet_server_id)[1]
        )
        base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        # Update reference to inverse orientation offset from pybullet base frame to world frame
        self.base_orn_offset_inv = T.quat2mat(T.quat_inverse(base_orn_in_world))

        # Update reference target orientation
        self.reference_target_orn = T.quat_multiply(self.reference_target_orn, base_orn_in_world)

        eef_pose_in_base = T.pose_in_A_to_pose_in_B(pose_A=eef_pose_in_world, pose_A_in_B=world_pose_in_base)

        return T.mat2pose(eef_pose_in_base)

    def get_control(self, dpos=None, rotation=None, update_targets=False):
        """
        Returns joint velocities to control the robot after the target end effector
        position and orientation are updated from arguments @dpos and @rotation.
        If no arguments are provided, joint velocities will be computed based
        on the previously recorded target.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): a rotation matrix of shape (3, 3) corresponding
                to the desired rotation from the current orientation of the end effector.
            update_targets (bool): whether to update ik target pos / ori attributes or not

        Returns:
            np.array: a flat array of joint velocity commands to apply to try and achieve the desired input control.
        """
        # Sync joint positions for IK.
        self.sync_ik_robot()

        # Compute new target joint positions if arguments are provided
        if (dpos is not None) and (rotation is not None):
            self.commanded_joint_positions = np.array(
                self.joint_positions_for_eef_command(dpos, rotation, update_targets)
            )

        # P controller from joint positions (from IK) to velocities
        velocities = np.zeros(self.joint_dim)
        deltas = self._get_current_error(self.joint_pos, self.commanded_joint_positions)
        for i, delta in enumerate(deltas):
            velocities[i] = -10.0 * delta

        self.commanded_joint_velocities = velocities
        return velocities

    def inverse_kinematics(self, target_position, target_orientation):
        """
        Helper function to do inverse kinematics for a given target position and
        orientation in the PyBullet world frame.

        Args:
            target_position (3-tuple): desired position
            target_orientation (4-tuple): desired orientation quaternion

        Returns:
            list: list of size @num_joints corresponding to the joint angle solution.
        """
        ik_solution = list(
            p.calculateInverseKinematics(
                bodyUniqueId=self.ik_robot,
                endEffectorLinkIndex=self.bullet_ee_idx,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                lowerLimits=list(self.sim.model.jnt_range[self.joint_index, 0]),
                upperLimits=list(self.sim.model.jnt_range[self.joint_index, 1]),
                jointRanges=list(
                    self.sim.model.jnt_range[self.joint_index, 1] - self.sim.model.jnt_range[self.joint_index, 0]
                ),
                restPoses=self.rest_poses,
                jointDamping=[0.1] * self.num_bullet_joints,
                physicsClientId=self.bullet_server_id,
            )
        )
        return list(np.array(ik_solution)[self.ik_command_indexes])

    def joint_positions_for_eef_command(self, dpos, rotation, update_targets=False):
        """
        This function runs inverse kinematics to back out target joint positions
        from the provided end effector command.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): a rotation matrix of shape (3, 3) corresponding
                to the desired rotation from the current orientation of the end effector.
            update_targets (bool): whether to update ik target pos / ori attributes or not

        Returns:
            list: A list of size @num_joints corresponding to the target joint angles.
        """

        # Calculate the rotation
        # This equals: inv base offset * eef * offset accounting for deviation between mujoco eef and pybullet eef
        rotation = self.base_orn_offset_inv @ self.ee_ori_mat @ rotation @ self.rotation_offset[:3, :3]

        # Determine targets based on whether we're using interpolator(s) or not
        if self.interpolator_pos or self.interpolator_ori:
            targets = (self.ee_pos + dpos + self.ik_robot_target_pos_offset, T.mat2quat(rotation))
        else:
            targets = (self.ik_robot_target_pos + dpos, T.mat2quat(rotation))

        # convert from target pose in base frame to target pose in bullet world frame
        world_targets = self.bullet_base_pose_to_world_pose(targets)

        # Update targets if required
        if update_targets:
            # Scale and increment target position
            self.ik_robot_target_pos += dpos

            # Convert the desired rotation into the target orientation quaternion
            self.ik_robot_target_orn = T.mat2quat(rotation)

        # Converge to IK solution
        arm_joint_pos = None
        for bullet_i in range(self.converge_steps):
            arm_joint_pos = self.inverse_kinematics(world_targets[0], world_targets[1])
            self.sync_ik_robot(arm_joint_pos, sync_last=True)

        return arm_joint_pos

    def bullet_base_pose_to_world_pose(self, pose_in_base):
        """
        Convert a pose in the base frame to a pose in the world frame.

        Args:
            pose_in_base (2-tuple): a (pos, orn) tuple.

        Returns:
            2-tuple: a (pos, orn) tuple reflecting robot pose in world coordinates
        """
        pose_in_base = T.pose2mat(pose_in_base)

        base_pos_in_world, base_orn_in_world = p.getBasePositionAndOrientation(
            self.ik_robot, physicsClientId=self.bullet_server_id
        )
        base_pos_in_world, base_orn_in_world = np.array(base_pos_in_world), np.array(base_orn_in_world)

        base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))

        pose_in_world = T.pose_in_A_to_pose_in_B(pose_A=pose_in_base, pose_A_in_B=base_pose_in_world)
        return T.mat2pose(pose_in_world)

    def set_goal(self, delta, set_ik=None):
        """
        Sets the internal goal state of this controller based on @delta

        Note that this controller wraps a VelocityController, and so determines the desired velocities
        to achieve the inputted pose, and sets its internal setpoint in terms of joint velocities

        TODO: Add feature so that using @set_ik automatically sets the target values to these absolute values

        Args:
            delta (Iterable): Desired relative position / orientation goal state
            set_ik (Iterable): If set, overrides @delta and sets the desired global position / orientation goal state
        """
        # Update state
        self.update()

        # Get requested delta inputs if we're using interpolators
        (dpos, dquat) = self._clip_ik_input(delta[:3], delta[3:7])

        # Set interpolated goals if necessary
        if self.interpolator_pos is not None:
            # Absolute position goal
            self.interpolator_pos.set_goal(dpos * self.user_sensitivity + self.reference_target_pos)

        if self.interpolator_ori is not None:
            # Relative orientation goal
            self.interpolator_ori.set_goal(dquat)  # goal is the relative change in orientation
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

        # Run ik prepropressing to convert pos, quat ori to desired velocities
        requested_control = self._make_input(delta, self.reference_target_orn)

        # Compute desired velocities to achieve eef pos / ori
        velocities = self.get_control(**requested_control, update_targets=True)

        # Set the goal velocities for the underlying velocity controller
        super().set_goal(velocities)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        # Update interpolated action if necessary
        desired_pos = None
        rotation = None
        update_velocity_goal = False

        # Update interpolated goals if active
        if self.interpolator_pos is not None:
            # Linear case
            if self.interpolator_pos.order == 1:
                desired_pos = self.interpolator_pos.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
            update_velocity_goal = True
        else:
            desired_pos = self.reference_target_pos

        if self.interpolator_ori is not None:
            # Linear case
            if self.interpolator_ori.order == 1:
                # relative orientation based on difference between current ori and ref
                self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)
                ori_error = self.interpolator_ori.get_interpolated_goal()
                rotation = T.quat2mat(ori_error)
            else:
                # Nonlinear case not currently supported
                pass
            update_velocity_goal = True
        else:
            rotation = T.quat2mat(self.reference_target_orn)

        # Only update the velocity goals if we're interpolating
        if update_velocity_goal:
            velocities = self.get_control(dpos=(desired_pos - self.ee_pos), rotation=rotation)
            super().set_goal(velocities)

        # Run controller with given action
        return super().run_controller()

    def update_base_pose(self, base_pos, base_ori):
        # Update pybullet robot base and orientation according to values
        base_ori_quat = T.mat2quat(base_ori)
        p.resetBasePositionAndOrientation(
            bodyUniqueId=self.ik_robot, posObj=base_pos, ornObj=base_ori_quat, physicsClientId=self.bullet_server_id
        )

        # Re-sync pybullet state
        self.sync_state()

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # Then, update the rest pose from the initial joints
        self.rest_poses = list(self.initial_joint)

    def reset_goal(self):
        """
        Resets the goal to the current pose of the robot
        """
        self.reference_target_pos = self.ee_pos
        self.reference_target_orn = T.mat2quat(self.ee_ori_mat)

        # Sync pybullet state as well
        self.sync_state()

    def _clip_ik_input(self, dpos, rotation):
        """
        Helper function that clips desired ik input deltas into a valid range.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): relative rotation in scaled axis angle form (ax, ay, az)
                corresponding to the (relative) desired orientation of the end effector.

        Returns:
            2-tuple:

                - (np.array) clipped dpos
                - (np.array) clipped rotation
        """
        # scale input range to desired magnitude
        if dpos.any():
            dpos, _ = T.clip_translation(dpos, self.ik_pos_limit)

        # Map input to quaternion
        rotation = T.axisangle2quat(rotation)

        # Clip orientation to desired magnitude
        rotation, _ = T.clip_rotation(rotation, self.ik_ori_limit)

        return dpos, rotation

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat. Additionally clips @action as well

        Args:
            action (np.array) should have form: [dx, dy, dz, ax, ay, az] (orientation in
                scaled axis-angle form)
            old_quat (np.array) the old target quaternion that will be updated with the relative change in @action
        """
        # Clip action appropriately
        dpos, rotation = self._clip_ik_input(action[:3], action[3:])

        # Update reference targets
        self.reference_target_pos += dpos * self.user_sensitivity
        self.reference_target_orn = T.quat_multiply(old_quat, rotation)

        return {"dpos": dpos * self.user_sensitivity, "rotation": T.quat2mat(rotation)}

    @staticmethod
    def _get_current_error(current, set_point):
        """
        Returns an array of differences between the desired joint positions and current
        joint positions. Useful for PID control.

        Args:
            current (np.array): the current joint positions
            set_point (np.array): the joint positions that are desired as a numpy array

        Returns:
            np.array: the current error in the joint positions
        """
        error = current - set_point
        return error

    @property
    def control_limits(self):
        """
        The limits over this controller's action space, as specified by self.ik_pos_limit and self.ik_ori_limit
        and overriding the superclass method

        Returns:
            2-tuple:

                - (np.array) minimum control values
                - (np.array) maximum control values
        """
        max_limit = np.concatenate([self.ik_pos_limit * np.ones(3), self.ik_ori_limit * np.ones(3)])
        return -max_limit, max_limit

    @property
    def name(self):
        return "IK_POSE"
