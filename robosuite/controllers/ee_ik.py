"""
NOTE: requires pybullet module.

Run `pip install pybullet==1.9.5`.
"""
try:
    import pybullet as p
except ImportError:
    raise Exception(
        "Please make sure pybullet is installed. Run `pip install pybullet==1.9.5`"
    )
import os
from os.path import join as pjoin
import robosuite

from robosuite.controllers.joint_vel import JointVelController
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
import numpy as np


class EEIKController(JointVelController):
    """
    Controller for joint velocity
    """

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 robot_name,
                 kv=40.0,
                 policy_freq=20,
                 ik_pos_limit=None,
                 ik_ori_limit=None,
                 interpolator=None,
                 converge_steps=5,
                 **kwargs
                 ):

        # Run sueprclass inits
        super().__init__(
            sim=sim,
            eef_name=eef_name,
            joint_indexes=joint_indexes,
            input_max=50,
            input_min=-50,
            output_max=50,
            output_min=-50,
            kv=kv,
            policy_freq=policy_freq,
            velocity_limits=[-50,50],
            interpolator=interpolator,
            **kwargs
        )

        # Initialize ik-specific attributes
        self.robot_name = robot_name        # Name of robot (e.g.: "panda", "sawyer", etc.)

        # Rotation offsets (for mujoco eef -> pybullet eef) and rest poses
        self.rotation_offset = None
        self.rest_poses = None

        # Set the reference robot target orientation (to prevent drift / weird ik numerical behavior over time)
        self.reference_target_orn = T.mat2quat(self.ee_ori_mat)


        # Values for initializing pybullet env
        self.ik_robot = None
        self.robot_urdf = None
        self.num_bullet_joints = None
        self.bullet_ee_idx = None
        self.bullet_joint_indexes = None   # Useful for splitting right and left hand indexes when controlling bimanual
        self.ik_command_indexes = None     # Relevant indices from ik loop; useful for splitting bimanual into left / right
        self.ik_robot_target_pos_offset = None
        self.converge_steps = converge_steps
        self.ik_pos_limit = ik_pos_limit
        self.ik_ori_limit = ik_ori_limit

        # Target pos and ori
        self.ik_robot_target_pos = None
        self.ik_robot_target_orn = None

        # Commanded pos and resulting commanded vel
        self.commanded_joint_positions = None
        self.commanded_joint_velocities = None

        # Should be in (0, 1], smaller values mean less sensitivity.
        self.user_sensitivity = .3

        # Setup inverse kinematics
        self.setup_inverse_kinematics()

        # Lastly, sync pybullet state to mujoco state
        self.sync_state()

    def setup_inverse_kinematics(self):
        """
        This function is responsible for doing any setup for inverse kinematics.
        Inverse Kinematics maps end effector (EEF) poses to joint angles that
        are necessary to achieve those poses.
        """

        # Set up a connection to the PyBullet simulator.
        p.connect(p.DIRECT)
        p.resetSimulation()

        # get paths to urdfs
        self.robot_urdf = pjoin(
            os.path.join(robosuite.models.assets_root, "bullet_data"),
            "{}_description/urdf/{}_arm.urdf".format(self.robot_name, self.robot_name)
        )

        # load the urdfs
        if self.robot_name == "baxter":
            self.ik_robot = p.loadURDF(self.robot_urdf, (0, 0, 0.0), useFixedBase=1)
        else:
            self.ik_robot = p.loadURDF(self.robot_urdf, (0, 0, 0.9), useFixedBase=1)

        # load the number of joints from the bullet data
        self.num_bullet_joints = p.getNumJoints(self.ik_robot)

        # For now, hard code baxter bullet eef idx
        if self.robot_name == "baxter":
            self.ik_robot_target_pos_offset = np.array([0, 0, 0.913])
            if self.eef_name == "right_hand":
                self.bullet_ee_idx = 27
                self.bullet_joint_indexes = [13, 14, 15, 16, 17, 19, 20]
                self.ik_command_indexes = np.arange(1, self.joint_dim + 1)
            elif self.eef_name == "left_hand":
                self.bullet_ee_idx = 45
                self.bullet_joint_indexes = [31, 32, 33, 34, 35, 37, 38]
                self.ik_command_indexes = np.arange(self.joint_dim + 1, self.joint_dim * 2 + 1)
        else:
            # Default assumes pybullet has same number of joints compared to mujoco sim
            self.bullet_ee_idx = self.num_bullet_joints - 1
            self.bullet_joint_indexes = np.arange(self.joint_dim)
            self.ik_command_indexes = np.arange(self.joint_dim)
            self.ik_robot_target_pos_offset = np.zeros(3)

        # Set rotation offsets (for mujoco eef -> pybullet eef) and rest poses
        if self.robot_name == "sawyer":
            self.rotation_offset = T.rotation_matrix(angle=-np.pi / 2, direction=[0., 0., 1.], point=None)
            self.rest_poses = [0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161]
        elif self.robot_name == "panda":
            self.rotation_offset = T.rotation_matrix(angle=np.pi/4, direction=[0., 0., 1.], point=None)
            self.rest_poses = [0, np.pi / 6, 0.00, -(np.pi - 2 * np.pi / 6), 0.00, (np.pi - np.pi / 6), np.pi / 4]
        elif self.robot_name == "baxter":
            self.rotation_offset = T.rotation_matrix(angle=0, direction=[0., 0., 1.], point=None)
            if self.eef_name == "right_hand":
                self.rest_poses = [0.535, -0.093, 0.038, 0.166, 0.643, 1.960, -1.297]
            elif self.eef_name == "left_hand":
                self.rest_poses = [-0.518, -0.026, -0.076, 0.175, -0.748, 1.641, -0.158]
            else:
                # Error with inputted id
                print("Error loading ik controller for Baxter -- arm id's must be either 'right_hand' or 'left_hand'!")
        else:
            # No other robots supported, print out to user
            print("ERROR: Unsupported robot requested for ik controller. Only sawyer, panda, and baxter "
                  "currently supported.")

        # Simulation will update as fast as it can in real time, instead of waiting for
        # step commands like in the non-realtime case.
        p.setRealTimeSimulation(1)

    def sync_state(self):
        """
        Syncs the internal Pybullet robot state to the joint positions of the
        robot being controlled.
        """

        # sync IK robot state to the current robot joint positions
        self.sync_ik_robot()

        # make sure target pose is up to date
        self.ik_robot_target_pos, self.ik_robot_target_orn = (
            self.ik_robot_eef_joint_cartesian_pose()
        )

    def sync_ik_robot(self, joint_positions=None, simulate=False, sync_last=True):
        """
        Force the internal robot model to match the provided joint angles.

        Args:
            joint_positions (list): a list or flat numpy array of joint positions. Default automatically updates to
                current mujoco joint pos state
            simulate (bool): If True, actually use physics simulation, else
                write to physics state directly.
            sync_last (bool): If False, don't sync the last joint angle. This
                is useful for directly controlling the roll at the end effector.
        """
        self.update()
        if not joint_positions:
            joint_positions = self.joint_pos
        num_joints = self.joint_dim
        if not sync_last and self.robot_name != "baxter":
            num_joints -= 1
        for i in range(num_joints):
            if simulate:
                p.setJointMotorControl2(
                    self.ik_robot,
                    self.bullet_joint_indexes[i],
                    p.POSITION_CONTROL,
                    targetVelocity=0,
                    targetPosition=joint_positions[i],
                    force=500,
                    positionGain=0.5,
                    velocityGain=1.,
                )
            else:
                p.resetJointState(self.ik_robot, self.bullet_joint_indexes[i], joint_positions[i], 0)

    def ik_robot_eef_joint_cartesian_pose(self):
        """
        Returns the current cartesian pose of the last joint of the ik robot with respect to the base frame as
        a (pos, orn) tuple where orn is a x-y-z-w quaternion
        """
        eef_pos_in_world = np.array(p.getLinkState(self.ik_robot, self.bullet_ee_idx)[0])
        eef_orn_in_world = np.array(p.getLinkState(self.ik_robot, self.bullet_ee_idx)[1])
        eef_pose_in_world = T.pose2mat((eef_pos_in_world, eef_orn_in_world))

        base_pos_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[0])
        base_orn_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[1])
        base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        eef_pose_in_base = T.pose_in_A_to_pose_in_B(
            pose_A=eef_pose_in_world, pose_A_in_B=world_pose_in_base
        )

        return T.mat2pose(eef_pose_in_base)

    def get_control(self, dpos=None, rotation=None):
        """
        Returns joint velocities to control the robot after the target end effector
        position and orientation are updated from arguments @dpos and @rotation.
        If no arguments are provided, joint velocities will be computed based
        on the previously recorded target.

        Args:
            dpos (numpy array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (numpy array): a rotation matrix of shape (3, 3) corresponding
                to the desired orientation of the end effector.

        Returns:
            velocities (numpy array): a flat array of joint velocity commands to apply
                to try and achieve the desired input control.
        """
        # Sync joint positions for IK.
        self.sync_ik_robot()

        # Compute new target joint positions if arguments are provided
        if (dpos is not None) and (rotation is not None):
            self.commanded_joint_positions = np.array(self.joint_positions_for_eef_command(
                dpos, rotation
            ))

        # P controller from joint positions (from IK) to velocities
        velocities = np.zeros(self.joint_dim)
        deltas = self._get_current_error(
            self.joint_pos, self.commanded_joint_positions
        )
        for i, delta in enumerate(deltas):
            velocities[i] = -10. * delta

        self.commanded_joint_velocities = velocities
        return velocities

    def inverse_kinematics(self, target_position, target_orientation):
        """
        Helper function to do inverse kinematics for a given target position and
        orientation in the PyBullet world frame.

        Args:
            target_position: A tuple, list, or numpy array of size 3 for position.
            target_orientation: A tuple, list, or numpy array of size 4 for
                a orientation quaternion

        Returns:
            A list of size @num_joints corresponding to the joint angle solution.
        """
        ik_solution = list(
            p.calculateInverseKinematics(
                self.ik_robot,
                self.bullet_ee_idx,
                target_position,
                targetOrientation=target_orientation,
                lowerLimits=list(self.sim.model.jnt_range[self.joint_index, 0]),
                upperLimits=list(self.sim.model.jnt_range[self.joint_index, 1]),
                jointRanges=list(self.sim.model.jnt_range[self.joint_index, 1] -
                                 self.sim.model.jnt_range[self.joint_index, 0]),
                restPoses=self.rest_poses,
                jointDamping=[0.1] * self.num_bullet_joints,
            )
        )
        return list(np.array(ik_solution)[self.ik_command_indexes])

    def joint_positions_for_eef_command(self, dpos, rotation):
        """
        This function runs inverse kinematics to back out target joint positions
        from the provided end effector command.

        Same arguments as @get_control.

        Returns:
            A list of size @num_joints corresponding to the target joint angles.
        """
        # Scale and increment target position
        self.ik_robot_target_pos += dpos * self.user_sensitivity

        # this rotation accounts for rotating the end effector (deviation between mujoco eef and pybullet eef)
        rotation = rotation.dot(self.rotation_offset[:3, :3])

        # Convert the desired rotation into the target orientation quaternion
        self.ik_robot_target_orn = T.mat2quat(rotation)

        # convert from target pose in base frame to target pose in bullet world frame
        world_targets = self.bullet_base_pose_to_world_pose(
            (self.ik_robot_target_pos, self.ik_robot_target_orn)
        )

        arm_joint_pos = None
        for bullet_i in range(self.converge_steps):
            arm_joint_pos = self.inverse_kinematics(world_targets[0], world_targets[1])
            self.sync_ik_robot(arm_joint_pos, sync_last=True)

        return arm_joint_pos

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

    def set_goal(self, delta, set_ik=None):
        # Run ik prepropressing to convert pos, quat ori to desired velocities
        requested_control = self._make_input(delta, self.reference_target_orn)

        # Compute desired velocities to achieve eef pos / ori
        velocities = self.get_control(**requested_control)

        super().set_goal(velocities)

    def run_controller(self, action=None):
        # First, update goal if action is not set to none
        # Action will be interpreted as delta value from current

        return super().run_controller(action)

    def _pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def _clip_ik_input(self, dpos, rotation):
        """
        Helper function that clips desired ik input deltas into a valid range.

        Args:
            dpos (numpy array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (numpy array): relative rotation quaternion (x, y, z, w) corresponding
                to the (relative) desired orientation of the end effector.

        Returns:
            clipped dpos, rotation (of same type)
        """
        # scale input range to desired magnitude
        if dpos.any():
            dpos = T.clip_translation(dpos, self.ik_pos_limit)

        # Clip orientation to desired magnitude
        rotation = T.clip_rotation(rotation, self.ik_ori_limit)

        return dpos, rotation

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat.

        Additionally clips actions as well
        """
        # Clip action appropriately
        dpos, rotation = self._clip_ik_input(action[:3], action[3:7])

        self.reference_target_orn = T.quat_multiply(old_quat, rotation)

        return {
            "dpos": dpos,
            # IK controller takes an absolute orientation in robot base frame
            "rotation": T.quat2mat(self.reference_target_orn)
        }

    @staticmethod
    def _get_current_error(current, set_point):
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

    @property
    def name(self):
        return 'ee_ik'

