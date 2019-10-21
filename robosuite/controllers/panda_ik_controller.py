"""
NOTE: requires pybullet module.

Run `pip install pybullet==1.9.5`.
"""

import numpy as np

try:
    import pybullet as p
except ImportError:
    raise Exception(
        "Please make sure pybullet is installed. Run `pip install pybullet==1.9.5`"
    )
from os.path import join as pjoin

import robosuite.utils.transform_utils as T
from robosuite.controllers import Controller


class PandaIKController(Controller):
    """
    Inverse kinematics for the Panda robot, using Pybullet and the urdf description
    files. Loads a panda robot into an internal Pybullet simulation, and uses it to
    do inverse kinematics computations. (##TODO: This is currently identical to the sawyer_ik_controller, check params)
    """

    def __init__(self, bullet_data_path, robot_jpos_getter):
        """
        Args:
            bullet_data_path (str): base path to bullet data.

            robot_jpos_getter (function): function that returns the joint positions of
                the robot to be controlled as a numpy array.
        """

        # path to data folder of bullet repository
        self.bullet_data_path = bullet_data_path

        # returns current robot joint positions
        self.robot_jpos_getter = robot_jpos_getter

        # Do any setup needed for Inverse Kinematics.
        self.setup_inverse_kinematics()

        # Should be in (0, 1], smaller values mean less sensitivity.
        self.user_sensitivity = .3

        self.sync_state()
        self.i = 0

    def get_control(self, dpos, rotation):
        """
        Returns joint velocities to control the robot after the target end effector
        position and orientation are updated from arguments @dpos and @rotation.

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
        self.sync_ik_robot(self.robot_jpos_getter())

        # TODO: EDITED -Josiah
        #rotation = np.array([[1,0,0],[0,1,0],[0,0,1]])
        #dpos = np.array([0,0,0])
        # TODO END


        self.commanded_joint_positions = self.joint_positions_for_eef_command(
            dpos, rotation
        )

        # P controller from joint positions (from IK) to velocities
        velocities = np.zeros(7)
        deltas = self._get_current_error(
            self.robot_jpos_getter(), self.commanded_joint_positions
        )
        print("Deltas: {}".format(deltas))
        for i, delta in enumerate(deltas):
            velocities[i] = -2. * delta  # -2. * delta
        velocities = self.clip_joint_velocities(velocities)
        print("Commanded vels: {}".format(velocities))

        self.commanded_joint_velocities = velocities
        return velocities

    def sync_state(self):
        """
        Syncs the internal Pybullet robot state to the joint positions of the
        robot being controlled.
        """

        # sync IK robot state to the current robot joint positions
        self.sync_ik_robot(self.robot_jpos_getter())

        # make sure target pose is up to date
        self.ik_robot_target_pos, self.ik_robot_target_orn = (
            self.ik_robot_eef_joint_cartesian_pose()
        )

        print("Initial ik target pos: {}\nInitial ik target orn: {}".format(self.ik_robot_target_pos, self.ik_robot_target_orn))

    def setup_inverse_kinematics(self):
        """
        This function is responsible for doing any setup for inverse kinematics.
        Inverse Kinematics maps end effector (EEF) poses to joint angles that
        are necessary to achieve those poses.
        """

        # Set up a connection to the PyBullet simulator.
        p.connect(p.DIRECT)
        p.resetSimulation()
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")

        # get paths to urdfs
        self.robot_urdf = pjoin(
            self.bullet_data_path, "panda_description/urdf/panda_arm.urdf"
        )

        # load the urdfs
        self.ik_robot = p.loadURDF(self.robot_urdf, (0, 0, 0.9), useFixedBase=1)

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
                    i,
                    p.POSITION_CONTROL,
                    targetVelocity=0,
                    targetPosition=joint_positions[i],
                    force=500,
                    positionGain=0.5,
                    velocityGain=1.,
                )
            else:
                p.resetJointState(self.ik_robot, i, joint_positions[i], 0)

    def ik_robot_eef_joint_cartesian_pose(self):
        """
        Returns the current cartesian pose of the last joint of the ik robot with respect to the base frame as
        a (pos, orn) tuple where orn is a x-y-z-w quaternion
        """
        eef_pos_in_world = np.array(p.getLinkState(self.ik_robot, 6)[0])
        eef_orn_in_world = np.array(p.getLinkState(self.ik_robot, 6)[1])
        eef_pose_in_world = T.pose2mat((eef_pos_in_world, eef_orn_in_world))

        base_pos_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[0])
        base_orn_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[1])
        base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        eef_pose_in_base = T.pose_in_A_to_pose_in_B(
            pose_A=eef_pose_in_world, pose_A_in_B=world_pose_in_base
        )


        return T.mat2pose(eef_pose_in_base)

    def inverse_kinematics(self, target_position, target_orientation, rest_poses=None):
        """
        Helper function to do inverse kinematics for a given target position and
        orientation in the PyBullet world frame.

        Args:
            target_position: A tuple, list, or numpy array of size 3 for position.
            target_orientation: A tuple, list, or numpy array of size 4 for
                a orientation quaternion.
            rest_poses: (optional) A list of size @num_joints to favor ik solutions close by.

        Returns:
            A list of size @num_joints corresponding to the joint angle solution.
        """

        # TODO: Trying magic numbers below JOSIAH
        #target_position=[0,0,1]
        #target_orientation=[0,0,0,1]
        lim = np.pi
        # TODO end



        if rest_poses is None:
            ik_solution = list(
                p.calculateInverseKinematics(
                    self.ik_robot,
                    7,
                    target_position,
                    targetOrientation=target_orientation,
                    restPoses=rest_poses,
                    jointDamping=[0.1] * 8,
                )
            )
        else:
            ik_solution = list(
                p.calculateInverseKinematics(
                    self.ik_robot,
                    7,
                    target_position,
                    targetOrientation=target_orientation,
                    lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                    upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                    jointRanges=[5.8, 3.5, 5.8, 3.1, 5.8, 3.8, 5.8],
                    #lowerLimits=[-lim, -lim,-lim,-lim,-lim,-lim,-lim],
                    #upperLimits=[lim,lim,lim,lim,lim,lim,lim],
                    #jointRanges=[2*lim, 2*lim,2*lim,2*lim,2*lim,2*lim,2*lim],
                    restPoses=rest_poses,
                    jointDamping=[0.1] * 8,
                )
            )
        return ik_solution

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
        print("base_pos: {}\nworld_pos: {}".format(T.mat2pose(pose_in_base), base_pos_in_world))
        base_orn_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[1])
        base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))

        pose_in_world = T.pose_in_A_to_pose_in_B(
            pose_A=pose_in_base, pose_A_in_B=base_pose_in_world
        )
        return T.mat2pose(pose_in_world)

    def joint_positions_for_eef_command(self, dpos, rotation):
        """
        This function runs inverse kinematics to back out target joint positions
        from the provided end effector command.

        Same arguments as @get_control.

        Returns:
            A list of size @num_joints corresponding to the target joint angles.
        """

        print("input dpos: {}\ninput rotation: {}".format(dpos, rotation))
        self.ik_robot_target_pos += dpos * self.user_sensitivity

        # this rotation accounts for rotating the end effector by 90 degrees
        # from its rest configuration. The corresponding line in most demo
        # scripts is:
        #   `env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])`
        """
        rotation = rotation.dot(
            T.rotation_matrix(angle=-np.pi / 2, direction=[0., 0., 1.], point=None)[
                :3, :3
            ]
        )
        """
        """
        rotation = rotation.dot(
            T.rotation_matrix(angle=np.pi, direction=[1., 0., 0.], point=None)[
            :3, :3
            ]
        )
        """

        rotation = rotation.dot(
            T.rotation_matrix(angle=-np.pi/4, direction=[0., 0., 1.], point=None)[
            :3, :3
            ]
        )

        """
        rotation = T.rotation_matrix(angle=-np.pi/4, direction=[0., 0., 1.], point=None)[
            :3, :3
            ] @ T.rotation_matrix(angle=np.pi, direction=[1., 0., 0.], point=None)[
            :3, :3
            ] @ rotation
        """



        print("output rotation: {}".format(rotation))

        self.ik_robot_target_orn = T.mat2quat(rotation)

        # convert from target pose in base frame to target pose in bullet world frame
        print("ik_robot_target_pos: {}".format(self.ik_robot_target_pos))
        print("ik_robot_target_orn: {}".format(self.ik_robot_target_orn))
        world_targets = self.bullet_base_pose_to_world_pose(
            (self.ik_robot_target_pos, self.ik_robot_target_orn)
        )

        # TODO - Check

        world_targets = list(world_targets)


        if self.i < 500:
            world_targets[0] -= np.array([0,0,0])
            #world_targets[0] = np.array([0.25, 0.0, 0.75])
            #world_targets[1] = np.array([0, 0.9999997, 0, 0.0007963 ])
            #world_targets[1] = np.array([0.9999997, 0, 0, 0.0007963])
            #world_targets[1] = np.array([0, 0, 0.9999997, 0.0007963])
            #world_targets[1] = np.array([-0.7071068, 0, 0, 0.7071068])
            #world_targets[1] = np.array([0, 1, 0, 0])
            #world_targets[1] = np.array([np.sqrt(0.001 * (self.i)), 0, 0, np.sqrt(0.001 * (-self.i + 500))])
        else:
            world_targets[0] -= np.array([0, 0, 0])
            #world_targets[1] = np.array([0, np.sqrt(0.001*(1500 - self.i)), 0, np.sqrt(0.001*(self.i-500))])
        self.i += 1


        print("World targets: {}".format(world_targets))
        print("ik_robot_pose: {}".format(self.ik_robot_eef_joint_cartesian_pose()))

        # Set default rest pose as a neutral down-position over the center of the table
        rest_poses = [0, np.pi/6, 0.00, -(np.pi - 2*np.pi/6), 0.00, (np.pi - np.pi/6), np.pi/4]
        #rest_poses = [-1.0454259,   1.25584966,  0.92494988, -2.2255072,   1.43708077,  0.87291517, -1.92567813]

        for bullet_i in range(100):
            arm_joint_pos = self.inverse_kinematics(
                world_targets[0], world_targets[1], rest_poses=rest_poses
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
