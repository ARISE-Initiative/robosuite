from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T

from mujoco_py import MjSim

from robosuite.models.robots import create_robot


class Robot(object):
    """Initializes a robot, as defined by a single corresponding XML"""

    def __init__(
        self,
        robot_type: str,
        idn=0,
        initial_qpos=None,
        initialization_noise=None,
    ):
        """
        Args:
            robot_type (str): Specification for specific robot arm to be instantiated within this env (e.g: "Panda")

            idn (int or str): Unique ID of this robot. Should be different from others

            initial_qpos (sequence of float): If set, determines the initial joint positions of the robot to be
                instantiated for the task

            initialization_noise (dict): Dict containing the initialization noise parameters. The expected keys and
                corresponding value types are specified below:
                "magnitude": The scale factor of uni-variate random noise applied to each of a robot's given initial
                    joint positions. Setting this value to "None" or 0.0 results in no noise being applied.
                    If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                    If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
                "type": Type of noise to apply. Can either specify "gaussian" or "uniform"
                Note: Specifying None will automatically create the required dict with "magnitude" set to 0.0

        """

        self.sim = None                                     # MjSim this robot is tied to
        self.name = robot_type                              # Specific robot to instantiate
        self.idn = idn                                      # Unique ID of this robot
        self.robot_model = None                             # object holding robot model-specific info

        # Scaling of Gaussian initial noise applied to robot joints
        self.initialization_noise = initialization_noise
        if self.initialization_noise is None:
            self.initialization_noise = {"magnitude": 0.0, "type": "gaussian"}  # no noise conditions
        elif self.initialization_noise == "default":
            self.initialization_noise = {"magnitude": 0.02, "type": "gaussian"}
        self.initialization_noise["magnitude"] = \
            self.initialization_noise["magnitude"] if self.initialization_noise["magnitude"] else 0.0

        self.init_qpos = initial_qpos  # n-dim list / array of robot joints

        self.robot_joints = None                            # xml joint names for robot
        self.base_pos = None                                # Base position in world coordinates (x,y,z)
        self.base_ori = None                                # Base rotation in world coordinates (x,y,z,w quat)
        self._ref_joint_indexes = None                      # xml joint indexes for robot in mjsim
        self._ref_joint_pos_indexes = None                  # xml joint position indexes in mjsim
        self._ref_joint_vel_indexes = None                  # xml joint velocity indexes in mjsim
        self._ref_joint_pos_actuator_indexes = None         # xml joint pos actuator indexes for robot in mjsim
        self._ref_joint_vel_actuator_indexes = None         # xml joint vel actuator indexes for robot in mjsim
        self._ref_joint_torq_actuator_indexes = None        # xml joint torq actuator indexes for robot in mjsim

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        raise NotImplementedError

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        self.robot_model = create_robot(self.name, idn=self.idn)

        # Use default from robot model for initial joint positions if not specified
        if self.init_qpos is None:
            self.init_qpos = self.robot_model.init_qpos

    def reset_sim(self, sim: MjSim):
        """
        Replaces current sim with a new sim

        sim (MjSim): New simulation being instantiated to replace the old one
        """
        self.sim = sim

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides robot joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        """
        init_qpos = self.init_qpos
        if not deterministic:
            # Determine noise
            if self.initialization_noise["type"] == "gaussian":
                noise = np.random.randn(len(self.init_qpos)) * self.initialization_noise["magnitude"]
            elif self.initialization_noise["type"] == "uniform":
                noise = np.random.uniform(-1.0, 1.0, len(self.init_qpos)) * self.initialization_noise["magnitude"]
            else:
                raise ValueError("Error: Invalid noise type specified. Options are 'gaussian' or 'uniform'.")
            init_qpos += noise

        # Set initial position in sim
        self.sim.data.qpos[self._ref_joint_pos_indexes] = init_qpos

        # Load controllers
        self._load_controller()

        # Update base pos / ori references
        self.base_pos = self.sim.data.get_body_xpos(self.robot_model.robot_base)
        self.base_ori = T.mat2quat(self.sim.data.get_body_xmat(self.robot_model.robot_base).reshape((3, 3)))

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        # indices for joints in qpos, qvel
        self.robot_joints = self.robot_model.joints
        self._ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]

        # indices for joint indexes
        self._ref_joint_indexes = [
            self.sim.model.joint_name2id(joint)
            for joint in self.robot_model.joints
        ]

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.robot_model.actuators["pos"]
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.robot_model.actuators["vel"]
        ]

        self._ref_joint_torq_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.robot_model.actuators["torq"]
        ]

    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.robot_model.dof dimensions should be the desired
                normalized joint velocities and if the robot has
                a gripper, the next @self.gripper.dof dimensions should be
                actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken
        """
        raise NotImplementedError

    def visualize_gripper(self):
        """
        Do any needed visualization here.
        """
        raise NotImplementedError

    def get_observations(self, di: OrderedDict):
        """
        Returns an OrderedDict containing robot observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
        """
        raise NotImplementedError

    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.
        """
        raise NotImplementedError

    @property
    def torque_limits(self):
        """
        Action lower/upper limits per dimension.
        """
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        Action space dimension for this robot (controller dimension + gripper dof)
        """
        raise NotImplementedError

    @property
    def dof(self):
        """
        Returns the active DoF of the robot (Number of robot joints + active gripper DoF).
        """
        dof = self.robot_model.dof
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos(self.robot_model.robot_base)
        base_rot_in_world = self.sim.data.get_body_xmat(self.robot_model.robot_base).reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def _joint_positions(self):
        """
        Returns a numpy array of joint positions.
        positions are in rotation angles.
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        velocities are angular velocities.
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    @property
    def joint_indexes(self):
        """
        Returns mujoco internal indexes for the robot joints
        """
        return self._ref_joint_indexes

    def get_sensor_measurement(self, sensor_name):
        """
        Returns array of sensor values
        Args:
            sensor_name (string): name of the sensor
        """
        sensor_idx = np.sum(
            self.sim.model.sensor_dim[:self.sim.model.sensor_name2id(sensor_name)])
        sensor_dim = self.sim.model.sensor_dim[self.sim.model.sensor_name2id(sensor_name)]
        return self.sim.data.sensordata[sensor_idx: sensor_idx + sensor_dim]
