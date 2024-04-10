from collections import OrderedDict

import numpy as np

import robosuite.macros as macros
import robosuite.utils.transform_utils as T
from robosuite.models.bases import base_factory
from robosuite.models.robots import create_robot
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.buffers import DeltaBuffer
from robosuite.utils.observables import Observable, sensor


class Robot(object):
    """
    Initializes a robot simulation object, as defined by a single corresponding robot XML

    Args:
        robot_type (str): Specification for specific robot arm to be instantiated within this env (e.g: "Panda")

        idn (int or str): Unique ID of this robot. Should be different from others

        initial_qpos (sequence of float): If set, determines the initial joint positions of the robot to be
            instantiated for the task

        initialization_noise (dict): Dict containing the initialization noise parameters. The expected keys and
            corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to "None" or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            :Note: Specifying None will automatically create the required dict with "magnitude" set to 0.0

        mount_type (str): type of mount, used to instantiate mount models from mount factory.
            Default is "default", which is the default mount associated with this robot's corresponding model.
            None results in no mount, and any other (valid) model overrides the default mount.

        control_freq (float): how many control signals to receive
            in every second. This sets the amount of simulation time
            that passes between every action input.
    """

    def __init__(
        self,
        robot_type: str,
        idn=0,
        initial_qpos=None,
        initialization_noise=None,
        mount_type="default",
        control_freq=20,
    ):
        # Set relevant attributes
        self.sim = None  # MjSim this robot is tied to
        self.name = robot_type  # Specific robot to instantiate
        self.idn = idn  # Unique ID of this robot
        self.robot_model = None  # object holding robot model-specific info
        self.control_freq = control_freq  # controller Hz
        self.mount_type = mount_type  # Type of mount to use

        # Scaling of Gaussian initial noise applied to robot joints
        self.initialization_noise = initialization_noise
        if self.initialization_noise is None:
            self.initialization_noise = {"magnitude": 0.0, "type": "gaussian"}  # no noise conditions
        elif self.initialization_noise == "default":
            self.initialization_noise = {"magnitude": 0.02, "type": "gaussian"}
        self.initialization_noise["magnitude"] = (
            self.initialization_noise["magnitude"] if self.initialization_noise["magnitude"] else 0.0
        )

        self.init_qpos = initial_qpos  # n-dim list / array of robot joints

        self.robot_joints = None  # xml joint names for robot
        self.base_pos = None  # Base position in world coordinates (x,y,z)
        self.base_ori = None  # Base rotation in world coordinates (x,y,z,w quat)
        self._ref_joint_indexes = None  # xml joint indexes for robot in mjsim
        self._ref_joint_pos_indexes = None  # xml joint position indexes in mjsim
        self._ref_joint_vel_indexes = None  # xml joint velocity indexes in mjsim
        self._ref_arm_joint_actuator_indexes = None  # xml joint (torq) actuator indexes for robot in mjsim

        self.recent_qpos = None  # Current and last robot arm qpos
        self.recent_actions = None  # Current and last action applied
        self.recent_torques = None  # Current and last torques applied

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories.
        """
        raise NotImplementedError

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        self.robot_model = create_robot(self.name, idn=self.idn)

        # Add mount if specified
        if self.mount_type == "default":
            self.robot_model.add_base(base=base_factory(self.robot_model.default_base, idn=self.idn))
        else:
            self.robot_model.add_base(base=base_factory(self.mount_type, idn=self.idn))

        # Use default from robot model for initial joint positions if not specified
        if self.init_qpos is None:
            self.init_qpos = self.robot_model.init_qpos

    def reset_sim(self, sim: MjSim):
        """
        Replaces current sim with a new sim

        Args:
            sim (MjSim): New simulation being instantiated to replace the old one
        """
        self.sim = sim

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides robot joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        Args:
            deterministic (bool): If true, will not randomize initializations within the sim

        Raises:
            ValueError: [Invalid noise type]
        """
        init_qpos = np.array(self.init_qpos)
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
        self.base_pos = self.sim.data.get_body_xpos(self.robot_model.root_body)
        self.base_ori = self.sim.data.get_body_xmat(self.robot_model.root_body).reshape((3, 3))

        # Setup buffers to hold recent values
        self.recent_qpos = DeltaBuffer(dim=len(self.joint_indexes))
        self.recent_actions = DeltaBuffer(dim=self.action_dim)
        self.recent_torques = DeltaBuffer(dim=len(self.joint_indexes))

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        # indices for joints in qpos, qvel
        self.robot_joints = self.robot_model.joints
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
        self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints]

        # indices for joint indexes
        self._ref_joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.robot_model.joints]

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_arm_joint_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.actuators
        ]

    def setup_observables(self):
        """
        Sets up observables to be used for this robot

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        # Get prefix from robot model to avoid naming clashes for multiple robots and define observables modality
        pf = self.robot_model.naming_prefix
        pre_compute = f"{pf}joint_pos"
        modality = f"{pf}proprio"

        # proprioceptive features
        @sensor(modality=modality)
        def joint_pos(obs_cache):
            return np.array([self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes])

        @sensor(modality=modality)
        def joint_pos_cos(obs_cache):
            return np.cos(obs_cache[pre_compute]) if pre_compute in obs_cache else np.zeros(self.robot_model.dof)

        @sensor(modality=modality)
        def joint_pos_sin(obs_cache):
            return np.sin(obs_cache[pre_compute]) if pre_compute in obs_cache else np.zeros(self.robot_model.dof)

        @sensor(modality=modality)
        def joint_vel(obs_cache):
            return np.array([self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes])

        sensors = [joint_pos, joint_pos_cos, joint_pos_sin, joint_vel]
        names = ["joint_pos", "joint_pos_cos", "joint_pos_sin", "joint_vel"]
        # We don't want to include the direct joint pos sensor outputs
        actives = [False, True, True, True]

        # Create observables for this robot
        observables = OrderedDict()
        for name, s, active in zip(names, sensors, actives):
            obs_name = pf + name
            observables[obs_name] = Observable(
                name=obs_name,
                sensor=s,
                sampling_rate=self.control_freq,
                active=active,
            )

        return observables

    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (np.array): The control to apply to the robot. The first @self.robot_model.dof dimensions should
                be the desired normalized joint velocities and if the robot has a gripper, the next @self.gripper.dof
                dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken
        """
        raise NotImplementedError

    def check_q_limits(self):
        """
        Check if this robot is either very close or at the joint limits

        Returns:
            bool: True if this arm is near its joint limits
        """
        tolerance = 0.1
        for (qidx, (q, q_limits)) in enumerate(
            zip(self.sim.data.qpos[self._ref_joint_pos_indexes], self.sim.model.jnt_range[self._ref_joint_indexes])
        ):
            if q_limits[0] != q_limits[1] and not (q_limits[0] + tolerance < q < q_limits[1] - tolerance):
                print("Joint limit reached in joint " + str(qidx))
                return True
        return False

    def visualize(self, vis_settings):
        """
        Do any necessary visualization for this robot

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "robots" keyword as well as any other robot-specific
                options specified.
        """
        self.robot_model.set_sites_visibility(sim=self.sim, visible=vis_settings["robots"])

    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        raise NotImplementedError

    @property
    def torque_limits(self):
        """
        Torque lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) torque values
                - (np.array) maximum (high) torque values
        """
        # Torque limit values pulled from relevant robot.xml file
        low = self.sim.model.actuator_ctrlrange[self._ref_arm_joint_actuator_indexes, 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_arm_joint_actuator_indexes, 1]

        return low, high

    @property
    def action_dim(self):
        """
        Action space dimension for this robot
        """
        return self.action_limits[0].shape[0]

    @property
    def dof(self):
        """
        Returns:
            int: the active DoF of the robot (Number of robot joints + active gripper DoF).
        """
        dof = self.robot_model.dof
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.

        Args:
            name (str): Name of body in sim to grab pose

        Returns:
            np.array: (4,4) array corresponding to the pose of @name in the base frame
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos(self.robot_model.root_body)
        base_rot_in_world = self.sim.data.get_body_xmat(self.robot_model.root_body).reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.

        Args:
            jpos (np.array): Joint positions to manually set the robot to
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def js_energy(self):
        """
        Returns:
            np.array: the energy consumed by each joint between previous and current steps
        """
        # We assume in the motors torque is proportional to current (and voltage is constant)
        # In that case the amount of power scales proportional to the torque and the energy is the
        # time integral of that
        # Note that we use mean torque
        return np.abs((1.0 / self.control_freq) * self.recent_torques.average)

    @property
    def _joint_positions(self):
        """
        Returns:
            np.array: joint positions (in angles / radians)
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """
        Returns:
            np.array: joint velocities (angular velocity)
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    @property
    def joint_indexes(self):
        """
        Returns:
            list: mujoco internal indexes for the robot joints
        """
        return self._ref_joint_indexes

    def get_sensor_measurement(self, sensor_name):
        """
        Grabs relevant sensor data from the sim object

        Args:
            sensor_name (str): name of the sensor

        Returns:
            np.array: sensor values
        """
        sensor_idx = np.sum(self.sim.model.sensor_dim[: self.sim.model.sensor_name2id(sensor_name)])
        sensor_dim = self.sim.model.sensor_dim[self.sim.model.sensor_name2id(sensor_name)]
        return np.array(self.sim.data.sensordata[sensor_idx : sensor_idx + sensor_dim])
