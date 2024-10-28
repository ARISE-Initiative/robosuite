import copy
import json
import os
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config, load_part_controller_config
from robosuite.models.bases import robot_base_factory
from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import create_robot
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.buffers import DeltaBuffer, RingBuffer
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
from robosuite.utils.mjcf_utils import array_to_string
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

        base_type (str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with this robot's corresponding model.
            None results in no base, and any other (valid) model overrides the default base.

        control_freq (float): how many control signals to receive
            in every second. This sets the amount of simulation time
            that passes between every action input.
    """

    def __init__(
        self,
        robot_type: str,
        idn=0,
        composite_controller_config=None,
        initial_qpos=None,
        initialization_noise=None,
        base_type="default",
        gripper_type="default",
        control_freq=20,
        lite_physics=True,
    ):
        self.arms = REGISTERED_ROBOTS[robot_type].arms

        # TODO: Merge self.part_controller_config and self.composite_controller_config into one
        if composite_controller_config is not None:
            self.composite_controller_config = composite_controller_config
        else:
            self.composite_controller_config = load_composite_controller_config(robot=robot_type)
        self.part_controller_config = copy.deepcopy(self.composite_controller_config.get("body_parts", {}))

        self.gripper = self._input2dict(None)
        self.gripper_type = self._input2dict(gripper_type)
        self.has_gripper = self._input2dict([gripper_type is not None for _, gripper_type in self.gripper_type.items()])

        self.gripper_joints = self._input2dict(None)  # xml joint names for gripper
        self._ref_gripper_joint_pos_indexes = self._input2dict(None)  # xml gripper joint position indexes in mjsim
        self._ref_gripper_joint_vel_indexes = self._input2dict(None)  # xml gripper joint velocity indexes in mjsim
        self._ref_joint_gripper_actuator_indexes = self._input2dict(
            None
        )  # xml gripper (pos) actuator indexes for robot in mjsim
        self.eef_rot_offset = self._input2dict(None)  # rotation offsets from final arm link to gripper (quat)
        self.eef_site_id = self._input2dict(None)  # xml element id for eef in mjsim
        self.eef_cylinder_id = self._input2dict(None)  # xml element id for eef cylinder in mjsim
        self.torques = None  # Current torques being applied

        self.recent_ee_forcetorques = self._input2dict(None)  # Current and last forces / torques sensed at eef
        self.recent_ee_pose = self._input2dict(None)  # Current and last eef pose (pos + ori (quat))
        self.recent_ee_vel = self._input2dict(None)  # Current and last eef velocity
        self.recent_ee_vel_buffer = self._input2dict(None)  # RingBuffer holding prior 10 values of velocity values
        self.recent_ee_acc = self._input2dict(None)  # Current and last eef acceleration

        # Set relevant attributes
        self.sim = None  # MjSim this robot is tied to
        self.name = robot_type  # Specific robot to instantiate
        self.idn = idn  # Unique ID of this robot
        self.robot_model = None  # object holding robot model-specific info
        self.control_freq = control_freq  # controller Hz
        self.lite_physics = lite_physics
        self.base_type = base_type  # Type of robot base to use

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
        self.init_torso_qpos = None

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

        self._ref_actuators_indexes_dict = {}
        self._ref_joints_indexes_dict = {}

        self._enabled_parts = {}
        self.composite_controller = None

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories.
        """
        raise NotImplementedError

    def _postprocess_part_controller_config(self):
        """
        Update part_controller_config with values from composite_controller_config for each body part.
        Remove unused parts that are not in the controller.
        Called by _load_controller() function
        """
        for part_name, controller_config in self.composite_controller_config.get("body_parts", {}).items():
            if not self.has_part(part_name):
                ROBOSUITE_DEFAULT_LOGGER.warn(
                    f'The config has defined for the controller "{part_name}", '
                    "but the robot does not have this component. Skipping, but make sure this is intended."
                    f"Removing the controller config for {part_name} from self.part_controller_config."
                )
                self.part_controller_config.pop(part_name, None)
                continue
            if part_name in self.part_controller_config:
                self.part_controller_config[part_name].update(controller_config)

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        self.robot_model = create_robot(self.name, idn=self.idn)

        # Add base if specified
        if self.base_type == "default":
            self.robot_model.add_base(base=robot_base_factory(self.robot_model.default_base, idn=self.idn))
        else:
            self.robot_model.add_base(base=robot_base_factory(self.base_type, idn=self.idn))

        self.robot_model.update_joints()
        self.robot_model.update_actuators()
        # Use default from robot model for initial joint positions if not specified
        if self.init_qpos is None:
            self.init_qpos = self.robot_model.init_qpos

        # Now, load the gripper if necessary
        for arm in self.arms:
            if self.has_gripper[arm]:
                if self.gripper_type[arm] == "default":
                    # Load the default gripper from the robot file
                    idn = "_".join((str(self.idn), arm))
                    self.gripper[arm] = gripper_factory(self.robot_model.default_gripper[arm], idn=idn)
                else:
                    # Load user-specified gripper
                    self.gripper[arm] = gripper_factory(self.gripper_type[arm], idn="_".join((str(self.idn), arm)))
            else:
                # Load null gripper
                self.gripper[arm] = gripper_factory(None, idn="_".join((str(self.idn), arm)))
            # Grab eef rotation offset
            self.eef_rot_offset[arm] = T.quat_multiply(
                self.robot_model.hand_rotation_offset[arm], self.gripper[arm].rotation_offset
            )

            # Adjust gripper mount offset and quaternion if users specify custom values. This part is essential to support more flexible composition of new robots.
            custom_gripper_mount_pos_offset = self.robot_model.gripper_mount_pos_offset.get(arm, None)
            custom_gripper_mount_quat_offset = self.robot_model.gripper_mount_quat_offset.get(arm, None)

            # The offset of position and oriientation (quaternion) is assumed to be the very first body in the gripper xml.
            if custom_gripper_mount_pos_offset is not None:
                assert len(custom_gripper_mount_pos_offset) == 3
                # update an attribute inside an xml object
                self.gripper[arm].worldbody.find("body").attrib["pos"] = array_to_string(
                    custom_gripper_mount_pos_offset
                )
            if custom_gripper_mount_quat_offset is not None:
                assert len(custom_gripper_mount_quat_offset) == 4
                self.gripper[arm].worldbody.find("body").attrib["quat"] = array_to_string(
                    custom_gripper_mount_quat_offset
                )

            # Add this gripper to the robot model
            self.robot_model.add_gripper(self.gripper[arm], self.robot_model.eef_name[arm])

        # calling `mujoco.MjModel.from_xml_path()` resets mujoco's internal XML model and thus
        # affects get_xml() used in calls in binding_utils.py
        # https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-savelastxml
        # https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-freelastxml
        # our solution to requiring robot-only mujoco MjModels is to load the robot MjModels once first
        # then using this mujoco MjModel herein
        self.robot_model.set_mujoco_model()

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

        if self.robot_model.init_base_qpos is not None:
            self.sim.data.qpos[self._ref_base_joint_pos_indexes] = self.robot_model.init_base_qpos

        init_torso_qpos = self.init_torso_qpos
        if self.init_torso_qpos is None:
            init_torso_qpos = self.robot_model.init_torso_qpos
        if init_torso_qpos is not None:
            self.sim.data.qpos[self._ref_torso_joint_pos_indexes] = init_torso_qpos

        # Load controllers
        self._load_controller()

        # Update base pos / ori references
        self.base_pos = self.sim.data.get_body_xpos(self.robot_model.root_body)
        self.base_ori = self.sim.data.get_body_xmat(self.robot_model.root_body).reshape((3, 3))

        # Setup buffers to hold recent values
        self.recent_qpos = DeltaBuffer(dim=len(self.joint_indexes))
        self.recent_actions = DeltaBuffer(dim=self.action_dim)
        self.recent_torques = DeltaBuffer(dim=len(self.joint_indexes))

        # Setup arm-specific values
        for arm in self.arms:
            # Now, reset the grippers if necessary
            if self.has_gripper[arm]:
                if not deterministic:
                    self.sim.data.qpos[self._ref_gripper_joint_pos_indexes[arm]] = self.gripper[arm].init_qpos

                self.gripper[arm].current_action = np.zeros(self.gripper[arm].dof)

            # Setup buffers for eef values
            self.recent_ee_forcetorques[arm] = DeltaBuffer(dim=6)
            self.recent_ee_pose[arm] = DeltaBuffer(dim=7)
            self.recent_ee_vel[arm] = DeltaBuffer(dim=6)
            self.recent_ee_vel_buffer[arm] = RingBuffer(dim=6, length=10)
            self.recent_ee_acc[arm] = DeltaBuffer(dim=6)

        # reset internal variables for composite controller
        self.composite_controller.update_state()
        self.composite_controller.reset()

    def setup_references(self):
        """
        Sets up necessary reference for robots, bases, grippers, and objects.
        """
        # indices for joints in qpos, qvel
        self.robot_joints = self.robot_model.joints
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
        self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints]

        # indices for joint indexes
        self._ref_joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.robot_joints]

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_arm_joint_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.arm_actuators
        ]

        self.robot_arm_joints = self.robot_model.arm_joints
        self._ref_arm_joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.robot_arm_joints]
        self._ref_arm_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_arm_joints]
        self._ref_arm_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_arm_joints]

        # indices for base joints
        self._ref_base_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_model._base_joints
        ]
        self._ref_torso_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_model._torso_joints
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

        for arm in self.arms:
            arm_sensors, arm_sensor_names = self._create_arm_sensors(arm, modality=modality)
            sensors += arm_sensors
            names += arm_sensor_names
            actives += [True] * len(arm_sensors)

        base_sensors, base_sensor_names = self._create_base_sensors(modality=modality)
        sensors += base_sensors
        names += base_sensor_names
        actives += [True] * len(base_sensors)

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

    def _create_arm_sensors(self, arm, modality):
        """
        Helper function to create sensors for a given arm. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            arm (str): Arm to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given arm
                names (list): array of corresponding observable names
        """

        # eef features
        @sensor(modality=modality)
        def eef_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.eef_site_id[arm]])

        @sensor(modality=modality)
        def eef_quat(obs_cache):
            """
            Args:
                obs_cache (dict): A dictionary containing cached observations.

            Returns:
                numpy.ndarray: The quaternion representing the orientation of the end effector *body*
                in the mujoco world coordinate frame.

            Note:
                In robosuite<=1.5, eef_quat has been queried from the body instead
                of the site and has thus been inconsistent with the eef_pos, which queries from the site.

                This inconsistency has been raised in issue https://github.com/ARISE-Initiative/robosuite/issues/298.

                Datasets collected with robosuite<=1.4 have use the eef_quat queried from the body, so we keep this key.
                New datasets should ideally use the logic in eef_quat_site.

                In a later robosuite release, we will directly update eef_quat to query
                the orientation from the site.
            """
            return T.convert_quat(self.sim.data.get_body_xquat(self.robot_model.eef_name[arm]), to="xyzw")

        @sensor(modality=modality)
        def eef_quat_site(obs_cache):
            """
            Args:
                obs_cache (dict): A dictionary containing cached observations.

            Returns:
                numpy.ndarray: The quaternion representing the orientation of the end effector *site*
                in the mujoco world coordinate frame.

            Note:
                In robosuite<=1.5, eef_quat_site has been queried from the body instead
                of the site and has thus been inconsistent with the eef_pos, which queries from the site.

                This inconsistency has been raised in issue https://github.com/ARISE-Initiative/robosuite/issues/298

                Datasets collected with robosuite<=1.4 have use the eef_quat queried from the body, so we keep this key.
                New datasets should ideally use the logic in eef_quat_site.

                In a later robosuite release, we will directly update eef_quat to query
                the orientation from the site, and then remove this eef_quat_site key.
            """
            return T.mat2quat(self.sim.data.site_xmat[self.eef_site_id[arm]].reshape((3, 3)))

        # only consider prefix if there is more than one arm
        pf = f"{arm}_" if len(self.arms) > 1 else ""

        sensors = [eef_pos, eef_quat, eef_quat_site]
        names = [f"{pf}eef_pos", f"{pf}eef_quat", f"{pf}eef_quat_site"]

        # add in gripper sensors if this robot has a gripper
        if self.has_gripper[arm]:

            @sensor(modality=modality)
            def gripper_qpos(obs_cache):
                return np.array([self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes[arm]])

            @sensor(modality=modality)
            def gripper_qvel(obs_cache):
                return np.array([self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes[arm]])

            sensors += [gripper_qpos, gripper_qvel]
            names += [f"{pf}gripper_qpos", f"{pf}gripper_qvel"]

        return sensors, names

    def _create_base_sensors(self, modality):
        """
        Helper function to create sensors for the robot base. This will be
        overriden by subclasses.

        Args:
            modality (str): Type/modality of the created sensor
        """
        return [], []

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
                ROBOSUITE_DEFAULT_LOGGER.warn("Joint limit reached in joint " + str(qidx))
                return True
        return False

    @property
    def is_mobile(self):
        return False

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

    def _input2dict(self, inp):
        """
        Helper function that converts an input that is either a single value or a list into a dict with keys for
        each arm: "right", "left"

        Args:
            inp (str or list or None): Input value to be converted to dict

            :Note: If inp is a list, then assumes format is [right, left]

        Returns:
            dict: Inputs mapped for each robot arm
        """
        # First, convert to list if necessary
        if type(inp) is not list:
            inp = [inp for _ in range(2)]
        # Now, convert list to dict and return
        return {key: copy.deepcopy(value) for key, value in zip(self.arms, inp)}

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
        return self.robot_model.dof

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

    @property
    def arm_joint_indexes(self):
        """
        Returns:
            list: mujoco internal indexes for the robot arm joints
        """
        return self._ref_arm_joint_indexes

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

    def visualize(self, vis_settings):
        """
        Do any necessary visualization for this manipulator

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "robots" and "grippers" keyword as well as any other
                robot-specific options specified.
        """
        self.robot_model.set_sites_visibility(sim=self.sim, visible=vis_settings["robots"])
        self._visualize_grippers(visible=vis_settings["grippers"])

    def _visualize_grippers(self, visible):
        """
        Visualizes the gripper site(s) if applicable.

        Args:
            visible (bool): True if visualizing the gripper for this arm.
        """
        for arm in self.arms:
            self.gripper[arm].set_sites_visibility(sim=self.sim, visible=visible)

    @property
    def action_limits(self):
        raise NotImplementedError

    @property
    def is_mobile(self):
        return NotImplementedError

    @property
    def ee_ft_integral(self):
        """
        Returns:
            dict: each arm-specific entry specifies the integral over time of the applied ee force-torque for that arm
        """
        vals = {}
        for arm in self.arms:
            vals[arm] = np.abs((1.0 / self.control_freq) * self.recent_ee_forcetorques[arm].average)
        return vals

    @property
    def ee_force(self):
        """
        Returns:
            dict: each arm-specific entry specifies the force applied at the force sensor at the robot arm's eef
        """
        vals = {}
        for arm in self.arms:
            vals[arm] = self.get_sensor_measurement(self.gripper[arm].important_sensors["force_ee"])
        return vals

    @property
    def ee_torque(self):
        """
        Returns:
            dict: each arm-specific entry specifies the torque applied at the torque sensor at the robot arm's eef
        """
        vals = {}
        for arm in self.arms:
            vals[arm] = self.get_sensor_measurement(self.gripper[arm].important_sensors["torque_ee"])
        return vals

    @property
    def _hand_pose(self):
        """
        Returns:
            dict: each arm-specific entry specifies the eef pose in base frame of robot.
        """
        vals = {}
        for arm in self.arms:
            vals[arm] = self.pose_in_base_from_name(self.robot_model.eef_name[arm])
        return vals

    @property
    def _hand_quat(self):
        """
        Returns:
            dict: each arm-specific entry specifies the eef quaternion in base frame of robot.
        """
        vals = {}
        orns = self._hand_orn
        for arm in self.arms:
            vals[arm] = T.mat2quat(orns[arm])
        return vals

    @property
    def _hand_total_velocity(self):
        """
        Returns:
            dict: each arm-specific entry specifies the total eef velocity (linear + angular) in the base frame
            as a numpy array of shape (6,)
        """
        vals = {}
        for arm in self.arms:
            # Determine correct start, end points based on arm
            (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)

            # Use jacobian to translate joint velocities to end effector velocities.
            Jp = self.sim.data.get_body_jacp(self.robot_model.eef_name[arm]).reshape((3, -1))
            Jp_joint = Jp[:, self._ref_joint_vel_indexes[start:end]]

            Jr = self.sim.data.get_body_jacr(self.robot_model.eef_name[arm]).reshape((3, -1))
            Jr_joint = Jr[:, self._ref_joint_vel_indexes[start:end]]

            eef_lin_vel = Jp_joint.dot(self._joint_velocities)
            eef_rot_vel = Jr_joint.dot(self._joint_velocities)
            vals[arm] = np.concatenate([eef_lin_vel, eef_rot_vel])
        return vals

    @property
    def _hand_pos(self):
        """
        Returns:
            dict: each arm-specific entry specifies the position of eef in base frame of robot.
        """
        vals = {}
        poses = self._hand_pose
        for arm in self.arms:
            eef_pose_in_base = poses[arm]
            vals[arm] = eef_pose_in_base[:3, 3]
        return vals

    @property
    def _hand_orn(self):
        """
        Returns:
            dict: each arm-specific entry specifies the orientation of eef in base frame of robot as a rotation matrix.
        """
        vals = {}
        poses = self._hand_pose
        for arm in self.arms:
            eef_pose_in_base = poses[arm]
            vals[arm] = eef_pose_in_base[:3, :3]
        return vals

    @property
    def _hand_vel(self):
        """
        Returns:
            dict: each arm-specific entry specifies the velocity of eef in base frame of robot.
        """
        vels = self._hand_total_velocity
        for arm in self.arms:
            vels[arm] = vels[arm][:3]
        return vels

    @property
    def _hand_ang_vel(self):
        """
        Returns:
            dict: each arm-specific entry specifies the angular velocity of eef in base frame of robot.
        """
        vels = self._hand_total_velocity
        for arm in self.arms:
            vels[arm] = vels[arm][3:]
        return vels

    def _load_arm_controllers(self):
        urdf_loaded = False
        # Load composite controller configs for both left and right arm
        for arm in self.arms:
            # Assert that the controller config is a dict file:
            #             NOTE: "type" must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
            #                                           OSC_POSITION, OSC_POSE, IK_POSE}
            assert (
                type(self.part_controller_config[arm]) == dict
            ), "Inputted controller config must be a dict! Instead, got type: {}".format(
                type(self.part_controller_config[arm])
            )

            # Add to the controller dict additional relevant params:
            #   the robot name, mujoco sim, eef_name, actuator_range, joint_indexes, timestep (model) freq,
            #   policy (control) freq, and ndim (# joints)
            self.part_controller_config[arm]["robot_name"] = self.name
            self.part_controller_config[arm]["sim"] = self.sim
            self.part_controller_config[arm]["ref_name"] = self.gripper[arm].important_sites["grip_site"]
            self.part_controller_config[arm]["part_name"] = arm
            self.part_controller_config[arm]["naming_prefix"] = self.robot_model.naming_prefix

            self.part_controller_config[arm]["eef_rot_offset"] = self.eef_rot_offset[arm]
            self.part_controller_config[arm]["ndim"] = self._joint_split_idx
            self.part_controller_config[arm]["policy_freq"] = self.control_freq
            self.part_controller_config[arm]["lite_physics"] = self.lite_physics
            (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)
            self.part_controller_config[arm]["joint_indexes"] = {
                "joints": self.arm_joint_indexes[start:end],
                "qpos": self._ref_arm_joint_pos_indexes[start:end],
                "qvel": self._ref_arm_joint_vel_indexes[start:end],
            }
            self.part_controller_config[arm]["actuator_range"] = (
                self.torque_limits[0][start:end],
                self.torque_limits[1][start:end],
            )

            # Only load urdf the first time this controller gets called
            self.part_controller_config[arm]["load_urdf"] = True if not urdf_loaded else False
            urdf_loaded = True

            # # Instantiate the relevant controller

            if self.has_gripper[arm]:
                # Load gripper controllers
                assert "gripper" in self.part_controller_config[arm], "Gripper controller config not found!"
                gripper_name = self.get_gripper_name(arm)
                self.part_controller_config[gripper_name] = {}
                self.part_controller_config[gripper_name]["type"] = self.part_controller_config[arm]["gripper"]["type"]
                self.part_controller_config[gripper_name]["use_action_scaling"] = self.part_controller_config[arm][
                    "gripper"
                ].get("use_action_scaling", True)
                self.part_controller_config[gripper_name]["robot_name"] = self.name
                self.part_controller_config[gripper_name]["sim"] = self.sim
                self.part_controller_config[gripper_name]["eef_name"] = self.gripper[arm].important_sites["grip_site"]
                self.part_controller_config[gripper_name]["part_name"] = gripper_name
                self.part_controller_config[gripper_name]["naming_prefix"] = self.robot_model.naming_prefix
                self.part_controller_config[gripper_name]["ndim"] = self.gripper[arm].dof
                self.part_controller_config[gripper_name]["policy_freq"] = self.control_freq
                self.part_controller_config[gripper_name]["joint_indexes"] = {
                    "joints": self.gripper_joints[arm],
                    "actuators": self._ref_joint_gripper_actuator_indexes[arm],
                    "qpos": self._ref_gripper_joint_pos_indexes[arm],
                    "qvel": self._ref_gripper_joint_vel_indexes[arm],
                }
                low = self.sim.model.actuator_ctrlrange[self._ref_joint_gripper_actuator_indexes[arm], 0]
                high = self.sim.model.actuator_ctrlrange[self._ref_joint_gripper_actuator_indexes[arm], 1]

                self.part_controller_config[gripper_name]["actuator_range"] = (low, high)

    def enable_parts(self, right=True, left=True):
        self._enabled_parts = {
            "right": right,
            "right_gripper": right,
            "left": left,
            "left_gripper": left,
        }

    def enabled(self, part_name):
        return self._enabled_parts[part_name]

    def create_action_vector(self, action_dict):
        """
        A helper function that creates the action vector given a dictionary
        """
        # check if there's a composite controller and if the controller has a create_action_vector method
        if self.composite_controller is not None and hasattr(self.composite_controller, "create_action_vector"):
            return self.composite_controller.create_action_vector(action_dict)
        else:
            full_action_vector = np.zeros(self.action_dim)
            for (part_name, action_vector) in action_dict.items():
                if part_name not in self._action_split_indexes:
                    ROBOSUITE_DEFAULT_LOGGER.debug(f"{part_name} is not specified in the action space")
                    continue
                start_idx, end_idx = self._action_split_indexes[part_name]
                if end_idx - start_idx == 0:
                    # skipping not controlling actions
                    continue
                assert len(action_vector) == (end_idx - start_idx), ROBOSUITE_DEFAULT_LOGGER.error(
                    f"Action vector for {part_name} is not the correct size. Expected {end_idx - start_idx} for {part_name}, got {len(action_vector)}"
                )
                full_action_vector[start_idx:end_idx] = action_vector
            return full_action_vector

    def print_action_info(self):
        if self.composite_controller is not None and hasattr(self.composite_controller, "print_action_info"):
            self.composite_controller.print_action_info()
            return
        action_index_info = []
        action_dim_info = []
        for part_name, (start_idx, end_idx) in self._action_split_indexes.items():
            action_dim_info.append(f"{part_name}: {(end_idx - start_idx)} dim")
            action_index_info.append(f"{part_name}: {start_idx}:{end_idx}")

        action_dim_info_str = ", ".join(action_dim_info)
        ROBOSUITE_DEFAULT_LOGGER.info(f"Action Dimensions: [{action_dim_info_str}]")

        action_index_info_str = ", ".join(action_index_info)
        ROBOSUITE_DEFAULT_LOGGER.info(f"Action Indices: [{action_index_info_str}]")

    def print_action_info_dict(self):
        if self.composite_controller is not None and hasattr(self.composite_controller, "print_action_info_dict"):
            self.composite_controller.print_action_info_dict(name=self.name)
            return
        info_dict = {}
        info_dict["Action Dimension"] = self.action_dim
        info_dict.update(dict(self._action_split_indexes))

        info_dict_str = f"\nAction Info for {self.name}:\n\n{json.dumps(dict(info_dict), indent=4)}"
        ROBOSUITE_DEFAULT_LOGGER.info(info_dict_str)

    def get_gripper_name(self, arm):
        return f"{arm}_gripper"

    def has_part(self, part_name):
        if part_name in self._ref_joints_indexes_dict.keys() and len(self._ref_joints_indexes_dict[part_name]) > 0:
            return True
        else:
            return False

    @property
    def _joint_split_idx(self):
        """
        Returns:
            int: the index that correctly splits the right arm from the left arm joints
        """
        return int(len(self.robot_arm_joints) / len(self.arms))

    @property
    def part_controllers(self):
        """
        Controller dictionary for the robot

        Returns:
            dict: Controller dictionary for the robot
        """
        return self.composite_controller.part_controllers
