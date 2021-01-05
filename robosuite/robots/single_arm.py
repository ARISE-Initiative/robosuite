import numpy as np

from collections import OrderedDict

import robosuite.utils.transform_utils as T

from robosuite.models.grippers import gripper_factory
from robosuite.controllers import controller_factory, load_controller_config

from robosuite.robots.manipulator import Manipulator
from robosuite.utils.buffers import DeltaBuffer, RingBuffer
from robosuite.utils.observables import Observable, sensor

import os
import copy


class SingleArm(Manipulator):
    """
    Initializes a single-armed robot simulation object.

    Args:
        robot_type (str): Specification for specific robot arm to be instantiated within this env (e.g: "Panda")

        idn (int or str): Unique ID of this robot. Should be different from others

        controller_config (dict): If set, contains relevant controller parameters for creating a custom controller.
            Else, uses the default controller for this specific task

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

        gripper_type (str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default gripper associated
            within the 'robot' specification. None removes the gripper, and any other (valid) model overrides the
            default gripper

        control_freq (float): how many control signals to receive
            in every second. This sets the amount of simulation time
            that passes between every action input.
    """

    def __init__(
        self,
        robot_type: str,
        idn=0,
        controller_config=None,
        initial_qpos=None,
        initialization_noise=None,
        mount_type="default",
        gripper_type="default",
        control_freq=20
    ):

        self.controller = None
        self.controller_config = copy.deepcopy(controller_config)
        self.gripper_type = gripper_type
        self.has_gripper = self.gripper_type is not None

        self.gripper = None                                 # Gripper class
        self.gripper_joints = None                          # xml joint names for gripper
        self._ref_gripper_joint_pos_indexes = None          # xml gripper joint position indexes in mjsim
        self._ref_gripper_joint_vel_indexes = None          # xml gripper joint velocity indexes in mjsim
        self._ref_joint_gripper_actuator_indexes = None     # xml gripper (pos) actuator indexes for robot in mjsim
        self.eef_rot_offset = None                          # rotation offsets from final arm link to gripper (quat)
        self.eef_site_id = None                             # xml element id for eef in mjsim
        self.eef_cylinder_id = None                         # xml element id for eef cylinder in mjsim
        self.torques = None                                 # Current torques being applied

        self.recent_ee_forcetorques = None                  # Current and last forces / torques sensed at eef
        self.recent_ee_pose = None                          # Current and last eef pose (pos + ori (quat))
        self.recent_ee_vel = None                           # Current and last eef velocity
        self.recent_ee_vel_buffer = None                    # RingBuffer holding prior 10 values of velocity values
        self.recent_ee_acc = None                           # Current and last eef acceleration

        super().__init__(
            robot_type=robot_type,
            idn=idn,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
            mount_type=mount_type,
            control_freq=control_freq,
        )

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        # First, load the default controller if none is specified
        if not self.controller_config:
            # Need to update default for a single agent
            controller_path = os.path.join(os.path.dirname(__file__), '..',
                                           'controllers/config/{}.json'.format(
                                               self.robot_model.default_controller_config))
            self.controller_config = load_controller_config(custom_fpath=controller_path)

        # Assert that the controller config is a dict file:
        #             NOTE: "type" must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
        #                                           OSC_POSITION, OSC_POSE, IK_POSE}
        assert type(self.controller_config) == dict, \
            "Inputted controller config must be a dict! Instead, got type: {}".format(type(self.controller_config))

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self.controller_config["robot_name"] = self.name
        self.controller_config["sim"] = self.sim
        self.controller_config["eef_name"] = self.gripper.important_sites["grip_site"]
        self.controller_config["eef_rot_offset"] = self.eef_rot_offset
        self.controller_config["joint_indexes"] = {
            "joints": self.joint_indexes,
            "qpos": self._ref_joint_pos_indexes,
            "qvel": self._ref_joint_vel_indexes
                                              }
        self.controller_config["actuator_range"] = self.torque_limits
        self.controller_config["policy_freq"] = self.control_freq
        self.controller_config["ndim"] = len(self.robot_joints)

        # Instantiate the relevant controller
        self.controller = controller_factory(self.controller_config["type"], self.controller_config)

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        # First, run the superclass method to load the relevant model
        super().load_model()

        # Verify that the loaded model is of the correct type for this robot
        if self.robot_model.arm_type != "single":
            raise TypeError("Error loading robot model: Incompatible arm type specified for this robot. "
                            "Requested model arm type: {}, robot arm type: {}"
                            .format(self.robot_model.arm_type, type(self)))

        # Now, load the gripper if necessary
        if self.has_gripper:
            if self.gripper_type == 'default':
                # Load the default gripper from the robot file
                self.gripper = gripper_factory(self.robot_model.default_gripper, idn=self.idn)
            else:
                # Load user-specified gripper
                self.gripper = gripper_factory(self.gripper_type, idn=self.idn)
        else:
            # Load null gripper
            self.gripper = gripper_factory(None, idn=self.idn)
        # Grab eef rotation offset
        self.eef_rot_offset = T.quat_multiply(self.robot_model.hand_rotation_offset, self.gripper.rotation_offset)
        # Add gripper to this robot model
        self.robot_model.add_gripper(self.gripper)

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides gripper joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        Args:
            deterministic (bool): If true, will not randomize initializations within the sim
        """
        # First, run the superclass method to reset the position and controller
        super().reset(deterministic)

        if not deterministic:
            # Now, reset the gripper if necessary
            if self.has_gripper:
                self.sim.data.qpos[
                    self._ref_gripper_joint_pos_indexes
                ] = self.gripper.init_qpos

        # Update base pos / ori references in controller
        self.controller.update_base_pose(self.base_pos, self.base_ori)

        # # Setup buffers to hold recent values
        self.recent_ee_forcetorques = DeltaBuffer(dim=6)
        self.recent_ee_pose = DeltaBuffer(dim=7)
        self.recent_ee_vel = DeltaBuffer(dim=6)
        self.recent_ee_vel_buffer = RingBuffer(dim=6, length=10)
        self.recent_ee_acc = DeltaBuffer(dim=6)

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.

        Note that this should get called during every reset from the environment
        """
        # First, run the superclass method to setup references for joint-related values / indexes
        super().setup_references()

        # Now, add references to gripper if necessary
        # indices for grippers in qpos, qvel
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self._ref_gripper_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
            ]
            self._ref_gripper_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
            ]
            self._ref_joint_gripper_actuator_indexes = [
                self.sim.model.actuator_name2id(actuator)
                for actuator in self.gripper.actuators
            ]

        # IDs of sites for eef visualization
        self.eef_site_id = self.sim.model.site_name2id(self.gripper.important_sites["grip_site"])
        self.eef_cylinder_id = self.sim.model.site_name2id(self.gripper.important_sites["grip_cylinder"])

    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (np.array): The control to apply to the robot. The first @self.robot_model.dof dimensions should be
                the desired normalized joint velocities and if the robot has a gripper, the next @self.gripper.dof
                dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """

        # clip actions into valid range
        assert len(action) == self.action_dim, \
            "environment got invalid action dimension -- expected {}, got {}".format(
                self.action_dim, len(action))

        gripper_action = None
        if self.has_gripper:
            gripper_action = action[self.controller.control_dim:]  # all indexes past controller dimension indexes
            arm_action = action[:self.controller.control_dim]
        else:
            arm_action = action

        # Update the controller goal if this is a new policy step
        if policy_step:
            self.controller.set_goal(arm_action)

        # Now run the controller for a step
        torques = self.controller.run_controller()

        # Clip the torques
        low, high = self.torque_limits
        self.torques = np.clip(torques, low, high)

        # Get gripper action, if applicable
        if self.has_gripper:
            self.grip_action(gripper=self.gripper, gripper_action=gripper_action)

        # Apply joint torque control
        self.sim.data.ctrl[self._ref_joint_actuator_indexes] = self.torques

        # If this is a policy step, also update buffers holding recent values of interest
        if policy_step:
            # Update proprioceptive values
            self.recent_qpos.push(self._joint_positions)
            self.recent_actions.push(action)
            self.recent_torques.push(self.torques)
            self.recent_ee_forcetorques.push(np.concatenate((self.ee_force, self.ee_torque)))
            self.recent_ee_pose.push(np.concatenate((self.controller.ee_pos, T.mat2quat(self.controller.ee_ori_mat))))
            self.recent_ee_vel.push(np.concatenate((self.controller.ee_pos_vel, self.controller.ee_ori_vel)))

            # Estimation of eef acceleration (averaged derivative of recent velocities)
            self.recent_ee_vel_buffer.push(np.concatenate((self.controller.ee_pos_vel, self.controller.ee_ori_vel)))
            diffs = np.vstack([self.recent_ee_acc.current,
                               self.control_freq * np.diff(self.recent_ee_vel_buffer.buf, axis=0)])
            ee_acc = np.array([np.convolve(col, np.ones(10) / 10., mode='valid')[0] for col in diffs.transpose()])
            self.recent_ee_acc.push(ee_acc)

    def _visualize_grippers(self, visible):
        """
        Visualizes the gripper site(s) if applicable.

        Args:
            visible (bool): True if visualizing the gripper for this arm.
        """
        self.gripper.set_sites_visibility(sim=self.sim, visible=visible)

    def setup_observables(self):
        """
        Sets up observables to be used for this robot

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        # Get general robot observables first
        observables = super().setup_observables()

        # Get prefix from robot model to avoid naming clashes for multiple robots and define observables modality
        pf = self.robot_model.naming_prefix
        modality = f"{pf}proprio"

        # eef features
        @sensor(modality=modality)
        def eef_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.eef_site_id])

        @sensor(modality=modality)
        def eef_quat(obs_cache):
            return T.convert_quat(self.sim.data.get_body_xquat(self.robot_model.eef_name), to="xyzw")

        sensors = [eef_pos, eef_quat]
        names = [f"{pf}eef_pos", f"{pf}eef_quat"]

        # add in gripper sensors if this robot has a gripper
        if self.has_gripper:
            @sensor(modality=modality)
            def gripper_qpos(obs_cache):
                return np.array([self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes])

            @sensor(modality=modality)
            def gripper_qvel(obs_cache):
                return np.array([self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes])

            sensors += [gripper_qpos, gripper_qvel]
            names += [f"{pf}gripper_qpos", f"{pf}gripper_qvel"]

        # Create observables for this robot
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        # Action limits based on controller limits
        low, high = ([-1] * self.gripper.dof, [1] * self.gripper.dof) if self.has_gripper else ([], [])
        low_c, high_c = self.controller.control_limits
        low = np.concatenate([low_c, low])
        high = np.concatenate([high_c, high])

        return low, high

    @property
    def ee_ft_integral(self):
        """
        Returns:
            np.array: the integral over time of the applied ee force-torque
        """
        return np.abs((1.0 / self.control_freq) * self.recent_ee_forcetorques.average)

    @property
    def ee_force(self):
        """
        Returns:
            np.array: force applied at the force sensor at the robot arm's eef
        """
        return self.get_sensor_measurement(self.gripper.important_sensors["force_ee"])

    @property
    def ee_torque(self):
        """
        Returns torque applied at the torque sensor at the robot arm's eef
        """
        return self.get_sensor_measurement(self.gripper.important_sensors["torque_ee"])

    @property
    def _hand_pose(self):
        """
        Returns:
            np.array: (4,4) array corresponding to the eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name(self.robot_model.eef_name)

    @property
    def _hand_quat(self):
        """
        Returns:
            np.array: (x,y,z,w) eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._hand_orn)

    @property
    def _hand_total_velocity(self):
        """
        Returns:
            np.array: 6-array representing the total eef velocity (linear + angular) in the base frame
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp(self.robot_model.eef_name).reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes]

        Jr = self.sim.data.get_body_jacr(self.robot_model.eef_name).reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _hand_pos(self):
        """
        Returns:
            np.array: 3-array representing the position of eef in base frame of robot.
        """
        eef_pose_in_base = self._hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _hand_orn(self):
        """
        Returns:
            np.array: (3,3) array representing the orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _hand_vel(self):
        """
        Returns:
            np.array: (x,y,z) velocity of eef in base frame of robot.
        """
        return self._hand_total_velocity[:3]

    @property
    def _hand_ang_vel(self):
        """
        Returns:
            np.array: (ax,ay,az) angular velocity of eef in base frame of robot.
        """
        return self._hand_total_velocity[3:]
