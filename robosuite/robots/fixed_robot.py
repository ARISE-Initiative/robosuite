import copy
import os
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers import controller_factory, load_controller_config
from robosuite.robots.robot import Robot
from robosuite.utils.observables import Observable, sensor


class FixedRobot(Robot):
    """
    Initializes a robot with a fixed base.
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
        control_freq=20,
    ):
        super().__init__(
            robot_type=robot_type,
            idn=idn,
            controller_config=controller_config,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
            mount_type=mount_type,
            gripper_type=gripper_type,
            control_freq=control_freq,
        )

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        # Flag for loading urdf once (only applicable for IK controllers)
        urdf_loaded = False

        # Load controller configs for both left and right arm
        for arm in self.arms:
            # First, load the default controller if none is specified
            if not self.controller_config[arm]:
                # Need to update default for a single agent
                controller_path = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "controllers/config/{}.json".format(self.robot_model.default_controller_config[arm]),
                )
                self.controller_config[arm] = load_controller_config(custom_fpath=controller_path)

            # Assert that the controller config is a dict file:
            #             NOTE: "type" must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
            #                                           OSC_POSITION, OSC_POSE, IK_POSE}
            assert (
                type(self.controller_config[arm]) == dict
            ), "Inputted controller config must be a dict! Instead, got type: {}".format(
                type(self.controller_config[arm])
            )

            # Add to the controller dict additional relevant params:
            #   the robot name, mujoco sim, eef_name, actuator_range, joint_indexes, timestep (model) freq,
            #   policy (control) freq, and ndim (# joints)
            self.controller_config[arm]["robot_name"] = self.name
            self.controller_config[arm]["sim"] = self.sim
            self.controller_config[arm]["eef_name"] = self.gripper[arm].important_sites["grip_site"]
            self.controller_config[arm]["eef_rot_offset"] = self.eef_rot_offset[arm]
            self.controller_config[arm]["ndim"] = self._joint_split_idx
            self.controller_config[arm]["policy_freq"] = self.control_freq
            (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)
            self.controller_config[arm]["joint_indexes"] = {
                "joints": self.joint_indexes[start:end],
                "qpos": self._ref_joint_pos_indexes[start:end],
                "qvel": self._ref_joint_vel_indexes[start:end],
            }
            self.controller_config[arm]["actuator_range"] = (
                self.torque_limits[0][start:end],
                self.torque_limits[1][start:end],
            )

            # Only load urdf the first time this controller gets called
            self.controller_config[arm]["load_urdf"] = True if not urdf_loaded else False
            urdf_loaded = True

            # Instantiate the relevant controller
            self.controller[arm] = controller_factory(self.controller_config[arm]["type"], self.controller_config[arm])

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        # First, run the superclass method to load the relevant model
        super().load_model()

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides gripper joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        Args:
            deterministic (bool): If true, will not randomize initializations within the sim
        """
        # First, run the superclass method to reset the position and controller
        super().reset(deterministic)

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.

        Note that this should get called during every reset from the environment
        """
        # First, run the superclass method to setup references for joint-related values / indexes
        super().setup_references()

        # Now, add references to gripper if necessary
        # indices for grippers in qpos, qvel
        for arm in self.arms:
            if self.has_gripper[arm]:
                self.gripper_joints[arm] = list(self.gripper[arm].joints)
                self._ref_gripper_joint_pos_indexes[arm] = [
                    self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints[arm]
                ]
                self._ref_gripper_joint_vel_indexes[arm] = [
                    self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints[arm]
                ]
                self._ref_joint_gripper_actuator_indexes[arm] = [
                    self.sim.model.actuator_name2id(actuator) for actuator in self.gripper[arm].actuators
                ]

            # IDs of sites for eef visualization
            self.eef_site_id[arm] = self.sim.model.site_name2id(self.gripper[arm].important_sites["grip_site"])
            self.eef_cylinder_id[arm] = self.sim.model.site_name2id(self.gripper[arm].important_sites["grip_cylinder"])

    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (np.array): The control to apply to the robot. The first @self.robot_model.dof dimensions should
                be the desired normalized joint velocities and if the robot has a gripper, the next @self.gripper.dof
                dimensions should be actuation controls for the gripper.

                :NOTE: Assumes inputted actions are of form:
                    [right_arm_control, right_gripper_control, left_arm_control, left_gripper_control]

            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """
        # clip actions into valid range
        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )

        for arm in self.arms:
            (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)
            # self.controller[arm].update_initial_joints(self.sim.data.qpos[self._ref_joint_pos_indexes[start:end]])
            self.controller[arm].update_base_pose(self.base_pos, self.base_ori)

        self.torques = np.array([])
        # Now execute actions for each arm
        for arm in self.arms:
            # Make sure to split action space correctly
            (start, end) = (None, self._action_split_idx) if arm == "right" else (self._action_split_idx, None)
            sub_action = action[start:end]

            gripper_action = None
            if self.has_gripper[arm]:
                # get all indexes past controller dimension indexes
                gripper_action = sub_action[self.controller[arm].control_dim :]
                sub_action = sub_action[: self.controller[arm].control_dim]

            # Update the controller goal if this is a new policy step
            if policy_step:
                self.controller[arm].set_goal(sub_action)

            # Now run the controller for a step and add it to the torques
            self.torques = np.concatenate((self.torques, self.controller[arm].run_controller()))

            # Get gripper action, if applicable
            if self.has_gripper[arm]:
                self.grip_action(gripper=self.gripper[arm], gripper_action=gripper_action)

        # Clip the torques
        low, high = self.torque_limits
        self.torques = np.clip(self.torques, low, high)

        # Apply joint torque control
        self.sim.data.ctrl[self._ref_joint_actuator_indexes] = self.torques

        # If this is a policy step, also update buffers holding recent values of interest
        if policy_step:
            # Update proprioceptive values
            self.recent_qpos.push(self._joint_positions)
            self.recent_actions.push(action)
            self.recent_torques.push(self.torques)

            for arm in self.arms:
                # Update arm-specific proprioceptive values
                self.recent_ee_forcetorques[arm].push(np.concatenate((self.ee_force[arm], self.ee_torque[arm])))
                self.recent_ee_pose[arm].push(
                    np.concatenate((self.controller[arm].ee_pos, T.mat2quat(self.controller[arm].ee_ori_mat)))
                )
                self.recent_ee_vel[arm].push(
                    np.concatenate((self.controller[arm].ee_pos_vel, self.controller[arm].ee_ori_vel))
                )

                # Estimation of eef acceleration (averaged derivative of recent velocities)
                self.recent_ee_vel_buffer[arm].push(
                    np.concatenate((self.controller[arm].ee_pos_vel, self.controller[arm].ee_ori_vel))
                )
                diffs = np.vstack(
                    [
                        self.recent_ee_acc[arm].current,
                        self.control_freq * np.diff(self.recent_ee_vel_buffer[arm].buf, axis=0),
                    ]
                )
                ee_acc = np.array([np.convolve(col, np.ones(10) / 10.0, mode="valid")[0] for col in diffs.transpose()])
                self.recent_ee_acc[arm].push(ee_acc)

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
        sensors = []
        names = []

        for arm in self.arms:
            # Add in eef info
            arm_sensors, arm_sensor_names = self._create_arm_sensors(arm=arm, modality=modality)
            sensors += arm_sensors
            names += arm_sensor_names

        # Create observables for this robot
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
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
        pf = self.robot_model.naming_prefix

        # eef features
        @sensor(modality=modality)
        def eef_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.eef_site_id[arm]])

        @sensor(modality=modality)
        def eef_quat(obs_cache):
            return T.convert_quat(self.sim.data.get_body_xquat(self.robot_model.eef_name[arm]), to="xyzw")

        sensors = [eef_pos, eef_quat]
        names = [f"{pf}{arm}_eef_pos", f"{pf}{arm}_eef_quat"]

        # add in gripper sensors if this robot has a gripper
        if self.has_gripper[arm]:

            @sensor(modality=modality)
            def gripper_qpos(obs_cache):
                return np.array([self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes[arm]])

            @sensor(modality=modality)
            def gripper_qvel(obs_cache):
                return np.array([self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes[arm]])

            sensors += [gripper_qpos, gripper_qvel]
            names += [f"{pf}{arm}_gripper_qpos", f"{pf}{arm}_gripper_qvel"]

        return sensors, names

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
        low, high = [], []
        for arm in self.arms:
            low_g, high_g = (
                ([-1] * self.gripper[arm].dof, [1] * self.gripper[arm].dof) if self.has_gripper[arm] else ([], [])
            )
            low_c, high_c = self.controller[arm].control_limits
            low, high = np.concatenate([low, low_c, low_g]), np.concatenate([high, high_c, high_g])
        return low, high

    @property
    def _action_split_idx(self):
        """
        Grabs the index that correctly splits the right arm from the left arm actions

        :NOTE: Assumes inputted actions are of form:
            [right_arm_control, right_gripper_control, left_arm_control, left_gripper_control]

        Returns:
            int: Index splitting right from left arm actions
        """
        return (
            self.controller["right"].control_dim + self.gripper["right"].dof
            if self.has_gripper["right"]
            else self.controller["right"].control_dim
        )

    @property
    def _joint_split_idx(self):
        """
        Returns:
            int: the index that correctly splits the right arm from the left arm joints
        """
        return int(len(self.robot_joints) / len(self.arms))

    @property
    def is_mobile(self):
        return False
