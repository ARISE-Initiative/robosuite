import copy
import os
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers import controller_factory, load_controller_config
from robosuite.controllers.mobile_base_controller import MobileBaseController
from robosuite.controllers.torso_height_controller import TorsoHeightController
from robosuite.robots.mobile_base_robot import MobileBaseRobot
from robosuite.utils.observables import Observable, sensor


class WheeledRobot(MobileBaseRobot):
    """
    Initializes a robot with a wheeled base.
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

    def _load_torso_controller(self):
        """
        Load torso controller
        """
        # if not self.controller_config[self.torso]:
        #         # Need to update default for a single agent
        #         controller_path = os.path.join(
        #             os.path.dirname(__file__),
        #             "..",
        #             "controllers/config/torso/{}.json".format(self.robot_model.default_controller_config[self.torso]),
        #         )
        #         self.controller_config[self.torso] = load_controller_config(custom_fpath=controller_path)
        # TODO: Add a default controller config for torso
        self.controller_config[self.torso] = {}
        self.controller_config[self.torso]["type"] = "JOINT_POSITION"
        self.controller_config[self.torso]["interpolation"] = "linear"
        self.controller_config[self.torso]["ramp_ratio"] = 1.0
        self.controller_config[self.torso]["robot_name"] = self.name
        self.controller_config[self.torso]["sim"] = self.sim
        self.controller_config[self.torso]["part_name"] = self.torso
        self.controller_config[self.torso]["naming_prefix"] = self.robot_model.naming_prefix
        self.controller_config[self.torso]["ndim"] = self._joint_split_idx
        self.controller_config[self.torso]["policy_freq"] = self.control_freq

        ref_torso_joint_indexes = [self.sim.model.joint_name2id(x) for x in self.robot_model.torso_joints]
        ref_torso_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_model.torso_joints]
        ref_torso_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_model.torso_joints]
        self.controller_config[self.torso]["joint_indexes"] = {
            "joints": ref_torso_joint_indexes,
            "qpos": ref_torso_joint_pos_indexes,
            "qvel": ref_torso_joint_vel_indexes,
        }

        self.controller_config[self.torso]["actuator_range"] = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.torso][0]]
        import pdb; pdb.set_trace()

        # self.controller[self.torso] = TorsoHeightController(**self.controller_config[self.torso])
        self.controller[self.torso] = controller_factory(self.controller_config[self.torso]["type"], self.controller_config[self.torso])
        # self.controller_config[self.torso]["actuator_range"] = (
            
        # )

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        # Flag for loading urdf once (only applicable for IK controllers)

        self._load_arm_controllers()

        self.controller[self.base] = MobileBaseController(self.sim, self.robot_model.base.naming_prefix)

        self._load_torso_controller()

        # self.controller[self.head] = controller_factory("OSC_POSE", self.controller_config["right"])

        # Set up split indices for arm actions
        self._action_split_indexes = OrderedDict()
        previous_idx = None
        last_idx = 0
        for arm in self.arms:
            last_idx += self.controller[arm].control_dim
            last_idx += self.gripper[arm].dof if self.has_gripper[arm] else 0
            self._action_split_indexes[arm] = (previous_idx, last_idx)
            previous_idx = last_idx

        previous_idx = self._action_split_indexes[self.arms[-1]][1]
        last_idx = previous_idx
        for part_name in [self.base, self.head, self.torso]:
            if part_name == self.head:
                last_idx += 2
            else:
                last_idx += self.controller[part_name].control_dim
            self._action_split_indexes[part_name] = (previous_idx, last_idx)
            previous_idx = last_idx


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
        
        for part_name in [self.base, self.head, self.torso]:
            if part_name == self.head:
                continue
            self.controller[part_name].reset()

        self.controller[self.base].reset()
        self.controller[self.torso].reset()

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

        # # set up references for mobile base
        # self._ref_base_actuator_indexes = [
        #     self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.base.actuators
        # ]

        # # set up references for torso
        # self._ref_torso_actuator_indexes = [
        #     self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.torso_actuators
        # ]


        self._ref_actuators_indexes_dict[self.base] = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.base_actuators
        ]

        self._ref_actuators_indexes_dict[self.torso] = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.torso_actuators
        ]

        self._ref_actuators_indexes_dict[self.head] = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.head_actuators
        ]

        self._ref_joints_indexes_dict[self.base] = [
            self.sim.model.joint_name2id(joint) for joint in self.robot_model.base_joints
        ]

        self._ref_joints_indexes_dict[self.torso] = [
            self.sim.model.joint_name2id(joint) for joint in self.robot_model.torso_joints
        ]
        
        self._ref_joints_indexes_dict[self.head] = [
            self.sim.model.joint_name2id(joint) for joint in self.robot_model.head_joints
        ]

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

        mode = "base" if action[-1] > 0 else "arm"

        self.base_pos, self.base_ori = self.controller[self.base].get_base_pose()
        for arm in self.arms:
            # (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)
            # self.controller[arm].update_initial_joints(self.sim.data.qpos[self._ref_joint_pos_indexes[start:end]])
            # TODO: This line should be removed for arms, and change it to internal computation of base. 
            self.controller[arm].update_base_pose()

        if mode == "base":
            mobile_base_dims = self.controller[self.base].control_dim
            torso_dims = self.controller[self.torso].control_dim

            (base_start, base_end) = self._action_split_indexes[self.base]
            (torso_start, torso_end) = self._action_split_indexes[self.torso]
            base_action = action[base_start:base_end]
            torso_action = action[torso_start:torso_end]
            # base_action = np.copy(action[-mobile_base_dims - torso_dims - 1 : -torso_dims - 1])
            # torso_action = np.copy(action[-torso_dims - 1 : -1])
            if policy_step:
                self.controller[self.base].set_goal(base_action)
                self.controller[self.torso].set_goal(torso_action)

            mobile_base_torques = self.controller[self.base].run_controller()
            self.sim.data.ctrl[self._ref_actuators_indexes_dict[self.base]] = mobile_base_torques

            # Apply torques for height control (if applicable)
            if len(self._ref_actuators_indexes_dict[self.torso]) > 0:
                self.sim.data.ctrl[self._ref_actuators_indexes_dict[self.torso]] = self.controller[self.torso].run_controller()

        self.torques = np.array([])
        # Now execute actions for each arm
        for arm in self.arms:
            # Make sure to split action space correctly
            # (start, end) = (None, self._action_split_idx) if arm == "right" else (self._action_split_idx, None)
            (start, end) = self._action_split_indexes[arm]
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
        # Clip the torques'
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

        @sensor(modality=modality)
        def base_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("mobile_base0_center")])

        sensors = [eef_pos, eef_quat, base_pos]
        names = [f"{pf}{arm}_eef_pos", f"{pf}{arm}_eef_quat", f"{pf}base_pos"]

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

        mobile_base_dims = self.controller[self.base].control_dim if self.base in self.controller else 0
        torso_dims = self.controller[self.torso].control_dim if self.torso in self.controller else 0
        head_dims = 0 # self.controller[self.head].control_dim if self.head in self.controller else 0
        low_b, high_b = ([-1] * mobile_base_dims, [1] * mobile_base_dims)  # base control dims
        low_t, high_t = ([-1] * torso_dims, [1] * torso_dims)  # base control dims
        low_h, high_h = ([-1] * head_dims, [1] * head_dims)  # base control dims

        # TODO: This mode thing should be removed and put into the controller manager
        low_m, high_m = ([-1] * 1, [1] * 1)  # mode control dims

        low = np.concatenate([low, low_b, low_t, low_h, low_m])
        high = np.concatenate([high, high_b, high_t, high_h, high_m])
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
        return int(len(self.robot_arm_joints) / len(self.arms))
