import copy
import os
from collections import OrderedDict

import time
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers import controller_manager_factory, load_controller_config
from robosuite.robots.mobile_base_robot import MobileBaseRobot
from robosuite.utils.observables import Observable, sensor
from robosuite.models.bases.leg_base_model import LegBaseModel


class LeggedRobot(MobileBaseRobot):
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

    def _load_leg_controllers(self):
        self.controller_config[self.legs] = {}
        self.controller_config[self.legs]["type"] = "JOINT_POSITION"
        self.controller_config[self.legs]["interpolation"] = "linear"
        self.controller_config[self.legs]["ramp_ratio"] = 1.0
        self.controller_config[self.legs]["robot_name"] = self.name

        self.controller_config[self.legs]["sim"] = self.sim
        self.controller_config[self.legs]["part_name"] = self.legs
        self.controller_config[self.legs]["naming_prefix"] = self.robot_model.base.naming_prefix
        self.controller_config[self.legs]["ndim"] = self.num_leg_joints
        self.controller_config[self.legs]["policy_freq"] = self.control_freq

        ref_legs_joint_indexes = [self.sim.model.joint_name2id(x) for x in self.robot_model.legs_joints]
        ref_legs_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_model.legs_joints]
        ref_legs_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_model.legs_joints]
        self.controller_config[self.legs]["joint_indexes"] = {
            "joints": ref_legs_joint_indexes,
            "qpos": ref_legs_joint_pos_indexes,
            "qvel": ref_legs_joint_vel_indexes,
        }

        low =  self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.legs], 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.legs], 1]

        self.controller_config[self.legs]["actuator_range"] = (
            low,
            high
        )
        # self.controller[self.legs] = controller_factory(self.controller_config[self.legs]["type"], self.controller_config[self.legs])

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        # Flag for loading urdf once (only applicable for IK controllers)

        self.controller_manager = controller_manager_factory(
            "BASE",
            self.sim,
            self.robot_model,
            grippers={self.get_gripper_name(arm): self.gripper[arm] for arm in self.arms},
        )

        self._load_arm_controllers()

        # # self.controller[self.base] = MobileBaseController(self.sim, self.robot_model.base.naming_prefix)


        # default base, torso, and head controllers are inherited from MobileBaseRobot
        self._load_base_controller()

        if self.is_legs_actuated:
            self._load_leg_controllers()

        self._load_head_controller()
        self._load_torso_controller()

        self.controller_manager.load_controller_config(self.controller_config)
        self.enable_parts()
        # # self.controller[self.head] = controller_factory("OSC_POSE", self.controller_config["right"])

        # # Set up split indices for arm actions
        # self._action_split_indexes.clear()
        # previous_idx = 0
        # last_idx = 0
        # for arm in self.arms:
        #     last_idx += self.controller[arm].control_dim
        #     last_idx += self.gripper[arm].dof if self.has_gripper[arm] else 0
        #     self._action_split_indexes[arm] = (previous_idx, last_idx)
        #     previous_idx = last_idx

        # previous_idx = self._action_split_indexes[self.arms[-1]][1]
        # last_idx = previous_idx
        # for part_name in [self.base, self.legs, self.head, self.torso]:
        #     if part_name not in self.controller:
        #         self._action_split_indexes[part_name] = (last_idx, last_idx)
        #         continue
        
        #     last_idx += self.controller[part_name].control_dim
        #     self._action_split_indexes[part_name] = (previous_idx, last_idx)
        #     previous_idx = last_idx

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
        
        # for part_name in [self.base, self.head, self.torso]:
        #     if part_name not in self.controller:
        #         continue
        #     self.controller[part_name].reset_goal()

        self.controller_manager.update_state()
        self.controller_manager.reset()
        # Set initial q pos of the legged base
        if isinstance(self.robot_model.base, LegBaseModel):
            # Set the initial joint positions of the legged base
            self.sim.data.qpos[self._ref_legs_joint_pos_indexes] = self.robot_model.base.init_qpos

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.

        Note that this should get called during every reset from the environment
        """
        # First, run the superclass method to setup references for joint-related values / indexes
        super().setup_references()

        # Now, add references to gripper if necessary
        # indices for grippers in qpos, qvel

        self._ref_actuators_indexes_dict[self.legs] = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.legs_actuators
        ]

        self._ref_joints_indexes_dict[self.legs] = [
            self.sim.model.joint_name2id(joint) for joint in self.robot_model.legs_joints
        ]
        self._ref_legs_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_model.legs_joints]
        self._ref_legs_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_model.legs_joints] 

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

        self.controller_manager.update_state()
        if policy_step:
            self.controller_manager.set_goal(action)

        applied_action_dict = self.controller_manager.compute_applied_action(self._enabled_parts)
        for part_name, applied_action in applied_action_dict.items():
            applied_action_low = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[part_name], 0]
            applied_action_high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[part_name], 1]
            applied_action = np.clip(applied_action, applied_action_low, applied_action_high)

            self.sim.data.ctrl[self._ref_actuators_indexes_dict[part_name]] = applied_action

        # mode = "base" if action[-1] > 0 else "arm"

        # if self.base in self.controller:
        #     self.base_pos, self.base_ori = self.controller[self.base].get_base_pose()
        # else:
        #     base_site_name = f"{self.robot_model.base.naming_prefix}center"
        #     self.base_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(base_site_name)])
        #     self.base_ori = np.array(
        #         self.sim.data.site_xmat[self.sim.model.site_name2id(base_site_name)].reshape([3, 3])
        #     )            
        # for arm in self.arms:
        #     # (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)
        #     # self.controller[arm].update_initial_joints(self.sim.data.qpos[self._ref_joint_pos_indexes[start:end]])
        #     # TODO: This line should be removed for arms, and change it to internal computation of base. 
        #     self.controller[arm].update_base_pose()

        # if self.enabled(self.base) and len(self._ref_actuators_indexes_dict[self.base]) > 0:
        #     mobile_base_dims = self.controller[self.base].control_dim
        #     (base_start, base_end) = self._action_split_indexes[self.base]
        #     base_action = action[base_start:base_end]


        #     if policy_step:
        #         self.controller[self.base].set_goal(base_action)

        #     applied_base_action = self.controller[self.base].run_controller()
        #     self.sim.data.ctrl[self._ref_actuators_indexes_dict[self.base]] = applied_base_action

        #     # Apply torques for height control (if applicable)

        # if self.enabled(self.legs) and len(self._ref_actuators_indexes_dict[self.legs]) > 0:
        #     legs_dims = self.controller[self.legs].control_dim
        #     (legs_start, legs_end) = self._action_split_indexes[self.legs]
        #     legs_action = action[legs_start:legs_end]
        #     if policy_step:
        #         self.controller[self.legs].set_goal(legs_action)
        #     self.sim.data.ctrl[self._ref_actuators_indexes_dict[self.legs]] = self.controller[self.legs].run_controller()

        # if self.enabled(self.head) and len(self._ref_actuators_indexes_dict[self.head]) > 0:
        #     head_dims = self.controller[self.head].control_dim
        #     (head_start, head_end) = self._action_split_indexes[self.head]
        #     head_action = action[head_start:head_end]
        #     if policy_step:
        #         self.controller[self.head].set_goal(head_action)
        #     self.sim.data.ctrl[self._ref_actuators_indexes_dict[self.head]] = self.controller[self.head].run_controller()

        # if self.enabled(self.torso) and len(self._ref_actuators_indexes_dict[self.torso]) > 0:
        #     torso_dims = self.controller[self.torso].control_dim
        #     (torso_start, torso_end) = self._action_split_indexes[self.torso]
        #     torso_action = action[torso_start:torso_end]
        #     if policy_step:
        #         self.controller[self.torso].set_goal(torso_action)
        #     self.sim.data.ctrl[self._ref_actuators_indexes_dict[self.torso]] = self.controller[self.torso].run_controller()

        # self.torques = np.array([])
        # # Now execute actions for each arm
        # for arm in self.arms:
        #     # Make sure to split action space correctly
        #     (start, end) = self._action_split_indexes[arm]
        #     sub_action = action[start:end]

        #     gripper_action = None
        #     if self.has_gripper[arm]:
        #         # get all indexes past controller dimension indexes
        #         gripper_action = sub_action[self.controller[arm].control_dim :]
        #         sub_action = sub_action[: self.controller[arm].control_dim]

        #     # Update the controller goal if this is a new policy step
        #     if policy_step:
        #         self.controller[arm].set_goal(sub_action)

        #     # Now run the controller for a step and add it to the torques
        #     applied_torque = self.controller[arm].run_controller()
        #     self.sim.data.ctrl[self._ref_actuators_indexes_dict[arm]] = applied_torque
        #     self.torques = np.concatenate((self.torques, applied_torque))

        #     # Get gripper action, if applicable
        #     if self.has_gripper[arm]:
        #         gripper_name = self.get_gripper_name(arm)
        #         # if policy_step:
        #         formatted_gripper_action = self.gripper[arm].format_action(gripper_action)
        #         self.controller[gripper_name].set_goal(formatted_gripper_action)
        #         applied_gripper_action = self.controller[gripper_name].run_controller()
        #         self.sim.data.ctrl[self._ref_actuators_indexes_dict[self.get_gripper_name(arm)]] = applied_gripper_action

        # # Clip the torques'
        # low, high = self.torque_limits
        # self.torques = np.clip(self.torques, low, high)
        # # Apply joint torque control
        # self.sim.data.ctrl[self._ref_arm_joint_actuator_indexes] = self.torques

        # If this is a policy step, also update buffers holding recent values of interest
        if policy_step:
            # Update proprioceptive values
            self.recent_qpos.push(self._joint_positions)
            self.recent_actions.push(action)
            self.recent_torques.push(self.torques)

            for arm in self.arms:
                controller = self.controller[arm]
                # Update arm-specific proprioceptive values
                self.recent_ee_forcetorques[arm].push(np.concatenate((self.ee_force[arm], self.ee_torque[arm])))
                self.recent_ee_pose[arm].push(
                    np.concatenate((controller.ee_pos, T.mat2quat(controller.ee_ori_mat)))
                )
                self.recent_ee_vel[arm].push(
                    np.concatenate((controller.ee_pos_vel, controller.ee_ori_vel))
                )

                # Estimation of eef acceleration (averaged derivative of recent velocities)
                self.recent_ee_vel_buffer[arm].push(
                    np.concatenate((controller.ee_pos_vel, controller.ee_ori_vel))
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
        return self.controller_manager.action_limits
        # Action limits based on controller limits
        # low, high = [], []
        # for arm in self.arms:
        #     low_g, high_g = (
        #         ([-1] * self.gripper[arm].dof, [1] * self.gripper[arm].dof) if self.has_gripper[arm] else ([], [])
        #     )
        #     low_c, high_c = self.controller[arm].control_limits
        #     low, high = np.concatenate([low, low_c, low_g]), np.concatenate([high, high_c, high_g])

        # mobile_base_dims = self.controller[self.base].control_dim if self.base in self.controller else 0
        # legs_dims = self.controller[self.legs].control_dim if self.legs in self.controller else 0
        # torso_dims = self.controller[self.torso].control_dim if self.torso in self.controller else 0
        # head_dims = self.controller[self.head].control_dim if self.head in self.controller else 0
        # low_b, high_b = ([-1] * mobile_base_dims, [1] * mobile_base_dims)  # base control dims
        # low_l, high_l = ([-1] * legs_dims, [1] * legs_dims)  # base control dims
        # low_t, high_t = ([-1] * torso_dims, [1] * torso_dims)  # base control dims
        # low_h, high_h = ([-1] * head_dims, [1] * head_dims)  # base control dims

        # low = np.concatenate([low, low_b, low_l, low_t, low_h])
        # high = np.concatenate([high, high_b, high_l, high_t, high_h])
        # return low, high

    @property
    def is_legs_actuated(self):
        return len(self.robot_model.legs_actuators) > 0

    @property
    def num_leg_joints(self):
        return len(self.robot_model.legs_joints)