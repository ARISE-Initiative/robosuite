import copy
import os
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers import composite_controller_factory, load_controller_config
from robosuite.robots.robot import Robot


class MobileBaseRobot(Robot):
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
        base_type="default",
        gripper_type="default",
        control_freq=20,
        lite_physics=False,
    ):
        super().__init__(
            robot_type=robot_type,
            idn=idn,
            controller_config=controller_config,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
            base_type=base_type,
            gripper_type=gripper_type,
            control_freq=control_freq,
            lite_physics=lite_physics,
        )

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        pass

    def _load_base_controller(self):
        """
        Load base controller
        """
        if len(self._ref_actuators_indexes_dict[self.base]) == 0:
            return None
        # TODO: Add a default controller config for base
        self.controller_config[self.base] = {}
        self.controller_config[self.base]["type"] = "JOINT_VELOCITY"
        self.controller_config[self.base]["interpolation"] = None
        self.controller_config[self.base]["ramp_ratio"] = 1.0
        self.controller_config[self.base]["robot_name"] = self.name

        self.controller_config[self.base]["sim"] = self.sim
        self.controller_config[self.base]["part_name"] = self.base
        self.controller_config[self.base]["naming_prefix"] = self.robot_model.base.naming_prefix
        self.controller_config[self.base]["ndim"] = self._joint_split_idx
        self.controller_config[self.base]["policy_freq"] = self.control_freq
        self.controller_config[self.base]["lite_physics"] = self.lite_physics

        ref_base_joint_indexes = [self.sim.model.joint_name2id(x) for x in self.robot_model.base_joints]
        ref_base_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_model.base_joints]
        ref_base_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_model.base_joints]
        self.controller_config[self.base]["joint_indexes"] = {
            "joints": ref_base_joint_indexes,
            "qpos": ref_base_joint_pos_indexes,
            "qvel": ref_base_joint_vel_indexes,
        }

        low = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.base], 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.base], 1]

        self.controller_config[self.base]["actuator_range"] = (low, high)

    def _load_torso_controller(self):
        """
        Load torso controller
        """
        if len(self._ref_actuators_indexes_dict[self.torso]) == 0:
            return None
        # TODO: Add a default controller config for torso
        self.controller_config[self.torso] = {}
        self.controller_config[self.torso]["type"] = "JOINT_POSITION"
        self.controller_config[self.torso]["interpolation"] = None
        self.controller_config[self.torso]["ramp_ratio"] = 1.0
        self.controller_config[self.torso]["robot_name"] = self.name
        self.controller_config[self.torso]["sim"] = self.sim
        self.controller_config[self.torso]["kp"] = 2000
        self.controller_config[self.torso]["part_name"] = self.torso
        self.controller_config[self.torso]["naming_prefix"] = self.robot_model.naming_prefix
        self.controller_config[self.torso]["ndim"] = self._joint_split_idx
        self.controller_config[self.torso]["policy_freq"] = self.control_freq
        self.controller_config[self.torso]["lite_physics"] = self.lite_physics

        ref_torso_joint_indexes = [self.sim.model.joint_name2id(x) for x in self.robot_model.torso_joints]
        ref_torso_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_model.torso_joints]
        ref_torso_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_model.torso_joints]
        self.controller_config[self.torso]["joint_indexes"] = {
            "joints": ref_torso_joint_indexes,
            "qpos": ref_torso_joint_pos_indexes,
            "qvel": ref_torso_joint_vel_indexes,
        }

        low = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.torso], 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.torso], 1]

        self.controller_config[self.torso]["actuator_range"] = (low, high)

    def _load_head_controller(self):
        """
        Load head controller
        """
        if len(self._ref_actuators_indexes_dict[self.head]) == 0:
            return None
        # TODO: Add a default controller config for head
        self.controller_config[self.head] = {}
        self.controller_config[self.head]["type"] = "JOINT_POSITION"
        self.controller_config[self.head]["interpolation"] = None
        self.controller_config[self.head]["ramp_ratio"] = 1.0
        self.controller_config[self.head]["robot_name"] = self.name
        self.controller_config[self.head]["sim"] = self.sim
        self.controller_config[self.head]["kp"] = 10
        self.controller_config[self.head]["output_max"] = 1.0
        self.controller_config[self.head]["output_min"] = -1.0

        self.controller_config[self.head]["part_name"] = self.head
        self.controller_config[self.head]["naming_prefix"] = self.robot_model.naming_prefix
        self.controller_config[self.head]["ndim"] = self._joint_split_idx
        self.controller_config[self.head]["policy_freq"] = self.control_freq

        ref_head_joint_indexes = [self.sim.model.joint_name2id(x) for x in self.robot_model.head_joints]
        ref_head_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_model.head_joints]
        ref_head_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_model.head_joints]
        self.controller_config[self.head]["joint_indexes"] = {
            "joints": ref_head_joint_indexes,
            "qpos": ref_head_joint_pos_indexes,
            "qvel": ref_head_joint_vel_indexes,
        }

        low = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.head], 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.head], 1]

        self.controller_config[self.head]["actuator_range"] = (low, high)

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
                (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)
                self._ref_joints_indexes_dict[arm] = [
                    self.sim.model.joint_name2id(joint) for joint in self.robot_model.arm_joints[start:end]
                ]
                self._ref_actuators_indexes_dict[arm] = [
                    self.sim.model.actuator_name2id(joint) for joint in self.robot_model.arm_actuators[start:end]
                ]
                self._ref_joints_indexes_dict[self.get_gripper_name(arm)] = [
                    self.sim.model.joint_name2id(joint) for joint in self.gripper_joints[arm]
                ]
                self._ref_actuators_indexes_dict[self.get_gripper_name(arm)] = self._ref_joint_gripper_actuator_indexes[
                    arm
                ]

            # IDs of sites for eef visualization
            self.eef_site_id[arm] = self.sim.model.site_name2id(self.gripper[arm].important_sites["grip_site"])
            self.eef_cylinder_id[arm] = self.sim.model.site_name2id(self.gripper[arm].important_sites["grip_cylinder"])

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
        raise NotImplementedError

    def setup_observables(self):
        """
        Sets up observables to be used for this robot

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        # Get general robot observables first
        observables = super().setup_observables()

        return observables

    def enable_parts(self, right=True, left=True, torso=True, head=True, base=True, legs=True):

        self._enabled_parts = {
            "right": right,
            "right_gripper": right,
            "left": left,
            "left_gripper": left,
            self.torso: torso,
            self.head: head,
            self.base: base,
            self.legs: legs,
        }

    @property
    def is_mobile(self):
        return True

    @property
    def base(self):
        return "base"

    @property
    def torso(self):
        return "torso"

    @property
    def head(self):
        return "head"

    @property
    def legs(self):
        return "legs"

    @property
    def _action_split_indexes(self):
        """
        Dictionary of split indexes for each part of the robot

        Returns:
            dict: Dictionary of split indexes for each part of the robot
        """
        return self.composite_controller._action_split_indexes

    @property
    def controller(self):
        """
        Controller dictionary for the robot

        Returns:
            dict: Controller dictionary for the robot
        """
        return self.composite_controller.controllers
