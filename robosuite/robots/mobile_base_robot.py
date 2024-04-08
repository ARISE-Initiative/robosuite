import copy
import os
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.robots.robot import Robot

from robosuite.controllers import controller_factory, load_controller_config



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
        pass


    def _load_base_controller(self):
        """
        Load base controller
        """
        if len(self._ref_actuators_indexes_dict[self.base]) == 0:
            return None
        # if not self.controller_config[self.torso]:
        #         # Need to update default for a single agent
        #         controller_path = os.path.join(
        #             os.path.dirname(__file__),
        #             "..",
        #             "controllers/config/torso/{}.json".format(self.robot_model.default_controller_config[self.torso]),
        #         )
        #         self.controller_config[self.torso] = load_controller_config(custom_fpath=controller_path)
        # TODO: Add a default controller config for torso
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

        ref_base_joint_indexes = [self.sim.model.joint_name2id(x) for x in self.robot_model.base_joints]
        ref_base_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_model.base_joints]
        ref_base_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_model.base_joints]
        self.controller_config[self.base]["joint_indexes"] = {
            "joints": ref_base_joint_indexes,
            "qpos": ref_base_joint_pos_indexes,
            "qvel": ref_base_joint_vel_indexes,
        }

        low =  self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.base], 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.base], 1]

        self.controller_config[self.base]["actuator_range"] = (
            low,
            high
        )
        self.controller[self.base] = controller_factory(self.controller_config[self.base]["type"], self.controller_config[self.base])

    def _load_torso_controller(self):
        """
        Load torso controller
        """
        if len(self._ref_actuators_indexes_dict[self.torso]) == 0:
            return None
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
        self.controller_config[self.torso]["type"] = "JOINT_VELOCITY"
        self.controller_config[self.torso]["interpolation"] = None
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

        low =  self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.torso], 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.torso], 1]

        self.controller_config[self.torso]["actuator_range"] = (
            low,
            high
        )

        self.controller[self.torso] = controller_factory(
            self.controller_config[self.torso]["type"],
            self.controller_config[self.torso]
        )

    def _load_head_controller(self):
        """
        Load head controller
        """
        if len(self._ref_actuators_indexes_dict[self.head]) == 0:
            return None
        # if not self.controller_config[self.head]:
        #         # Need to update default for a single agent
        #         controller_path = os.path.join(
        #             os.path.dirname(__file__),
        #             "..",
        #             "controllers/config/head/{}.json".format(self.robot_model.default_controller_config[self.head]),
        #         )
        #         self.controller_config[self.head] = load_controller_config(custom_fpath=controller_path)
        # TODO: Add a default controller config for head
        self.controller_config[self.head] = {}
        self.controller_config[self.head]["type"] = "JOINT_VELOCITY" # "JOINT_POSITION"
        self.controller_config[self.head]["interpolation"] = None
        self.controller_config[self.head]["ramp_ratio"] = 1.0
        self.controller_config[self.head]["robot_name"] = self.name
        self.controller_config[self.head]["sim"] = self.sim
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

        low =  self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.head], 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.head], 1]

        self.controller_config[self.head]["actuator_range"] = (
            low,
            high
        )

        self.controller[self.head] = controller_factory(
            self.controller_config[self.head]["type"],
            self.controller_config[self.head]
        )        

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
    

    def enable_parts(self,
                    right_arm=True,
                    left_arm=False,
                    torso=False,
                    head=False,
                    base=True,
                    legs=False
                     ):
        # TBC
        self._enabled_parts = {
            "right": right_arm, 
            "left": left_arm,
            self.torso: torso,
            self.head: head,
            self.base: base,
            self.legs: legs
        }