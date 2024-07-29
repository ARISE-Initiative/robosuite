import copy
import os
import time
from collections import OrderedDict
from typing import Dict, List

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers import composite_controller_factory, load_controller_config
from robosuite.models.bases.leg_base_model import LegBaseModel
from robosuite.robots.mobile_base_robot import MobileBaseRobot
from robosuite.utils.observables import Observable, sensor


class LeggedRobot(MobileBaseRobot):
    """
    Initializes a robot with a wheeled base.
    """

    def __init__(
        self,
        robot_type: str,
        idn=0,
        controller_config=None,
        composite_controller_config=None,
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
            composite_controller_config=composite_controller_config,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
            base_type=base_type,
            gripper_type=gripper_type,
            control_freq=control_freq,
            lite_physics=lite_physics,
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

        low = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.legs], 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.legs], 1]

        self.controller_config[self.legs]["actuator_range"] = (low, high)

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        # Flag for loading urdf once (only applicable for IK controllers)
        self.composite_controller = composite_controller_factory(
            type=self.composite_controller_config.get("type", "BASE"),
            sim=self.sim,
            robot_model=self.robot_model,
            grippers={self.get_gripper_name(arm): self.gripper[arm] for arm in self.arms},
            lite_physics=self.lite_physics,
        )

        self._load_arm_controllers()

        # default base, torso, and head controllers are inherited from MobileBaseRobot
        self._load_base_controller()

        if self.is_legs_actuated:
            self._load_leg_controllers()

        self._load_head_controller()
        self._load_torso_controller()

        self.composite_controller.load_controller_config(
            self.composite_controller_config['controller_configs']
            if self.composite_controller_config.get('controller_configs', None) is not None else self.controller_config
        )
        self.enable_parts()

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

        self.composite_controller.update_state()
        self.composite_controller.reset()
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
        def get_indices_for_keys(index_dict: Dict[str, List[int]], keys: List[str]) -> List[int]:
            indices = []
            for key in keys:
                if key in index_dict:
                    indices.extend(index_dict[key])
            return indices
        
        for body_part, body_controller_config in self.composite_controller_config["controller_configs"].items():
            new_joint_indexes = get_indices_for_keys(self._ref_joints_indexes_dict, body_controller_config["individual_part_names"])
            new_actuator_indexes = get_indices_for_keys(self._ref_actuators_indexes_dict, body_controller_config["individual_part_names"])
            self._ref_joints_indexes_dict[body_part] = new_joint_indexes
            self._ref_actuators_indexes_dict[body_part] = new_actuator_indexes

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

        self.composite_controller.update_state()
        if policy_step:
            self.composite_controller.set_goal(action)

        applied_action_dict = self.composite_controller.run_controller(self._enabled_parts)
        for part_name, applied_action in applied_action_dict.items():
            applied_action_low = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[part_name], 0]
            applied_action_high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[part_name], 1]
            applied_action = np.clip(applied_action, applied_action_low, applied_action_high)

            self.sim.data.ctrl[self._ref_actuators_indexes_dict[part_name]] = applied_action

        # If this is a policy step, also update buffers holding recent values of interest
        if policy_step:
            # Update proprioceptive values
            self.recent_qpos.push(self._joint_positions)
            self.recent_actions.push(action)
            self.recent_torques.push(self.torques)

            for arm in self.arms:
                controller = self.controller.get(arm, None)
                if controller is None:
                    # TODO: enable buffer update for whole body controllers not using individual arm controllers
                    continue
                # Update arm-specific proprioceptive values
                self.recent_ee_forcetorques[arm].push(np.concatenate((self.ee_force[arm], self.ee_torque[arm])))
                self.recent_ee_pose[arm].push(np.concatenate((controller.ee_pos, T.mat2quat(controller.ee_ori_mat))))
                self.recent_ee_vel[arm].push(np.concatenate((controller.ee_pos_vel, controller.ee_ori_vel)))

                # Estimation of eef acceleration (averaged derivative of recent velocities)
                self.recent_ee_vel_buffer[arm].push(np.concatenate((controller.ee_pos_vel, controller.ee_ori_vel)))
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

        # only consider prefix if there is more than one arm
        pf = f"{arm}_" if len(self.arms) > 1 else ""

        sensors = [eef_pos, eef_quat]  # , base_pos]
        names = [f"{pf}eef_pos", f"{pf}eef_quat"]  # , f"base_pos"]

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

    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        return self.composite_controller.action_limits

    @property
    def is_legs_actuated(self):
        return len(self.robot_model.legs_actuators) > 0

    @property
    def num_leg_joints(self):
        return len(self.robot_model.legs_joints)
