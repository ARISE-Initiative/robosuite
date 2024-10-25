import copy
import os
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers import composite_controller_factory
from robosuite.robots.robot import Robot


class FixedBaseRobot(Robot):
    """
    Initializes a robot with a fixed base.
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
        lite_physics=False,
    ):
        super().__init__(
            robot_type=robot_type,
            idn=idn,
            composite_controller_config=composite_controller_config,
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
        # Flag for loading urdf once (only applicable for IK controllers)
        self.composite_controller = composite_controller_factory(
            type=self.composite_controller_config.get("type", "BASIC"),
            sim=self.sim,
            robot_model=self.robot_model,
            grippers={self.get_gripper_name(arm): self.gripper[arm] for arm in self.arms},
        )

        self._load_arm_controllers()

        self._postprocess_part_controller_config()
        self.composite_controller.load_controller_config(
            self.part_controller_config,
            self.composite_controller_config.get("composite_controller_specific_configs", {}),
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
            (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)
            self._ref_joints_indexes_dict[arm] = [
                self.sim.model.joint_name2id(joint) for joint in self.robot_model.arm_joints[start:end]
            ]
            self._ref_actuators_indexes_dict[arm] = [
                self.sim.model.actuator_name2id(joint) for joint in self.robot_model.arm_actuators[start:end]
            ]
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
                self._ref_joints_indexes_dict[self.get_gripper_name(arm)] = [
                    self.sim.model.joint_name2id(joint) for joint in self.gripper_joints[arm]
                ]
                self._ref_actuators_indexes_dict[self.get_gripper_name(arm)] = self._ref_joint_gripper_actuator_indexes[
                    arm
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

        self.composite_controller.update_state()
        if policy_step:
            self.composite_controller.set_goal(action)

        applied_action_dict = self.composite_controller.run_controller(self._enabled_parts)
        for part_name, applied_action in applied_action_dict.items():
            applied_action_low = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[part_name], 0]
            applied_action_high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[part_name], 1]
            applied_action = np.clip(applied_action, applied_action_low, applied_action_high)
            self.sim.data.ctrl[self._ref_actuators_indexes_dict[part_name]] = applied_action

        if policy_step:
            # Update proprioceptive values
            self.recent_qpos.push(self._joint_positions)
            self.recent_actions.push(action)
            self.recent_torques.push(self.torques)

            for arm in self.arms:
                controller = self.part_controllers[arm]
                # Update arm-specific proprioceptive values
                self.recent_ee_forcetorques[arm].push(np.concatenate((self.ee_force[arm], self.ee_torque[arm])))
                self.recent_ee_pose[arm].push(np.concatenate((controller.ref_pos, T.mat2quat(controller.ref_ori_mat))))
                self.recent_ee_vel[arm].push(np.concatenate((controller.ref_pos_vel, controller.ref_ori_vel)))

                # Estimation of eef acceleration (averaged derivative of recent velocities)
                self.recent_ee_vel_buffer[arm].push(np.concatenate((controller.ref_pos_vel, controller.ref_ori_vel)))
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
    def is_mobile(self):
        return False

    @property
    def _action_split_indexes(self):
        """
        Dictionary of split indexes for each part of the robot

        Returns:
            dict: Dictionary of split indexes for each part of the robot
        """
        return self.composite_controller._action_split_indexes
