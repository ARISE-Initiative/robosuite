import copy
import os
import time
from collections import OrderedDict
from typing import Dict, List

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers import composite_controller_factory
from robosuite.models.bases.leg_base_model import LegBaseModel
from robosuite.robots.mobile_robot import MobileRobot


class LeggedRobot(MobileRobot):
    """
    Initializes a robot with a wheeled base.
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

    def _load_leg_controllers(self):
        if len(self._ref_actuators_indexes_dict[self.legs]) == 0:
            return None

        if self.part_controller_config.get(self.legs) is None:
            controller_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "controllers/config/{}.json".format(self.robot_model.default_controller_config[self.legs]),
            )
            self.part_controller_config[self.legs] = load_controller_config(custom_fpath=controller_path)

            # Assert that the controller config is a dict file:
            #             NOTE: "type" must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
            #                                           OSC_POSITION, OSC_POSE, IK_POSE}
            assert (
                type(self.part_controller_config[self.legs]) == dict
            ), "Inputted controller config must be a dict! Instead, got type: {}".format(
                type(self.part_controller_config[self.legs])
            )
        self.part_controller_config[self.legs] = {}
        self.part_controller_config[self.legs]["ramp_ratio"] = 1.0
        self.part_controller_config[self.legs]["robot_name"] = self.name

        self.part_controller_config[self.legs]["sim"] = self.sim
        self.part_controller_config[self.legs]["part_name"] = self.legs
        self.part_controller_config[self.legs]["naming_prefix"] = self.robot_model.base.naming_prefix
        self.part_controller_config[self.legs]["ndim"] = self.num_leg_joints
        self.part_controller_config[self.legs]["policy_freq"] = self.control_freq

        ref_legs_joint_indexes = [self.sim.model.joint_name2id(x) for x in self.robot_model.legs_joints]
        ref_legs_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_model.legs_joints]
        ref_legs_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_model.legs_joints]
        self.part_controller_config[self.legs]["joint_indexes"] = {
            "joints": ref_legs_joint_indexes,
            "qpos": ref_legs_joint_pos_indexes,
            "qvel": ref_legs_joint_vel_indexes,
        }

        low = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.legs], 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_actuators_indexes_dict[self.legs], 1]

        self.part_controller_config[self.legs]["actuator_range"] = (low, high)

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
            lite_physics=self.lite_physics,
        )

        self._load_arm_controllers()

        # default base, torso, and head controllers are inherited from MobileRobot
        self._load_base_controller()

        if self.is_legs_actuated:
            self._load_leg_controllers()

        self._load_head_controller()
        self._load_torso_controller()
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
                controller = self.part_controllers.get(arm, None)
                if controller is None:
                    # TODO: enable buffer update for whole body controllers not using individual arm controllers
                    continue
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
    def is_legs_actuated(self):
        return len(self.robot_model.legs_actuators) > 0

    @property
    def num_leg_joints(self):
        return len(self.robot_model.legs_joints)
