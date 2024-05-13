import copy
import os
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers import composite_controller_factory, load_controller_config
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

        self._load_base_controller()
        self._load_torso_controller()

        self.composite_controller.load_controller_config(self.controller_config)
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

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.

        Note that this should get called during every reset from the environment
        """
        # All the references are by default set up in the superclass method
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
        # clip actions into valid range
        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )

        # self.composite_controller.update_state() # remove this for now, messes up base velocity calculation
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
                controller = self.controller[arm]
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
            return np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("base0_center")])

        @sensor(modality=modality)
        def base_quat(obs_cache):
            return T.mat2quat(self.sim.data.get_site_xmat("base0_center"))

        @sensor(modality=modality)
        def base_to_eef_pos(obs_cache):
            eef_pos = np.array(self.sim.data.site_xpos[self.eef_site_id[arm]])
            base_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("base0_center")])

            eef_quat = T.convert_quat(self.sim.data.get_body_xquat(self.robot_model.eef_name[arm]), to="xyzw")
            eef_mat = T.quat2mat(eef_quat)
            base_mat = self.sim.data.get_site_xmat("base0_center")

            T_WA = np.vstack((np.hstack((base_mat, base_pos[:, None])), [0, 0, 0, 1]))
            T_WB = np.vstack((np.hstack((eef_mat, eef_pos[:, None])), [0, 0, 0, 1]))
            T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)
            base_to_eef_pos = T_AB[:3, 3]
            return base_to_eef_pos

        @sensor(modality=modality)
        def base_to_eef_quat(obs_cache):
            eef_pos = np.array(self.sim.data.site_xpos[self.eef_site_id[arm]])
            base_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("base0_center")])

            eef_quat = T.convert_quat(self.sim.data.get_body_xquat(self.robot_model.eef_name[arm]), to="xyzw")
            eef_mat = T.quat2mat(eef_quat)
            base_mat = self.sim.data.get_site_xmat("base0_center")

            T_WA = np.vstack((np.hstack((base_mat, base_pos[:, None])), [0, 0, 0, 1]))
            T_WB = np.vstack((np.hstack((eef_mat, eef_pos[:, None])), [0, 0, 0, 1]))
            T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)
            base_to_eef_mat = T_AB[:3, :3]
            return T.mat2quat(base_to_eef_mat)

        # only consider prefix if there is more than one arm
        pf = f"{arm}_" if len(self.arms) > 1 else ""

        sensors = [eef_pos, eef_quat, base_pos, base_quat, base_to_eef_pos, base_to_eef_quat]
        names = [
            f"{pf}eef_pos",
            f"{pf}eef_quat",
            f"base_pos",
            f"base_quat",
            f"base_to_{pf}eef_pos",
            f"base_to_{pf}eef_quat",
        ]

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
