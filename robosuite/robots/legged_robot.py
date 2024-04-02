import os
import numpy as np

from robosuite.robots.mobile_robot import MobileRobot
import robosuite.utils.transform_utils as T
from robosuite.controllers import controller_factory, load_controller_config
from robosuite.utils.observables import Observable, sensor



class LeggedRobot(MobileRobot):
    """
    Initializes a robot with a legged base.
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

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides gripper joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        Args:
            deterministic (bool): If true, will not randomize initializations within the sim
        """
        # First, run the superclass method to reset the position and controller
        super().reset(deterministic)

        self.controller_manager.reset()

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

        self.controller[self.base] = base_controller_factory(self.controller_config[self.base]["type"], self.controller_config[self.base])
        self.controller[self.torso] = torso_controller_factory(self.controller_config[self.torso]["type"], self.controller_config[self.torso])
        # gripper controller
        for gripper in self.gripper:
            self.gripper_controller[gripper] = gripper_controller_factory(
                self.gripper_controller_config[gripper]["type"], self.gripper_controller_config[gripper]
            )

        # head controller
        self.controller[self.head] = head_controller_factory(
            self.controller_config[self.head]["type"], self.controller_config[self.head])

    def reset(self, deterministic=False):
        raise NotImplementedError

    def setup_references(self):
        super().setup_references()
        raise NotImplementedError

    def control(self, action, policy_step=False):
        assert(len(action) == self.dof),  "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )

        if policy_step:
            self.controller_manager.set_goal(action)

        ctrl_dict = self.controller_manager.compute_ctrl()

        for (entity_name, ctrl_cmds) in ctrl_dict.items():
            self.sim.data.ctrl[self._ref_actuator_indexes[entity_name]] = ctrl_cmds[entity_name]

    @property
    def action_limits(self):
        low, high = [], []
        for arm in self.arms:
            low_g, high_g = (
                ([-1] * self.gripper[arm].dof, [1] * self.gripper[arm].dof) if self.has_gripper[arm] else ([], [])
            )
            low_c, high_c = self.controller[arm].control_limits
            low, high = np.concatenate([low, low_c, low_g]), np.concatenate([high, high_c, high_g])

        low_b, high_b = self.controller[self.base].control_limits
        low_t, hight_t = self.controller[self.torso].control_limits
        low_h, high_h = self.controller[self.head].control_limits
        low = np.concatenate([low, low_b, low_t, low_h])
        high = np.concatenate([high, high_b, hight_t, high_h])
        return low, high

    @property
    def _action_split_idx(self):
        raise NotImplementedError

    @property
    def _joint_split_idx(self):
        raise NotImplementedError
