import numpy as np

from collections import OrderedDict

import robosuite.utils.transform_utils as T

from robosuite.models.grippers import gripper_factory
from robosuite.controllers import controller_factory, load_controller_config

from robosuite.agents.robot import Robot

import os


class SingleArm(Robot):
    """Initializes a single-armed robot, as defined by a single corresponding XML"""

    def __init__(
        self,
        robot_type: str,
        idn=0,
        controller_config=None,
        initialization_noise=None,
        gripper_type="default",
        gripper_visualization=False,
        control_freq=10
    ):
        """
        Args:
            robot_type (str): Specification for specific robot arm to be instantiated within this env (e.g: "Panda")

            idn (int or str): Unique ID of this robot. Should be different from others

            controller_config (dict): If set, contains relevant controller parameters for creating a custom controller.
                Else, uses the default controller for this specific task

            initialization_noise (float): The scale factor of uni-variate Gaussian random noise
                applied to each of a robot's given initial joint positions. Setting this value to "None" or 0.0 results
                in no noise being applied

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory. Default is "default", which is the default gripper associated
                within the 'robot' specification. None removes the gripper, and any other (valid) model overrides the
                default gripper

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.
        """

        self.controller = None
        self.controller_config = controller_config
        self.gripper_type = gripper_type
        self.has_gripper = self.gripper_type is not None
        self.gripper_visualization = gripper_visualization
        self.control_freq = control_freq

        self.gripper = None         # Gripper class
        self.gripper_joints = None              # xml joint names for gripper
        self._ref_gripper_joint_pos_indexes = None  # xml gripper joint position indexes in mjsim
        self._ref_gripper_joint_vel_indexes = None  # xml gripper joint velocity indexes in mjsim
        self._ref_joint_gripper_actuator_indexes = None     # xml gripper (pos) actuator indexes for robot in mjsim
        self.eef_site_id = None                             # xml element id for eef in mjsim
        self.eef_cylinder_id = None                         # xml element id for eef cylinder in mjsim
        self.torques = None                                 # Current torques being applied

        super().__init__(
            robot_type=robot_type,
            idn=idn,
            initialization_noise=initialization_noise,
        )

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        # First, load the default controller if none is specified
        if not self.controller_config:
            # Need to update default for a single agent
            controller_path = os.path.join(os.path.dirname(__file__), '..',
                                           'controllers/config/{}.json'.format(
                                               self.robot_model.default_controller_config))
            self.controller_config = load_controller_config(custom_fpath=controller_path)

        # Assert that the controller config is a dict file:
        #             NOTE: "type" must be one of: {JOINT_IMP, JOINT_TOR, JOINT_VEL, EE_POS, EE_POS_ORI, EE_IK}
        assert type(self.controller_config) == dict, \
            "Inputted controller config must be a dict! Instead, got type: {}".format(type(self.controller_config))

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self.controller_config["robot_name"] = self.name
        self.controller_config["sim"] = self.sim
        self.controller_config["eef_name"] = self.robot_model.eef_name
        self.controller_config["joint_indexes"] = {
            "joints": self.joint_indexes,
            "qpos": self._ref_joint_pos_indexes,
            "qvel": self._ref_joint_vel_indexes
                                              }
        self.controller_config["actuator_range"] = self.torque_limits
        self.controller_config["policy_freq"] = self.control_freq
        self.controller_config["ndim"] = len(self.robot_joints)

        # Instantiate the relevant controller
        self.controller = controller_factory(self.controller_config["type"], self.controller_config)

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        # First, run the superclass method to load the relevant model
        super().load_model()

        # Verify that the loaded model is of the correct type for this robot
        if self.robot_model.arm_type != "single":
            raise TypeError("Error loading robot model: Incompatible arm type specified for this robot. "
                            "Requested model arm type: {}, robot arm type: {}"
                            .format(self.robot_model.arm_type, type(self)))

        # Now, load the gripper if necessary
        if self.has_gripper:
            if self.gripper_type == 'default':
                # Load the default gripper from the robot file
                self.gripper = gripper_factory(self.robot_model.gripper, idn=self.idn)
            else:
                # Load user-specified gripper
                self.gripper = gripper_factory(self.gripper_type)
            if not self.gripper_visualization:
                self.gripper.hide_visualization()
            self.robot_model.add_gripper(self.gripper)

    def reset(self):
        """
        Sets initial pose of arm and grippers.

        """
        # First, run the superclass method to reset the position and controller
        super().reset()

        # Now, reset the griipper if necessary
        if self.has_gripper:
            self.sim.data.qpos[
                self._ref_gripper_joint_pos_indexes
            ] = self.gripper.init_qpos

        # Update base pos / ori references in controller
        self.controller.update_base_pose(self.base_pos, self.base_ori)

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        # First, run the superclass method to setup references for joint-related values / indexes
        super().setup_references()

        # Now, add references to gripper if necessary
        # indices for grippers in qpos, qvel
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self._ref_gripper_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
            ]
            self._ref_gripper_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
            ]
            self._ref_joint_gripper_actuator_indexes = [
                self.sim.model.actuator_name2id(actuator)
                for actuator in self.gripper.actuators
            ]

            # IDs of sites for gripper visualization
            self.eef_site_id = self.sim.model.site_name2id(self.gripper.visualization_sites["grip_site"])
            self.eef_cylinder_id = self.sim.model.site_name2id(self.gripper.visualization_sites["grip_cylinder"])

    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.robot_model.dof dimensions should be the desired
                normalized joint velocities and if the robot has
                a gripper, the next @self.gripper.dof dimensions should be
                actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken
        """

        # clip actions into valid range
        assert len(action) == self.action_dim, \
            "environment got invalid action dimension -- expected {}, got {}".format(
                self.action_dim, len(action))

        gripper_action = None
        if self.has_gripper:
            gripper_action = action[self.controller.control_dim:]  # all indexes past controller dimension indexes
            action = action[:self.controller.control_dim]

        # Update model in controller
        self.controller.update()

        # Update the controller goal if this is a new policy step
        if policy_step:
            self.controller.set_goal(action)

        # Now run the controller for a step
        torques = self.controller.run_controller()

        # Clip the torques
        low, high = self.torque_limits
        self.torques = np.clip(torques, low, high)

        # Get gripper action, if applicable
        if self.has_gripper:
            gripper_action_actual = self.gripper.format_action(gripper_action)
            # rescale normalized gripper action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange[self._ref_joint_gripper_actuator_indexes]
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_gripper_action = bias + weight * gripper_action_actual
            self.sim.data.ctrl[self._ref_joint_gripper_actuator_indexes] = applied_gripper_action

        # Apply joint torque control
        self.sim.data.ctrl[self._ref_joint_torq_actuator_indexes] = self.torques

    def gripper_visualization(self):
        """
        Do any needed visualization here.
        """
        if self.gripper_visualization:
            # By default, don't do any coloring.
            self.sim.model.site_rgba[self.eef_site_id] = [0., 0., 0., 0.]

    def get_observations(self, di: OrderedDict):
        """
        Returns an OrderedDict containing robot observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
        """
        # Get prefix from robot model to avoid naming clashes for multiple robots
        pf = self.robot_model.naming_prefix

        # proprioceptive features
        di[pf + "joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        )
        di[pf + "joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        )

        robot_states = [
            np.sin(di[pf + "joint_pos"]),
            np.cos(di[pf + "joint_pos"]),
            di[pf + "joint_vel"],
        ]

        if self.has_gripper:
            di[pf + "gripper_qpos"] = np.array(
                [self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes]
            )
            di[pf + "gripper_qvel"] = np.array(
                [self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes]
            )

            di[pf + "eef_pos"] = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di[pf + "eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat(self.robot_model.eef_name), to="xyzw"
            )

            # add in gripper information
            robot_states.extend([di[pf + "gripper_qpos"], di[pf + "eef_pos"], di[pf + "eef_quat"]])

        di[pf + "robot-state"] = np.concatenate(robot_states)
        return di

    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.
        """
        # Action limits based on controller limits
        low, high = ([-1] * self.gripper.dof, [1] * self.gripper.dof) if self.has_gripper else ([], [])
        low = np.concatenate([self.controller.input_min, low])
        high = np.concatenate([self.controller.input_max, high])

        return low, high

    @property
    def torque_limits(self):
        """
        Action lower/upper limits per dimension.
        """
        # Torque limit values pulled from relevant robot.xml file
        low = self.sim.model.actuator_ctrlrange[self._ref_joint_torq_actuator_indexes, 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_joint_torq_actuator_indexes, 1]

        return low, high

    @property
    def action_dim(self):
        """
        Action space dimension for this robot (controller dimension + gripper dof)
        """
        return self.controller.control_dim + self.gripper.dof if self.has_gripper else self.controller.control_dim

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with grippers).
        """
        # Get the dof of the base robot model
        dof = super().dof
        if self.has_gripper:
            dof += self.gripper.dof
        return dof

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name(self.robot_model.eef_name)

    @property
    def _right_hand_quat(self):
        """
        Returns eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame
        as a numpy array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp(self.robot_model.eef_name).reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes]

        Jr = self.sim.data.get_body_jacr(self.robot_model.eef_name).reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[:3]

    @property
    def _right_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[3:]
