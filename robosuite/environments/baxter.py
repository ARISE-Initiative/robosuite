from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments import MujocoEnv

from robosuite.models.grippers import gripper_factory
from robosuite.controllers import controller_factory
from robosuite.models.robots import Baxter


class BaxterEnv(MujocoEnv):
    """Initializes a Baxter robot environment."""

    def __init__(
        self,
        controller_config,
        gripper_right=None,
        gripper_left=None,
        gripper_visualization=False,
        use_indicator_object=False,
        rescale_actions=True,
        **kwargs
    ):
        """
        Args:
            controller_config (dict): If set, contains relevant controller parameters for creating a custom controller.
                Else, uses the default controller for this specific task

            gripper_right (str): type of gripper used on the right hand, used to
                instantiate gripper models from gripper factory.

            gripper_left (str): type of gripper used on the left hand.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that 
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in 
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes 
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes 
                in camera. False otherwise.

            control_freq (float): how many control signals to receive 
                in every second. This sets the amount of simulation time 
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            use_camera_obs (bool): if True, every observation includes a 
                rendered image.

            camera_name (str): name of camera to be rendered. Must be 
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """
        self.has_gripper_right = not (gripper_right is None)
        self.has_gripper_left = not (gripper_left is None)
        self.gripper_right_name = gripper_right
        self.gripper_left_name = gripper_left
        self.gripper_visualization = gripper_visualization
        self.use_indicator_object = use_indicator_object
        self.rescale_actions = rescale_actions
        self.controller_config = controller_config
        super().__init__(**kwargs)

    def _load_controller(self, controller_config):
        """
        Loads controller to be used for dynamic trajectories

        @controller_config (dict): Dict of relevant controller parameters, including controller type
            NOTE: Type must be one of: {JOINT_IMP, JOINT_TORQUE, JOINT_VEL, EE_POS, EE_POS_ORI}
        """
        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, robot_id, joint_indexes, timestep (model) freq, policy (control) freq, and ndim (# joints)
        controller_config["robot_name"] = self.mujoco_robot.name
        controller_config["sim"] = self.sim
        controller_config["controller_freq"] = 1.0 / self.model_timestep
        controller_config["policy_freq"] = self.control_freq
        controller_config["ndim"] = int(len(self.robot_joints) / 2)

        # Instantiate the relevant controllers (one for each arm)
        controller_config["eef_name"] = "right_hand"
        controller_config["joint_indexes"] = {
            "joints": self.joint_indexes[:controller_config["ndim"]],
            "qpos": self._ref_joint_pos_indexes[:controller_config["ndim"]],
            "qvel": self._ref_joint_vel_indexes[:controller_config["ndim"]]
        }
        self.right_controller = controller_factory(controller_config["type"], controller_config)
        controller_config["eef_name"] = "left_hand"
        controller_config["joint_indexes"] = {
            "joints": self.joint_indexes[controller_config["ndim"]:],
            "qpos": self._ref_joint_pos_indexes[controller_config["ndim"]:],
            "qvel": self._ref_joint_vel_indexes[controller_config["ndim"]:]
        }
        self.left_controller = controller_factory(controller_config["type"], controller_config)

    def _load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        super()._load_model()
        self.mujoco_robot = Baxter()
        if self.has_gripper_right:
            self.gripper_right = gripper_factory(self.gripper_right_name)
            if not self.gripper_visualization:
                self.gripper_right.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper_right)

        if self.has_gripper_left:
            self.gripper_left = gripper_factory(self.gripper_left_name)
            if not self.gripper_visualization:
                self.gripper_left.hide_visualization()
            self.mujoco_robot.add_gripper("left_hand", self.gripper_left)

    def _reset_internal(self):
        """Resets the pose of the arm and grippers."""
        super()._reset_internal()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos
        self._load_controller(self.controller_config)

        if self.has_gripper_right:
            self.sim.data.qpos[
                self._ref_gripper_right_joint_pos_indexes
            ] = self.gripper_right.init_qpos

        if self.has_gripper_left:
            self.sim.data.qpos[
                self._ref_gripper_left_joint_pos_indexes
            ] = self.gripper_left.init_qpos

    def _get_reference(self):
        """Sets up references for robots, grippers, and objects."""
        super()._get_reference()

        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]
        if self.use_indicator_object:
            ind_qpos = self.sim.model.get_joint_qpos_addr("pos_indicator")
            self._ref_indicator_pos_low, self._ref_indicator_pos_high = ind_qpos

            ind_qvel = self.sim.model.get_joint_qvel_addr("pos_indicator")
            self._ref_indicator_vel_low, self._ref_indicator_vel_high = ind_qvel

            self.indicator_id = self.sim.model.body_name2id("pos_indicator")

        # indices for grippers in qpos, qvel
        if self.has_gripper_left:
            self.gripper_left_joints = list(self.gripper_left.joints)
            self._ref_gripper_left_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_left_joints
            ]
            self._ref_gripper_left_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_left_joints
            ]
            self.left_eef_site_id = self.sim.model.site_name2id("l_g_grip_site")

        if self.has_gripper_right:
            self.gripper_right_joints = list(self.gripper_right.joints)
            self._ref_gripper_right_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_right_joints
            ]
            self._ref_gripper_right_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_right_joints
            ]
            self.right_eef_site_id = self.sim.model.site_name2id("grip_site")

        # indices for joint indexes
        self._ref_joint_indexes = [
            self.sim.model.joint_name2id(joint)
            for joint in self.sim.model.joint_names
            if joint.startswith("right") or joint.startswith("left")
        ]

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("pos")
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("vel")
        ]

        self._ref_joint_torq_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("torq")
        ]

        if self.has_gripper_left:
            self._ref_joint_gripper_left_actuator_indexes = [
                self.sim.model.actuator_name2id(actuator)
                for actuator in self.sim.model.actuator_names
                if actuator.startswith("gripper_l")
            ]

        if self.has_gripper_right:
            self._ref_joint_gripper_right_actuator_indexes = [
                self.sim.model.actuator_name2id(actuator)
                for actuator in self.sim.model.actuator_names
                if actuator.startswith("gripper_r")
            ]

        if self.has_gripper_right:
            # IDs of sites for gripper visualization
            self.eef_site_id = self.sim.model.site_name2id("grip_site")
            self.eef_cylinder_id = self.sim.model.site_name2id("grip_site_cylinder")

    def move_indicator(self, pos):
        """Moves the position of the indicator object to @pos."""
        if self.use_indicator_object:
            self.sim.data.qpos[
                self._ref_indicator_pos_low : self._ref_indicator_pos_low + 3
            ] = pos

    # Note: Overrides super
    def _pre_action(self, action, policy_step=False):
        # Optionally (and by default) rescale actions to [-1, 1]. Not desirable
        # for certain controllers. They later get normalized to the control range.
        if self.rescale_actions:
            action = np.clip(action, -1, 1)

        # Action is stored as [right arm, left arm, right gripper?, left gripper?]
        # We retrieve the relevant actions.
        last = len(action)

        # Left gripper
        if self.has_gripper_left:
            gripper_left_action_in = action[last-self.gripper_left.dof: last]
            gripper_left_action_actual = self.gripper_left.format_action(
                gripper_left_action_in
            )
            last -= self.gripper_left.dof

        # Right gripper
        if self.has_gripper_right:
            gripper_right_action_in = action[last-self.gripper_right.dof: last]
            gripper_right_action_actual = self.gripper_right.format_action(
                gripper_right_action_in
            )
            last -= self.gripper_right.dof

        arm_split = int(last / 2)
        right_action = action[:arm_split]
        left_action = action[arm_split:last]

        # Update model in controller
        self.left_controller.update()
        self.right_controller.update()

        # Update the controller goals if this is a new policy step
        if policy_step:
            self.left_controller.set_goal(left_action)
            self.right_controller.set_goal(right_action)

        # Now run the controller for a step
        left_torques = self.left_controller.run_controller()
        right_torques = self.right_controller.run_controller()

        # Clip the torques
        low, high = self.torque_spec
        self.torques = np.clip(np.concatenate((right_torques, left_torques)), low, high)

        # Control gripper if applicable
        if self.has_gripper_left:
            # rescale normalized action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange[self._ref_joint_gripper_left_actuator_indexes]
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_left_gripper_action = bias + weight * gripper_left_action_actual
            self.sim.data.ctrl[self._ref_joint_gripper_left_actuator_indexes] = applied_left_gripper_action
        if self.has_gripper_right:
            # rescale normalized action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange[self._ref_joint_gripper_right_actuator_indexes]
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_right_gripper_action = bias + weight * gripper_right_action_actual
            self.sim.data.ctrl[self._ref_joint_gripper_right_actuator_indexes] = applied_right_gripper_action

        # Apply joint torque control (with gravity compensation)
        self.sim.data.ctrl[self._ref_joint_torq_actuator_indexes] = self.sim.data.qfrc_bias[
                                                                        self._ref_joint_vel_indexes] + self.torques

        if self.use_indicator_object:
            # Apply gravity compensation to indicator object too
            self.sim.data.qfrc_applied[
                self._ref_indicator_vel_low : self._ref_indicator_vel_high
            ] = self.sim.data.qfrc_bias[
                self._ref_indicator_vel_low : self._ref_indicator_vel_high
            ]

    def _post_action(self, action):
        """Optionally performs gripper visualization after the actions."""
        ret = super()._post_action(action)
        self._gripper_visualization()
        return ret

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        
        Important keys:
            robot-state: contains robot-centric information.
        """
        di = super()._get_observation()
        # proprioceptive features
        di["joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        )
        di["joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        )
        robot_states = [
            np.sin(di["joint_pos"]),
            np.cos(di["joint_pos"]),
            di["joint_vel"],
        ]

        if self.has_gripper_right:
            di["right_gripper_qpos"] = np.array(
                [
                    self.sim.data.qpos[x]
                    for x in self._ref_gripper_right_joint_pos_indexes
                ]
            )
            di["right_gripper_qvel"] = np.array(
                [
                    self.sim.data.qvel[x]
                    for x in self._ref_gripper_right_joint_vel_indexes
                ]
            )
            di["right_eef_pos"] = np.array(self.sim.data.site_xpos[self.right_eef_site_id])
            di["right_eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )
            robot_states.extend(
                [di["right_gripper_qpos"], di["right_eef_pos"], di["right_eef_quat"]]
            )

        if self.has_gripper_left:
            di["left_gripper_qpos"] = np.array(
                [
                    self.sim.data.qpos[x]
                    for x in self._ref_gripper_left_joint_pos_indexes
                ]
            )
            di["left_gripper_qvel"] = np.array(
                [
                    self.sim.data.qvel[x]
                    for x in self._ref_gripper_left_joint_vel_indexes
                ]
            )
            di["left_eef_pos"] = np.array(self.sim.data.site_xpos[self.left_eef_site_id])
            di["left_eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("left_hand"), to="xyzw"
            )
            robot_states.extend(
                [di["left_gripper_qpos"], di["left_eef_pos"], di["left_eef_quat"]]
            )

        di["robot-state"] = np.concatenate(robot_states)
        return di

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        # Action limits based on controller limits
        gripper_count = int(self.has_gripper_left) + int(self.has_gripper_right)
        low, high = [-1] * gripper_count, [1] * gripper_count
        low = np.concatenate([self.right_controller.input_min, self.left_controller.input_min, low])
        high = np.concatenate([self.right_controller.input_max, self.left_controller.input_max, high])

        return low, high

    @property
    def torque_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        # Torque limit values pulled from relevant robot.xml file
        low = self.sim.model.actuator_ctrlrange[self._ref_joint_torq_actuator_indexes, 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_joint_torq_actuator_indexes, 1]

        return low, high

    @property
    def dof(self):
        """Returns the DoF of the robot (with grippers)."""
        dof = self.mujoco_robot.dof
        if self.has_gripper_left:
            dof += self.gripper_left.dof
        if self.has_gripper_right:
            dof += self.gripper_right.dof
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame as a numpy
        array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp("right_hand").reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes[:7]]

        Jr = self.sim.data.get_body_jacr("right_hand").reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes[:7]]

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
    def _right_hand_quat(self):
        """
        Returns eef orientation of right hand in base from of robot.
        """
        return T.mat2quat(self._right_hand_orn)

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

    @property
    def _left_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("left_hand")

    @property
    def _left_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame as a numpy
        array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp("left_hand").reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes[7:]]

        Jr = self.sim.data.get_body_jacr("left_hand").reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes[7:]]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _left_hand_pos(self):
        """
        Returns position of eef in base frame of robot. 
        """
        eef_pose_in_base = self._left_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _left_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._left_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _left_hand_quat(self):
        """
        Returns eef orientation of left hand in base from of robot.
        """
        return T.mat2quat(self._left_hand_orn)

    @property
    def _left_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._left_hand_total_velocity[:3]

    @property
    def _left_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._left_hand_total_velocity[3:]

    @property
    def _joint_positions(self):
        """Returns a numpy array of joint positions (angles), of dimension 14."""
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """Returns a numpy array of joint (angular) velocities, of dimension 14."""
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    @property
    def joint_indexes(self):
        """
        Returns mujoco internal indexes for the robot joints
        """
        return self._ref_joint_indexes

    @property
    def _l_eef_xpos(self):
        """Returns the position of the left hand."""
        return self.sim.data.site_xpos[self.left_eef_site_id]

    @property
    def _r_eef_xpos(self):
        """Returns the position of the right hand."""
        return self.sim.data.site_xpos[self.right_eef_site_id]

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Using defaults.
        """
        pass

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False
