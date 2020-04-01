from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments import MujocoEnv

from robosuite.models.grippers import gripper_factory
from robosuite.controllers import controller_factory
from robosuite.models.robots import Sawyer


class SawyerEnv(MujocoEnv):
    """Initializes a Sawyer robot environment."""

    def __init__(
        self,
        controller_config,
        gripper_type=None,
        gripper_visualization=False,
        use_indicator_object=False,
        indicator_num=1,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        use_camera_obs=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        eval_mode=False,
        num_evals=50,
        perturb_evals=False,
    ):
        """
        Args:
            controller_config (dict): If set, contains relevant controller parameters for creating a custom controller.
                Else, uses the default controller for this specific task

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            indicator_num (int): number of indicator objects to add to the environment.
                Only used if @use_indicator_object is True.

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

        self.has_gripper = gripper_type is not None
        self.gripper_type = gripper_type
        self.gripper_visualization = gripper_visualization
        self.use_indicator_object = use_indicator_object
        self.indicator_num = indicator_num
        self.controller_config = controller_config

        self.eval_mode = eval_mode
        self.num_evals = num_evals
        self.perturb_evals = perturb_evals
        if self.eval_mode:
            # replace placement initializer with one for consistent task evaluations!
            self._get_placement_initializer_for_eval_mode()

        super().__init__(
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

    def _get_placement_initializer_for_eval_mode(self):
        """
        This method is used by subclasses to implement a 
        placement initializer that is used to initialize the
        environment into a fixed set of known task instances.
        This is for reproducibility in policy evaluation.
        """
        raise Exception("Must implement this in subclass.")

    def _load_controller(self, controller_config):
        """
        Loads controller to be used for dynamic trajectories

        @controller_config (dict): Dict of relevant controller parameters, including controller type
            NOTE: Type must be one of: {JOINT_IMP, JOINT_TOR, JOINT_VEL, EE_POS, EE_POS_ORI, EE_IK}
        """
        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, robot_id, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        controller_config["robot_name"] = self.mujoco_robot.name
        controller_config["sim"] = self.sim
        controller_config["eef_name"] = "right_hand"
        controller_config["joint_indexes"] = {
            "joints": self.joint_indexes,
            "qpos": self._ref_joint_pos_indexes,
            "qvel": self._ref_joint_vel_indexes
                                              }
        controller_config["actuator_range"] = self.torque_spec
        controller_config["controller_freq"] = 1.0 / self.model_timestep
        controller_config["policy_freq"] = self.control_freq
        controller_config["ndim"] = len(self.robot_joints)

        # Instantiate the relevant controller
        self.controller = controller_factory(controller_config["type"], controller_config)

    def _load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        super()._load_model()
        self.mujoco_robot = Sawyer()
        self.init_qpos = self.mujoco_robot.init_qpos
        if self.has_gripper:
            self.gripper = gripper_factory(self.gripper_type)
            if not self.gripper_visualization:
                self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)

    def _reset_internal(self):
        """
        Sets initial pose of arm and grippers.
        """
        super()._reset_internal()
        self._has_interaction = False

        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.init_qpos
        self._load_controller(self.controller_config)

        if self.has_gripper:
            self.sim.data.qpos[
                self._ref_gripper_joint_pos_indexes
            ] = self.gripper.init_qpos

    def _get_reference(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
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
            self._ref_indicator_pos_low = [0] * self.indicator_num
            self._ref_indicator_pos_high = [0] * self.indicator_num
            self._ref_indicator_vel_low = [0] * self.indicator_num
            self._ref_indicator_vel_high = [0] * self.indicator_num
            self.indicator_id = [0] * self.indicator_num
            for i in range(self.indicator_num):
                ind_qpos = self.sim.model.get_joint_qpos_addr("pos_indicator_{}".format(i))
                self._ref_indicator_pos_low[i], self._ref_indicator_pos_high[i] = ind_qpos

                ind_qvel = self.sim.model.get_joint_qvel_addr("pos_indicator_{}".format(i))
                self._ref_indicator_vel_low[i], self._ref_indicator_vel_high[i] = ind_qvel

                self.indicator_id[i] = self.sim.model.body_name2id("pos_indicator_{}".format(i))

        # indices for grippers in qpos, qvel
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self._ref_gripper_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
            ]
            self._ref_gripper_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
            ]

        # indices for joint indexes
        self._ref_joint_indexes = [
            # TODO: In arm refactoring, will need to think carefully about how to solve this name-overlap for multiagents
            self.sim.model.joint_name2id(joint)
            for joint in self.sim.model.joint_names
            if joint.startswith("right")
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

        if self.has_gripper:
            self._ref_joint_gripper_actuator_indexes = [
                self.sim.model.actuator_name2id(actuator)
                for actuator in self.sim.model.actuator_names
                if actuator.startswith("gripper")
            ]

        # IDs of sites for gripper visualization
        self.eef_site_id = self.sim.model.site_name2id("grip_site")
        self.eef_cylinder_id = self.sim.model.site_name2id("grip_site_cylinder")

    def move_indicator(self, pos, indicator_index=0):
        """
        Sets 3d position of indicator object to @pos.
        """
        if self.use_indicator_object:
            index = self._ref_indicator_pos_low[indicator_index]
            self.sim.data.qpos[index: index + 3] = pos

    def step(self, action):
        if not self._has_interaction and self.eval_mode:
            # this is the first step call of the episode
            self.placement_initializer.increment_counter()
        self._has_interaction = True
        return super().step(action)

    def _pre_action(self, action, policy_step=False):
        """
        Overrides the superclass method to actuate the robot with the 
        passed joint velocities and gripper control.

        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.mujoco_robot.dof dimensions should be the desired 
                normalized joint velocities and if the robot has 
                a gripper, the next @self.gripper.dof dimensions should be
                actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken
        """

        # clip actions into valid range
        assert len(action) == self.controller.control_dim + self.gripper.dof, \
            "environment got invalid action dimension -- expected {}, got {}".format(
                self.controller.control_dim+self.gripper.dof, len(action))

        gripper_action = None
        if self.has_gripper:
            gripper_action = action[-self.gripper.dof:]  # all indexes at end
            action = action[:-self.gripper.dof]

        # Update model in controller
        self.controller.update()

        # Update the controller goal if this is a new policy step
        if policy_step:
            self.controller.set_goal(action)

        # Now run the controller for a step
        torques = self.controller.run_controller()

        # Clip the torques
        low, high = self.torque_spec
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

        if self.use_indicator_object:
            for i in range(self.indicator_num):
                # Apply gravity compensation to indicator object too
                self.sim.data.qfrc_applied[
                self._ref_indicator_vel_low[i]: self._ref_indicator_vel_high[i]
                ] = self.sim.data.qfrc_bias[
                    self._ref_indicator_vel_low[i]: self._ref_indicator_vel_high[i]]

    def _post_action(self, action):
        """
        (Optional) does gripper visualization after actions.
        """
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

        if self.has_gripper:
            di["gripper_qpos"] = np.array(
                [self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes]
            )
            di["gripper_qvel"] = np.array(
                [self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes]
            )

            di["eef_pos"] = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )
            di["eef_vlin"] = np.array(self.sim.data.get_body_xvelp("right_hand"))
            di["eef_vang"] = np.array(self.sim.data.get_body_xvelr("right_hand"))

            # add in gripper information
            robot_states.extend([di["gripper_qpos"], di["eef_pos"], di["eef_quat"], di["eef_vlin"], di["eef_vang"]])

        di["robot-state"] = np.concatenate(robot_states)
        return di

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        # Action limits based on controller limits
        low, high = [-1] * self.has_gripper, [1] * self.has_gripper
        low = np.concatenate([self.controller.input_min, low])
        high = np.concatenate([self.controller.input_max, high])

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
        """
        Returns the DoF of the robot (with grippers).
        """
        dof = self.mujoco_robot.dof
        if self.has_gripper:
            dof += self.gripper.dof
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
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
    def _right_hand_joint_cartesian_pose(self):
        """
        Returns the cartesian pose of the last robot joint in base frame of robot.
        """
        return self.pose_in_base_from_name("right_l6")

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

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
        Jp = self.sim.data.get_body_jacp("right_hand").reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes]

        Jr = self.sim.data.get_body_jacr("right_hand").reshape((3, -1))
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

    @property
    def _joint_positions(self):
        """
        Returns a numpy array of joint positions.
        Sawyer robots have 7 joints and positions are in rotation angles.
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        Sawyer robots have 7 joints and velocities are angular velocities.
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    @property
    def joint_indexes(self):
        """
        Returns mujoco internal indexes for the robot joints
        """
        return self._ref_joint_indexes

    def _gripper_visualization(self):
        """
        Do any needed visualization here.
        """

        # By default, don't do any coloring.
        self.sim.model.site_rgba[self.eef_site_id] = [0., 0., 0., 0.]

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False
