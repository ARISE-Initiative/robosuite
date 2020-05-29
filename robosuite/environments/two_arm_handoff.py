from collections import OrderedDict
import numpy as np

from robosuite.environments.robot_env import RobotEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import HammerObject
from robosuite.models.tasks import TableTopTask, UniformRandomSampler
from robosuite.models.robots import check_bimanual

import robosuite.utils.transform_utils as T


class TwoArmHandoff(RobotEnv):
    """
    This class corresponds to the handoff task for two robot arms.
    """

    def __init__(
        self,
        robots,
        env_configuration="single-arm-opposed",
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        initialization_noise="default",
        prehensile=True,
        table_full_size=(0.8, 1.2, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=2.0,
        reward_shaping=False,
        placement_initializer=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        """
        Args:
            robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
                (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
                Note: Must be either 2 single single-arm robots or 1 bimanual robot!

            env_configuration (str): Specifies how to position the robots within the environment. Can be either:
                "bimanual": Only applicable for bimanual robot setups. Sets up the (single) bimanual robot on the -x
                    side of the table
                "single-arm-parallel": Only applicable for multi single arm setups. Sets up the (two) single armed
                    robots next to each other on the -x side of the table
                "single-arm-opposed": Only applicable for multi single arm setups. Sets up the (two) single armed
                    robots opposed from each others on the opposite +/-y sides of the table (Default option)

            controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
                custom controller. Else, uses the default controller for this specific task. Should either be single
                dict if same controller is to be used for all robots or else it should be a list of the same length as
                "robots" param

            gripper_types (str or list of str): type of gripper, used to instantiate
                gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
                with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
                overrides the default gripper. Should either be single str if same gripper type is to be used for all
                robots or else it should be a list of the same length as "robots" param

            gripper_visualizations (bool or list of bool): True if using gripper visualization.
                Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
                robots or else it should be a list of the same length as "robots" param

            initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
                The expected keys and corresponding value types are specified below:
                "magnitude": The scale factor of uni-variate random noise applied to each of a robot's given initial
                    joint positions. Setting this value to "None" or 0.0 results in no noise being applied.
                    If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                    If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
                "type": Type of noise to apply. Can either specify "gaussian" or "uniform"
                Should either be single dict if same noise value is to be used for all robots or else it should be a
                list of the same length as "robots" param
                Note: Specifying "default" will automatically use the default noise settings for this task
                    (see __init__() call below)
                    Specifying None will automatically create the required dict with "magnitude" set to 0.0

            prehensile (bool): If true, handoff object starts on the table. Else, the object starts in Arm0's gripper

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes rendered image(s)

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_scale (float): Scales the normalized reward function by the amount specified

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering

            render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
                will result in the default angle being applied, which is useful as it can be dragged / panned by
                the user using the mouse

            render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

            control_freq (float): how many control signals to receive in every second. This sets the amount of
                simulation time that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
                only calls sim.reset and resets all robosuite-internal variables

            camera_names (str or list of str): name of camera to be rendered. Should either be single str if
                same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
                Note: At least one camera must be specified if @use_camera_obs is True.
                Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                    convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                    robot's camera list).

            camera_heights (int or list of int): height of camera frame. Should either be single int if
                same height is to be used for all cameras' frames or else it should be a list of the same length as
                "camera names" param.

            camera_widths (int or list of int): width of camera frame. Should either be single int if
                same width is to be used for all cameras' frames or else it should be a list of the same length as
                "camera names" param.

            camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
                bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
                "camera names" param.

        """
        # First, verify that correct number of robots are being inputted
        self.env_configuration = env_configuration
        self._check_robot_configuration(robots)

        # Task settings
        self.prehensile = prehensile

        # settings for table top
        self.table_full_size = table_full_size
        self.table_true_size = list(table_full_size)
        self.table_true_size[1] *= 0.25     # true size will only be partially wide
        self.table_friction = table_friction
        self.table_offset = [0, self.table_full_size[1] * (-3/8), 0]

        # reward configuration
        self._success = False
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # Setup default initialization noise if not customized
        if initialization_noise == "default":
            initialization_noise = {"magnitude": 0.02, "type": "gaussian"}

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.1, 0.1],
                y_range=[-0.5, -0.4],
                ensure_object_boundary_in_range=False,
                z_rotation=None,
            )

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            gripper_visualizations=gripper_visualizations,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:
          1. a discrete reward of 2.0 is provided when only Arm 1 is gripping the handle and has the handle
             lifted above a certain threshold

        Un-normalized piecewise components if using reward shaping:

          1. a smoothed reward between 0 and 0.25 is provided relative to the distance between Arm 0 and the handle
          2. a discrete reward of 0.5 is provided when Arm 0 grips the handle.
          3. a discrete reward of 1.0 is provided when Arm 0 lifts the handle from the table
             past a certain threshold
          4. a smoothed reward between 1.0 and 1.25 is provided relative to the distance between the handle and Arm 1,
             conditioned on the handle being lifted from the table and being grasped by Arm 0
          5. a discrete reward of 1.5 is provided when both Arm 0 and Arm 1 are gripping the handle while
             lifted above the table
          6. the max reward of 2.0 is provided when only Arm 1 is gripping the handle and has the handle
             lifted above the table

        Note that the final reward is normalized and scaled by reward_scale / 2.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        # Initialize reward and set success to false
        reward = 0
        self._success = False

        # Get height of hammer and table and define height threshold
        hammer_angle_offset = (self.hammer.handle_length / 2 + 2*self.hammer.head_halfsize) * np.sin(self._hammer_angle)
        hammer_height = self.sim.data.geom_xpos[self.hammer_handle_geom_id][2]\
            - self.hammer.get_top_offset()[2]\
            - hammer_angle_offset
        table_height = self.sim.data.site_xpos[self.table_top_id][2]
        height_threshold = 0.1

        # Check if any Arm's gripper is grasping the hammer handle

        # Single bimanual robot setting
        if self.env_configuration == "bimanual":
            _contacts_0 = list(
                self.find_contacts(
                    self.robots[0].gripper["left"].contact_geoms, self.hammer.handle_geoms
                )
            )
            _contacts_1 = list(
                self.find_contacts(
                    self.robots[0].gripper["right"].contact_geoms, self.hammer.handle_geoms
                )
            )
        # Multi single arm setting
        else:
            _contacts_0 = list(
                self.find_contacts(
                    self.robots[0].gripper.contact_geoms, self.hammer.handle_geoms
                )
            )
            _contacts_1 = list(
                self.find_contacts(
                    self.robots[1].gripper.contact_geoms, self.hammer.handle_geoms
                )
            )
        arm0_grasp = True if len(_contacts_0) > 0 else False
        arm1_grasp = True if len(_contacts_1) > 0 else False

        # use a shaping reward if specified
        if self.reward_shaping:
            # First, we'll consider the cases if the hammer is lifted above the threshold (step 3 - 6)
            if hammer_height - table_height > height_threshold:
                # Split cases depending on whether arm1 is currently grasping the handle or not
                if arm1_grasp:
                    # Check if arm0 is grasping
                    if arm0_grasp:
                        # This is step 5
                        reward = 1.5
                    else:
                        # This is step 6 (completed task!)
                        reward = 2.0
                        self._success = True
                # This is the case where only arm0 is grasping (step 2-3)
                else:
                    reward = 1.0
                    # Add in up to 0.25 based on distance between handle and arm1
                    dist = np.linalg.norm(self._gripper_1_to_handle)
                    reaching_reward = 0.25*(1 - np.tanh(10.0 * dist))
                    reward += reaching_reward
            # Else, the hammer is still on the ground ):
            else:
                # Split cases depending on whether arm0 is currently grasping the handle or not
                if arm0_grasp:
                    # This is step 2
                    reward = 0.5
                else:
                    # This is step 1, we want to encourage arm0 to reach for the handle
                    dist = np.linalg.norm(self._gripper_0_to_handle)
                    reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
                    reward = reaching_reward

        # Else this is the sparse reward setting
        else:
            # Provide reward if only Arm 1 is grasping the hammer and the handle lifted above the pre-defined threshold
            if arm1_grasp and not arm0_grasp and hammer_height - table_height > height_threshold:
                reward = 2.0
                self._success = True

        return reward * self.reward_scale / 2.0

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation, offset in zip(self.robots, (np.pi/2, -np.pi/2), (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    xpos += np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:   # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.6, 0.6)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_true_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.hammer = HammerObject()
        self.mujoco_objects = OrderedDict([("hammer", self.hammer)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            [robot.robot_model for robot in self.robots],
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Hammer object references from this env
        self.hammer_body_id = self.sim.model.body_name2id("hammer")
        self.hammer_handle_geom_id = self.sim.model.geom_name2id("hammer_handle")
        self.hammer_head_geom_id = self.sim.model.geom_name2id("hammer_head")
        self.hammer_face_geom_id = self.sim.model.geom_name2id("hammer_face")
        self.hammer_claw_geom_id = self.sim.model.geom_name2id("hammer_claw")

        # General env references
        self.table_top_id = self.sim.model.site_name2id("table_top")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            obj_pos, obj_quat = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for i, (obj_name, _) in enumerate(self.mujoco_objects.items()):
                # If prehensile, set the object normally
                if self.prehensile:
                    self.sim.data.set_joint_qpos(obj_name,
                                                 np.concatenate([np.array(obj_pos[i]), np.array(obj_quat[i])]))
                # Else, set the object in the hand of the robot and loop a few steps to guarantee the robot is grasping
                #   the object initially
                else:
                    eef_rot_quat = T.mat2quat(T.euler2mat([np.pi - T.mat2euler(self._eef0_xmat)[2], 0, 0]))
                    obj_quat[i] = T.quat_multiply(obj_quat[i], eef_rot_quat)
                    for j in range(100):
                        # Set object in hand
                        self.sim.data.set_joint_qpos(obj_name,
                                                     np.concatenate([self._eef0_xpos, np.array(obj_quat[i])]))
                        # Close gripper (action = 1) and prevent arm from moving
                        if self.env_configuration == 'bimanual':
                            # Execute no-op action with gravity compensation
                            torques = np.concatenate([self.robots[0].controller["right"].torque_compensation,
                                                      self.robots[0].controller["left"].torque_compensation])
                            self.sim.data.ctrl[self.robots[0]._ref_joint_torq_actuator_indexes] = torques
                            # Execute gripper action
                            self.robots[0].grip_action([1], "right")
                        else:
                            # Execute no-op action with gravity compensation
                            self.sim.data.ctrl[self.robots[0]._ref_joint_torq_actuator_indexes] =\
                                self.robots[0].controller.torque_compensation
                            self.sim.data.ctrl[self.robots[1]._ref_joint_torq_actuator_indexes] = \
                                self.robots[1].controller.torque_compensation
                            # Execute gripper action
                            self.robots[0].grip_action([1])
                        # Take forward step
                        self.sim.step()

                    # Reset internal counters
                    #self.cur_time = 0
                    #self.timestep = 0

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix
            if self.env_configuration == "bimanual":
                pr0 = self.robots[0].robot_model.naming_prefix + "right_"
                pr1 = self.robots[0].robot_model.naming_prefix + "left_"
            else:
                pr0 = self.robots[0].robot_model.naming_prefix
                pr1 = self.robots[1].robot_model.naming_prefix

            # position and rotation of hammer
            di["hammer_pos"] = np.array(self._hammer_pos)
            di["hammer_quat"] = np.array(self._hammer_quat)
            di["handle_xpos"] = np.array(self._handle_xpos)

            di[pr0 + "eef_xpos"] = np.array(self._eef0_xpos)
            di[pr1 + "eef_xpos"] = np.array(self._eef1_xpos)
            di[pr0 + "gripper_to_handle"] = np.array(self._gripper_0_to_handle)
            di[pr1 + "gripper_to_handle"] = np.array(self._gripper_1_to_handle)

            di["object-state"] = np.concatenate(
                [
                    di["hammer_pos"],
                    di["hammer_quat"],
                    di["handle_xpos"],
                    di[pr0 + "eef_xpos"],
                    di[pr1 + "eef_xpos"],
                    di[pr0 + "gripper_to_handle"],
                    di[pr1 + "gripper_to_handle"],
                ]
            )

        return di

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        return self._success

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable
        """
        robots = robots if type(robots) == list or type(robots) == tuple else [robots]
        if self.env_configuration == "single-arm-opposed" or self.env_configuration == "single-arm-parallel":
            # Specifically two robots should be inputted!
            is_bimanual = False
            if type(robots) is not list or len(robots) != 2:
                raise ValueError("Error: Exactly two single-armed robots should be inputted "
                                 "for this task configuration!")
        elif self.env_configuration == "bimanual":
            is_bimanual = True
            # Specifically one robot should be inputted!
            if type(robots) is list and len(robots) != 1:
                raise ValueError("Error: Exactly one bimanual robot should be inputted "
                                 "for this task configuration!")
        else:
            # This is an unknown env configuration, print error
            raise ValueError("Error: Unknown environment configuration received. Only 'bimanual',"
                             "'single-arm-parallel', and 'single-arm-opposed' are supported. Got: {}"
                             .format(self.env_configuration))

        # Lastly, check to make sure all inputted robot names are of their correct type (bimanual / not bimanual)
        for robot in robots:
            if check_bimanual(robot) != is_bimanual:
                raise ValueError("Error: For {} configuration, expected bimanual check to return {}; "
                                 "instead, got {}.".format(self.env_configuration, is_bimanual, check_bimanual(robot)))

    @property
    def _handle_xpos(self):
        """Returns the position of the hammer handle."""
        return self.sim.data.geom_xpos[self.hammer_handle_geom_id]

    @property
    def _head_xpos(self):
        """Returns the position of the hammer head."""
        return self.sim.data.geom_xpos[self.hammer_head_geom_id]

    @property
    def _face_xpos(self):
        """Returns the position of the hammer face."""
        return self.sim.data.geom_xpos[self.hammer_face_geom_id]

    @property
    def _claw_xpos(self):
        """Returns the position of the hammer claw."""
        return self.sim.data.geom_xpos[self.hammer_claw_geom_id]

    @property
    def _hammer_pos(self):
        """Returns the orientation of the pot."""
        return np.array(self.sim.data.body_xpos[self.hammer_body_id])

    @property
    def _hammer_quat(self):
        """Returns the orientation of the pot."""
        return T.convert_quat(self.sim.data.body_xquat[self.hammer_body_id], to="xyzw")

    @property
    def _hammer_angle(self):
        """Returns angle of hammer with the ground, relative to it resting horizontally"""
        mat = T.quat2mat(self._hammer_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        return np.pi/2 - np.arccos(np.dot(z_unit, z_rotated))

    @property
    def _world_quat(self):
        """World quaternion."""
        return T.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    @property
    def _eef0_xpos(self):
        """End Effector 0 position"""
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])
        else:
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    @property
    def _eef1_xpos(self):
        """End Effector 1 position"""
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["left"]])
        else:
            return np.array(self.sim.data.site_xpos[self.robots[1].eef_site_id])

    @property
    def _eef0_xmat(self):
        """
        End Effector 0 orientation as a rotation matrix
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!
        """
        pf = self.robots[0].robot_model.naming_prefix
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "right_ee")]).reshape(3, 3)
        else:
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "ee")]).reshape(3, 3)

    @property
    def _eef1_xmat(self):
        """
        End Effector 1 orientation as a rotation matrix
        Note that this draws the orientation from the "right_/left_hand" body, NOT the gripper site, since the gripper
        orientations are inconsistent!
        """
        if self.env_configuration == "bimanual":
            pf = self.robots[0].robot_model.naming_prefix
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "left_ee")]).reshape(3, 3)
        else:
            pf = self.robots[1].robot_model.naming_prefix
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "ee")]).reshape(3, 3)

    @property
    def _gripper_0_to_handle(self):
        """Returns vector from the left gripper to the hammer handle."""
        return self._handle_xpos - self._eef0_xpos

    @property
    def _gripper_1_to_handle(self):
        """Returns vector from the right gripper to the hammer handle."""
        return self._handle_xpos - self._eef1_xpos
