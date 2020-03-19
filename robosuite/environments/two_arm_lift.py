from collections import OrderedDict
import numpy as np

from robosuite.environments.robot_env import RobotEnv
from robosuite.agents import *

from robosuite.models.arenas import TableArena
from robosuite.models.objects import PotWithHandlesObject
from robosuite.models.tasks import TableTopTask, UniformRandomSampler

import robosuite.utils.transform_utils as T


class TwoArmLift(RobotEnv):
    """
    This class corresponds to the lifting task for two robot arms.
    """

    def __init__(
        self,
        robots,
        env_configuration="single-arm-opposed",
        controller_configs=None,
        gripper_types="default",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        gripper_visualizations=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderers=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
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

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool or list of bool): if True, every observation for a specific robot includes a rendered
            image. Should either be single bool if camera obs value is to be used for all
                robots or else it should be a list of the same length as "robots" param

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderers (bool or list of bool): True if using off-screen rendering. Should either be single
                bool if same offscreen renderering setting is to be used for all cameras or else it should be a list of
                the same length as "robots" param

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_names (str or list of str): name of camera to be rendered. Should either be single str if
                same name is to be used for all cameras' rendering or else it should be a list of the same length as
                "robots" param. Note: Each name must be set if the corresponding @use_camera_obs value is True.

            camera_heights (int or list of int): height of camera frame. Should either be single int if
                same height is to be used for all cameras' frames or else it should be a list of the same length as
                "robots" param.

            camera_widths (int or list of int): width of camera frame. Should either be single int if
                same width is to be used for all cameras' frames or else it should be a list of the same length as
                "robots" param.

            camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
                bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
                "robots" param.
        """
        # First, verify that correct number of robots are being inputted
        self.env_configuration = env_configuration
        self._check_robot_configuration(robots)

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # reward configuration
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                z_rotation=None,
            )

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            gripper_visualizations=gripper_visualizations,
            use_camera_obs=use_camera_obs,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderers=has_offscreen_renderers,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

          1. the agent only gets the lifting reward when flipping no more than 30 degrees.
          2. the lifting reward is smoothed and ranged from 0 to 2, capped at 2.0.
             the initial lifting reward is 0 when the pot is on the table;
             the agent gets the maximum 2.0 reward when the pot’s height is above a threshold.
          3. the reaching reward is 0.5 when the left gripper touches the left handle,
             or when the right gripper touches the right handle before the gripper geom
             touches the handle geom, and once it touches we use 0.5

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        reward = 0

        cube_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.get_top_offset()[2]
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # check if the pot is tilted more than 30 degrees
        mat = T.quat2mat(self._pot_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_30 = np.cos(np.pi / 6)
        direction_coef = 1 if cos_z >= cos_30 else 0

        # cube is higher than the table top above a margin
        if cube_height > table_height + 0.15:
            reward = 1.0 * direction_coef

        # use a shaping reward
        if self.reward_shaping:
            reward = 0

            # lifting reward
            elevation = cube_height - table_height
            r_lift = min(max(elevation - 0.05, 0), 0.2)
            reward += 10. * direction_coef * r_lift

            _gripper_0_to_handle = self._gripper_0_to_handle
            _gripper_1_to_handle = self._gripper_1_to_handle

            # gh stands for gripper-handle
            # When grippers are far away, tell them to be closer

            # Single bimanual robot setting
            if self.env_configuration == "bimanual":
                _contacts_0 = list(
                    self.find_contacts(
                        self.robots[0].gripper["left"].contact_geoms, self.pot.handle_2_geoms()
                    )
                )
                _contacts_1 = list(
                    self.find_contacts(
                        self.robots[0].gripper["right"].contact_geoms, self.pot.handle_1_geoms()
                    )
                )
            # Multi single arm setting
            else:
                _contacts_0 = list(
                    self.find_contacts(
                        self.robots[0].gripper.contact_geoms, self.pot.handle_2_geoms()
                    )
                )
                _contacts_1 = list(
                    self.find_contacts(
                        self.robots[1].gripper.contact_geoms, self.pot.handle_1_geoms()
                    )
                )
            _g0h_dist = np.linalg.norm(_gripper_0_to_handle)
            _g1h_dist = np.linalg.norm(_gripper_1_to_handle)

            if len(_contacts_0) > 0:
                reward += 0.5
            else:
                reward += 0.5 * (1 - np.tanh(_g0h_dist))

            if len(_contacts_1) > 0:
                reward += 0.5
            else:
                reward += 0.5 * (1 - np.tanh(_g1h_dist))

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Vary the initial qpos of the robot
        for robot in self.robots:
            robot.init_qpos += np.random.randn(robot.init_qpos.shape[0]) * 0.02

        # Verify the correct robots have been loaded and adjust base pose accordingly
        # TODO: Account for variations in robot start position? Where 2nd robot will be placed?
        if self.env_configuration == "bimanual":
            assert isinstance(self.robots[0], Bimanual), "Error: For bimanual configuration, expected a " \
                "bimanual robot! Got {} type instead.".format(type(self.robots[0]))
            self.robots[0].robot_model.set_base_xpos([-0.29, 0, 0])
        else:
            assert isinstance(self.robots[0], SingleArm) and isinstance(self.robots[1], SingleArm), \
                "Error: For multi single arm configurations, expected two single-armed robot! " \
                "Got {} and {} types instead.".format(type(self.robots[0]), type(self.robots[1]))
            if self.env_configuration == "single-arm-opposed":
                self.robots[0].robot_model.set_base_xpos([0.55, -0.55, 0])
                self.robots[0].robot_model.set_base_ori([0,0,np.pi / 2])
                self.robots[1].robot_model.set_base_xpos([0.55, 0.55, 0])
                self.robots[1].robot_model.set_base_ori([0, 0, -np.pi/2])
            else:   # "single-arm-parallel" configuration setting
                self.robots[0].robot_model.set_base_xpos([0, -0.25, 0])
                self.robots[1].robot_model.set_base_xpos([0, 0.25, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Assumes the robot has a pedestal, we want to align it with the table
        # TODO: Add specs in robot model to account for varying base positions maybe?
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # initialize objects of interest
        self.pot = PotWithHandlesObject()
        self.mujoco_objects = OrderedDict([("pot", self.pot)])

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

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id("pot")
        self.handle_1_site_id = self.sim.model.site_name2id("pot_handle_1")
        self.handle_0_site_id = self.sim.model.site_name2id("pot_handle_2")
        self.table_top_id = self.sim.model.site_name2id("table_top")
        self.pot_center_id = self.sim.model.site_name2id("pot_center")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

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
                pr0 = self.robots[0].robot_model.naming_prefix + "left_"
                pr1 = self.robots[0].robot_model.naming_prefix + "right_"
            else:
                pr0 = self.robots[0].robot_model.naming_prefix
                pr1 = self.robots[1].robot_model.naming_prefix

            # position and rotation of object
            cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
            cube_quat = T.convert_quat(
                self.sim.data.body_xquat[self.cube_body_id], to="xyzw"
            )
            di["cube_pos"] = cube_pos
            di["cube_quat"] = cube_quat

            di[pr0 + "eef_xpos"] = self._eef0_xpos
            di[pr1 + "eef_xpos"] = self._eef1_xpos
            di["handle_0_xpos"] = np.array(self._handle_0_xpos)
            di["handle_1_xpos"] = np.array(self._handle_1_xpos)
            di[pr0 + "gripper_to_handle"] = np.array(self._gripper_0_to_handle)
            di[pr1 + "gripper_to_handle"] = np.array(self._gripper_1_to_handle)

            di["object-state"] = np.concatenate(
                [
                    di["cube_pos"],
                    di["cube_quat"],
                    di[pr0 + "eef_xpos"],
                    di[pr1 + "eef_xpos"],
                    di["handle_0_xpos"],
                    di["handle_1_xpos"],
                    di[pr0 + "gripper_to_handle"],
                    di[pr1 + "gripper_to_handle"],
                ]
            )

        return di

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_full_size[2]

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.10

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable
        """
        if self.env_configuration == "single-arm-opposed" or self.env_configuration == "single-arm-parallel":
            # Specifically two robots should be inputted!
            if type(robots) is not list or len(robots) != 2:
                raise ValueError("Error: Exactly two single-armed robots should be inputted "
                                 "for this task configuration!")
        elif self.env_configuration == "bimanual":
            # Specifically one robot should be inputted!
            if type(robots) is list and len(robots) != 1:
                raise ValueError("Error: Exactly one bimanual robot should be inputted "
                                 "for this task configuration!")
        else:
            # This is an unknown env configuration, print error
            raise ValueError("Error: Unknown environment configuration received. Only 'bimanual',"
                             "'single-arm-parallel', and 'single-arm-opposed' are supported. Got: {}"
                             .format(self.env_configuration))

    @property
    def _handle_0_xpos(self):
        """Returns the position of the first handle."""
        return self.sim.data.site_xpos[self.handle_0_site_id]

    @property
    def _handle_1_xpos(self):
        """Returns the position of the second handle."""
        return self.sim.data.site_xpos[self.handle_1_site_id]

    @property
    def _pot_quat(self):
        """Returns the orientation of the pot."""
        return T.convert_quat(self.sim.data.body_xquat[self.cube_body_id], to="xyzw")

    @property
    def _world_quat(self):
        """World quaternion."""
        return T.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    @property
    def _eef0_xpos(self):
        """End Effector 0 position"""
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["left"]])
        else:
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    @property
    def _eef1_xpos(self):
        """End Effector 1 position"""
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])
        else:
            return np.array(self.sim.data.site_xpos[self.robots[1].eef_site_id])

    @property
    def _gripper_0_to_handle(self):
        """Returns vector from the left gripper to the handle."""
        if self.env_configuration == "bimanual":
            return self._handle_0_xpos - self.robots[0].eef_site_id["left"]
        else:
            return self._handle_0_xpos - self.robots[0].eef_site_id

    @property
    def _gripper_1_to_handle(self):
        """Returns vector from the right gripper to the handle."""
        if self.env_configuration == "bimanual":
            return self._handle_0_xpos - self.robots[0].eef_site_id["right"]
        else:
            return self._handle_1_xpos - self.robots[1].eef_site_id
