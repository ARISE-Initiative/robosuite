import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.robot_env import RobotEnv
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string

from robosuite.models.objects import CylinderObject, PlateWithHoleObject
from robosuite.models.arenas import EmptyArena
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import check_bimanual


class TwoArmPegInHole(RobotEnv):
    """
    This class corresponds to the peg-in-hole task for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be either 2 single single-arm robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment. Can be either:

            :`'bimanual'`: Only applicable for bimanual robot setups. Sets up the (single) bimanual robot on the -x
                side of the table
            :`'single-arm-parallel'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots next to each other on the -x side of the table
            :`'single-arm-opposed'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots opposed from each others on the opposite +/-y sides of the table (Default option)

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory.
            For this environment, setting a value other than the default (None) will raise an AssertionError, as
            this environment is not meant to be used with any gripper at all.

        gripper_visualizations (bool or list of bool): True if using gripper visualization.
            Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool or list of bool): if True, every observation for a specific robot includes a rendered
        image. Should either be single bool if camera obs value is to be used for all
            robots or else it should be a list of the same length as "robots" param

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        peg_radius (2-tuple): low and high limits of the (uniformly sampled)
            radius of the peg

        peg_length (float): length of the peg

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

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
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

    Raises:
        AssertionError: [Gripper specified]
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        env_configuration="single-arm-opposed",
        controller_configs=None,
        gripper_types=None,
        gripper_visualizations=False,
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        peg_radius=(0.015, 0.03),
        peg_length=0.13,
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
        # First, verify that correct number of robots are being inputted
        self.env_configuration = env_configuration
        self._check_robot_configuration(robots)

        # Assert that the gripper type is None
        assert gripper_types is None, "Tried to specify gripper other than None in TwoArmPegInHole environment!"

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # Save peg specs
        self.peg_radius = peg_radius
        self.peg_length = peg_length

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

    def reward(self, action):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 5.0 is provided if the peg is inside the plate's hole
              - Note that we enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arms to approach each other
            - Perpendicular Distance: in [0,1], to encourage the arms to approach each other
            - Parallel Distance: in [0,1], to encourage the arms to approach each other
            - Alignment: in [0, 1], to encourage having the right orientation between the peg and hole.
            - Placement: in {0, 1}, nonzero if the peg is in the hole with a relatively correct alignment

        Note that the final reward is normalized and scaled by reward_scale / 5.0 as
        well so that the max score is equal to reward_scale

        """
        reward = 0

        t, d, cos = self._compute_orientation()

        # Right location and angle
        if d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95:
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            # reaching reward
            hole_pos = self.sim.data.body_xpos[self.hole_body_id]
            gripper_site_pos = self.sim.data.body_xpos[self.peg_body_id]
            dist = np.linalg.norm(gripper_site_pos - hole_pos)
            reaching_reward = 1 - np.tanh(1.0 * dist)
            reward += reaching_reward

            # Orientation reward
            reward += 1 - np.tanh(d)
            reward += 1 - np.tanh(np.abs(t))
            reward += cos

        # if we're not reward shaping, we need to scale our sparse reward so that the max reward is identical
        # to its dense version
        else:
            reward *= 5.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 5.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["empty"]
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["empty"]
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # Add arena and robot
        self.model = MujocoWorldBase()
        self.mujoco_arena = EmptyArena()
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()
        self.model.merge(self.mujoco_arena)
        for robot in self.robots:
            self.model.merge(robot.robot_model)

        # initialize objects of interest
        self.hole = PlateWithHoleObject(
            name="hole",
        )
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.peg = CylinderObject(
            name="peg",
            size_min=(self.peg_radius[0], self.peg_length),
            size_max=(self.peg_radius[1], self.peg_length),
            material=greenwood,
            rgba=[0, 1, 0, 1],
        )

        # Load hole object
        self.hole_obj = self.hole.get_collision(site=True)
        self.hole_obj.set("quat", "0 0 0.707 0.707")
        self.hole_obj.set("pos", "0.11 0 0.17")
        self.model.merge_asset(self.hole)

        # Load peg object
        self.peg_obj = self.peg.get_collision(site=True)
        self.peg_obj.set("pos", array_to_string((0, 0, self.peg_length)))
        self.model.merge_asset(self.peg)

        # Depending on env configuration, append appropriate objects to arms
        if self.env_configuration == "bimanual":
            self.model.worldbody.find(".//body[@name='{}']"
                                      .format(self.robots[0].robot_model.eef_name["left"])).append(self.hole_obj)
            self.model.worldbody.find(".//body[@name='{}']"
                                      .format(self.robots[0].robot_model.eef_name["right"])).append(self.peg_obj)
        else:
            self.model.worldbody.find(".//body[@name='{}']"
                                      .format(self.robots[1].robot_model.eef_name)).append(self.hole_obj)
            self.model.worldbody.find(".//body[@name='{}']"
                                      .format(self.robots[0].robot_model.eef_name)).append(self.peg_obj)

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id("hole")
        self.peg_body_id = self.sim.model.body_name2id("peg")

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

        Returns:
            OrderedDict: Observations from the environment
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

            # position and rotation of peg and hole
            hole_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
            hole_quat = T.convert_quat(
                self.sim.data.body_xquat[self.hole_body_id], to="xyzw"
            )
            di["hole_pos"] = hole_pos
            di["hole_quat"] = hole_quat

            peg_pos = np.array(self.sim.data.body_xpos[self.peg_body_id])
            peg_quat = T.convert_quat(
                self.sim.data.body_xquat[self.peg_body_id], to="xyzw"
            )
            di["peg_to_hole"] = peg_pos - hole_pos
            di["peg_quat"] = peg_quat

            # Relative orientation parameters
            t, d, cos = self._compute_orientation()
            di["angle"] = cos
            di["t"] = t
            di["d"] = d

            di["object-state"] = np.concatenate(
                [
                    di["hole_pos"],
                    di["hole_quat"],
                    di["peg_to_hole"],
                    di["peg_quat"],
                    [di["angle"]],
                    [di["t"]],
                    [di["d"]],
                ]
            )

        return di

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        t, d, cos = self._compute_orientation()

        return d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95

    def _compute_orientation(self):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.

        Returns:
            3-tuple:

                - (float): parallel distance
                - (float): perpendicular distance
                - (float): angle
        """
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]

        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)

        v = peg_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center = hole_pos + hole_mat @ np.array([0.1, 0, 0])

        t = (center - peg_pos) @ v / (np.linalg.norm(v) ** 2)
        d = np.linalg.norm(np.cross(v, peg_pos - center)) / np.linalg.norm(v)

        hole_normal = hole_mat @ np.array([0, 0, 1])
        return (
            t,
            d,
            abs(
                np.dot(hole_normal, v) / np.linalg.norm(hole_normal) / np.linalg.norm(v)
            ),
        )

    def _peg_pose_in_hole_frame(self):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.

        Returns:
            np.array: (4,4) matrix corresponding to the pose of the peg in the hole frame
        """
        # World frame
        peg_pos_in_world = self.sim.data.get_body_xpos("peg")
        peg_rot_in_world = self.sim.data.get_body_xmat("peg").reshape((3, 3))
        peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)

        # World frame
        hole_pos_in_world = self.sim.data.get_body_xpos("hole")
        hole_rot_in_world = self.sim.data.get_body_xmat("hole").reshape((3, 3))
        hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)

        world_pose_in_hole = T.pose_inv(hole_pose_in_world)

        peg_pose_in_hole = T.pose_in_A_to_pose_in_B(
            peg_pose_in_world, world_pose_in_hole
        )
        return peg_pose_in_hole

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
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
