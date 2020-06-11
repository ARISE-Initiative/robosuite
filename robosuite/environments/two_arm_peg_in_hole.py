from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.robot_env import RobotEnv

from robosuite.models.objects import CylinderObject, PlateWithHoleObject
from robosuite.models.arenas import EmptyArena
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import check_bimanual


class TwoArmPegInHole(RobotEnv):
    """
    This class corresponds to the peg-in-hole task for two robot arms.
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
        reward_scale=5.0,
        reward_shaping=False,
        cylinder_radius=(0.015, 0.03),
        cylinder_length=0.13,
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
                Note: Specifying "default" will automatically use the default noise settings
                    Specifying None will automatically create the required dict with "magnitude" set to 0.0

            use_camera_obs (bool or list of bool): if True, every observation for a specific robot includes a rendered
            image. Should either be single bool if camera obs value is to be used for all
                robots or else it should be a list of the same length as "robots" param

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_scale (float): Scales the normalized reward function by the amount specified

            reward_shaping (bool): if True, use dense rewards.

            cylinder_radius (2-tuple): low and high limits of the (uniformly sampled)
                radius of the cylinder

            cylinder_length (float): length of the cylinder

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

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # Save cylinder specs
        self.cylinder_radius = cylinder_radius
        self.cylinder_length = cylinder_length

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

        The un-normalized rewards are as follows:

        The sparse reward is 0 if the peg is outside the hole, and 1 if it's inside.
        We enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        The dense reward has four additional components.

            Reaching: in [0, 1], to encourage the arms to get together.
            Perpendicular and parallel distance: in [0,1], for the same purpose.
            Cosine of the angle: in [0, 1], to encourage having the right orientation.

        Note that the final reward is normalized and scaled by reward_scale / 5.0 as
        well so that the max score is equal to reward_scale

        """
        reward = 0

        t, d, cos = self._compute_orientation()

        # Right location and angle
        if d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95:
            reward = 1

        # use a shaping reward
        if self.reward_shaping:
            # reaching reward
            hole_pos = self.sim.data.body_xpos[self.hole_body_id]
            gripper_site_pos = self.sim.data.body_xpos[self.cyl_body_id]
            dist = np.linalg.norm(gripper_site_pos - hole_pos)
            reaching_reward = 1 - np.tanh(1.0 * dist)
            reward += reaching_reward

            # Orientation reward
            reward += 1 - np.tanh(d)
            reward += 1 - np.tanh(np.abs(t))
            reward += cos

        return reward * self.reward_scale / 5.0

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
        self.cylinder = CylinderObject(
            name="cylinder",
            size_min=(self.cylinder_radius[0], self.cylinder_length),
            size_max=(self.cylinder_radius[1], self.cylinder_length),
            add_material=True,
        )

        # Load hole object
        self.hole_obj = self.hole.get_collision(site=True)
        self.hole_obj.set("quat", "0 0 0.707 0.707")
        self.hole_obj.set("pos", "0.11 0 0.18")
        self.model.merge_asset(self.hole)

        # Load cylinder object
        self.cyl_obj = self.cylinder.get_collision(site=True)
        self.cyl_obj.set("pos", "0 0 0.15")
        self.model.merge_asset(self.cylinder)

        # Depending on env configuration, append appropriate objects to arms
        if self.env_configuration == "bimanual":
            self.model.worldbody.find(".//body[@name='{}']"
                                      .format(self.robots[0].robot_model.eef_name["left"])).append(self.hole_obj)
            self.model.worldbody.find(".//body[@name='{}']"
                                      .format(self.robots[0].robot_model.eef_name["right"])).append(self.cyl_obj)
        else:
            self.model.worldbody.find(".//body[@name='{}']"
                                      .format(self.robots[1].robot_model.eef_name)).append(self.hole_obj)
            self.model.worldbody.find(".//body[@name='{}']"
                                      .format(self.robots[0].robot_model.eef_name)).append(self.cyl_obj)

        # Color the cylinder appropriately
        self.model.worldbody.find(".//geom[@name='cylinder']").set("rgba", "0 1 0 1")

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id("hole")
        self.cyl_body_id = self.sim.model.body_name2id("cylinder")

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

            # position and rotation of cylinder and hole
            hole_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
            hole_quat = T.convert_quat(
                self.sim.data.body_xquat[self.hole_body_id], to="xyzw"
            )
            di["hole_pos"] = hole_pos
            di["hole_quat"] = hole_quat

            cyl_pos = np.array(self.sim.data.body_xpos[self.cyl_body_id])
            cyl_quat = T.convert_quat(
                self.sim.data.body_xquat[self.cyl_body_id], to="xyzw"
            )
            di["cyl_to_hole"] = cyl_pos - hole_pos
            di["cyl_quat"] = cyl_quat

            # Relative orientation parameters
            t, d, cos = self._compute_orientation()
            di["angle"] = cos
            di["t"] = t
            di["d"] = d

            di["object-state"] = np.concatenate(
                [
                    di["hole_pos"],
                    di["hole_quat"],
                    di["cyl_to_hole"],
                    di["cyl_quat"],
                    [di["angle"]],
                    [di["t"]],
                    [di["d"]],
                ]
            )

        return di

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        t, d, cos = self._compute_orientation()

        return d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95

    def _compute_orientation(self):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.
        """
        cyl_mat = self.sim.data.body_xmat[self.cyl_body_id]
        cyl_mat.shape = (3, 3)
        cyl_pos = self.sim.data.body_xpos[self.cyl_body_id]

        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)

        v = cyl_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center = hole_pos + hole_mat @ np.array([0.1, 0, 0])

        t = (center - cyl_pos) @ v / (np.linalg.norm(v) ** 2)
        d = np.linalg.norm(np.cross(v, cyl_pos - center)) / np.linalg.norm(v)

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
        """
        # World frame
        peg_pos_in_world = self.sim.data.get_body_xpos("cylinder")
        peg_rot_in_world = self.sim.data.get_body_xmat("cylinder").reshape((3, 3))
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
