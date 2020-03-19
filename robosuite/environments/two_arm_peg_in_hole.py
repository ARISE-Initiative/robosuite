from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.robot_env import RobotEnv
from robosuite.agents import *

from robosuite.models.objects import CylinderObject, PlateWithHoleObject
from robosuite.models.arenas import EmptyArena
from robosuite.models import MujocoWorldBase


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
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        cylinder_radius=(0.015, 0.03),
        cylinder_length=0.13,
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

            use_camera_obs (bool or list of bool): if True, every observation for a specific robot includes a rendered
            image. Should either be single bool if camera obs value is to be used for all
                robots or else it should be a list of the same length as "robots" param

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            cylinder_radius (2-tuple): low and high limits of the (uniformly sampled)
                radius of the cylinder

            cylinder_length (float): length of the cylinder

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

        # reward configuration
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

    def reward(self, action):
        """
        Reward function for the task.

        The sparse reward is 0 if the peg is outside the hole, and 1 if it's inside.
        We enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        The dense reward has four components.

            Reaching: in [0, 1], to encourage the arms to get together.
            Perpendicular and parallel distance: in [0,1], for the same purpose.
            Cosine of the angle: in [0, 1], to encourage having the right orientation.
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
                self.robots[0].robot_model.set_base_xpos([0.55, -0.60, 0])
                self.robots[0].robot_model.set_base_ori([0,0,np.pi / 2])
                self.robots[1].robot_model.set_base_xpos([0.55, 0.60, 0])
                self.robots[1].robot_model.set_base_ori([0, 0, -np.pi/2])
            else:   # "single-arm-parallel" configuration setting
                self.robots[0].robot_model.set_base_xpos([0, -0.25, 0])
                self.robots[1].robot_model.set_base_xpos([0, 0.25, 0])

        # Add arena and robot
        self.model = MujocoWorldBase()
        self.mujoco_arena = EmptyArena()
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()
        self.model.merge(self.mujoco_arena)
        for robot in self.robots:
            self.model.merge(robot.robot_model)

        # initialize objects of interest
        self.mujoco_objects = OrderedDict()
        self.hole = PlateWithHoleObject()
        self.cylinder = CylinderObject(
            size_min=(self.cylinder_radius[0], self.cylinder_length),
            size_max=(self.cylinder_radius[1], self.cylinder_length),
        )

        # Load hole object
        self.hole_obj = self.hole.get_collision(name="hole", site=True)
        self.hole_obj.set("quat", "0 0 0.707 0.707")
        self.hole_obj.set("pos", "0.11 0 0.18")
        self.model.merge_asset(self.hole)

        # Load cylinder object
        self.cyl_obj = self.cylinder.get_collision(name="cylinder", site=True)
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
