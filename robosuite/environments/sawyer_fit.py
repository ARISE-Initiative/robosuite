from collections import OrderedDict
import numpy as np
from copy import deepcopy

from robosuite.utils.mjcf_utils import xml_path_completion, bounds_to_grid
import robosuite.utils.transform_utils as T
import robosuite.utils.env_utils as EU
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas import LegoArena
from robosuite.models.objects import BoxObject, WoodenPieceObject, BoundingObject, BoxPatternObject, BoundingPatternObject, CompositeBoxObject, CompositeObject
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import TableTopMergedTask, UniformRandomSampler, SequentialCompositeSampler, RoundRobinSampler
from robosuite.controllers import load_controller_config
import os


class SawyerFit(SawyerEnv):
    """
    This class corresponds to the fitting task for the Sawyer robot arm.
    """

    def __init__(
        self,
        controller_config=None,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
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
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        camera_real_depth=False,
        camera_segmentation=False,
        eval_mode=False,
        perturb_evals=False,
    ):
        """
        Args:
            controller_config (dict): If set, contains relevant controller parameters for creating a custom controller.
                Else, uses the default controller for this specific task

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

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

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.

            camera_real_depth (bool): True if convert depth to real depth in meters

            camera_segmentation (bool): True if also return semantic segmentation of the camera view
        """

        # Load the default controller if none is specified
        if controller_config is None:
            controller_path = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/default_sawyer.json')
            controller_config = load_controller_config(custom_fpath=controller_path)

        # Assert that the controller config is a dict file
        assert type(controller_config) == dict, \
            "Inputted controller config must be a dict! Instead, got type: {}".format(type(controller_config))

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer
        if placement_initializer is not None:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = self._get_default_initializer()

        super().__init__(
            controller_config=controller_config,
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            indicator_num=indicator_num,
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
            camera_real_depth=camera_real_depth,
            camera_segmentation=camera_segmentation,
            eval_mode=eval_mode,
            perturb_evals=perturb_evals,
        )

    def _get_default_initializer(self):
        initializer = SequentialCompositeSampler()
        initializer.sample_on_top(
            "hole",
            surface_name="table",
            x_range=(0.0, 0.0),
            y_range=(0.0, 0.0),
            z_rotation=0.,
            ensure_object_boundary_in_range=False,
        )
        initializer.sample_on_top(
            "block",
            surface_name="table",
            x_range=[-0.3, -0.1],
            y_range=[-0.3, -0.1],
            z_rotation=None,
            ensure_object_boundary_in_range=False,
        )
        return initializer

    def _get_placement_initializer_for_eval_mode(self):
        """
        Sets a placement initializer that is used to initialize the
        environment into a fixed set of known task instances.
        This is for reproducibility in policy evaluation.
        """

        assert(self.eval_mode)

        ordered_object_names = ["hole", "block"]
        bounds = self._grid_bounds_for_eval_mode()
        initializer = SequentialCompositeSampler(round_robin_all_pairs=True)

        for name in ordered_object_names:
            if self.perturb_evals:
                # perturbation sizes should be half the grid spacing
                perturb_sizes = [((b[1] - b[0]) / b[2]) / 2. for b in bounds[name][:3]]
            else:
                perturb_sizes = [None for b in bounds[name][:3]]

            grid = bounds_to_grid(bounds[name][:3])
            sampler = RoundRobinSampler(
                x_range=grid[0],
                y_range=grid[1],
                ensure_object_boundary_in_range=False,
                z_rotation=grid[2],
                x_perturb=perturb_sizes[0],
                y_perturb=perturb_sizes[1],
                z_rotation_perturb=perturb_sizes[2],
                z_offset=bounds[name][3],
            )
            initializer.append_sampler(name, sampler)

        self.placement_initializer = initializer
        return initializer

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """
        ret = {}

        # (low, high, number of grid points for this dimension)
        hole_x_bounds = (0., 0., 1)
        hole_y_bounds = (0., 0., 1)
        hole_z_rot_bounds = (0., 0., 1)
        hole_z_offset = 0.
        ret["hole"] = [hole_x_bounds, hole_y_bounds, hole_z_rot_bounds, hole_z_offset]

        block_x_bounds = (-0.3, -0.1, 3)
        block_y_bounds = (-0.3, -0.1, 3)
        block_z_rot_bounds = (0., 2. * np.pi, 3)
        block_z_offset = 0.
        ret["block"] = [block_x_bounds, block_y_bounds, block_z_rot_bounds, block_z_offset]

        return ret

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = LegoArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # initialize objects of interest
        # piece = WoodenPieceObject()

        # # inverse hole will be some random rotation of the wooden piece
        # self.obj_size = piece.get_bounding_box()
        # # np.random.shuffle(self.obj_size)

        TOLERANCE = 1.03
        self.obj_size = np.array([0.02, 0.04, 0.07])
        self.hole_size = TOLERANCE * self.obj_size

        piece = BoxObject(
            size=self.obj_size,
            rgba=[1, 0, 0, 1],
        )

        self.hole = BoundingObject(
            size=[0.1, 0.1, 0.1],
            hole_size=self.hole_size, 
            joint=[],
            rgba=[0, 0, 1, 1],
            hole_rgba=[0, 1, 0, 1],
        )

        self.mujoco_objects = OrderedDict([
            ("block", piece), 
            ("hole", self.hole),
        ])

        # reset initial joint positions (gets reset in sim during super() call in _reset_internal)
        # self.init_qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        self.init_qpos = np.array([0.00, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        self.init_qpos += np.random.randn(self.init_qpos.shape[0]) * 0.02

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
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
        self.object_body_ids = {}
        self.object_body_ids["hole"]  = self.sim.model.body_name2id("hole")
        self.object_body_ids["block"] = self.sim.model.body_name2id("block")

        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def reward(self, action=None):
        """
        Reward function for the task.

        The dense reward has three components.

            Reaching: in [0, 1], to encourage the arm to reach the cube
            Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            pass

        return reward

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

            # remember the keys to collect into object info
            object_state_keys = []

            # for conversion to relative gripper frame
            gripper_pose = T.pose2mat((di["eef_pos"], di["eef_quat"]))
            world_pose_in_gripper = T.pose_inv(gripper_pose)

            for k in self.object_body_ids:
                # position and rotation of the pieces
                body_id = self.object_body_ids[k]
                block_pos = np.array(self.sim.data.body_xpos[body_id])
                block_quat = T.convert_quat(
                    np.array(self.sim.data.body_xquat[body_id]), to="xyzw"
                )
                di["{}_pos".format(k)] = block_pos
                di["{}_quat".format(k)] = block_quat

                # get relative pose of object in gripper frame
                block_pose = T.pose2mat((block_pos, block_quat))
                rel_pose = T.pose_in_A_to_pose_in_B(block_pose, world_pose_in_gripper)
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                di["{}_to_eef_pos".format(k)] = rel_pos
                di["{}_to_eef_quat".format(k)] = rel_quat

                object_state_keys.append("{}_pos".format(k))
                object_state_keys.append("{}_quat".format(k))
                object_state_keys.append("{}_to_eef_pos".format(k))
                object_state_keys.append("{}_to_eef_quat".format(k))

            di["object-state"] = np.concatenate([di[k] for k in object_state_keys])

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1)
                in self.gripper.contact_geoms()
                or self.sim.model.geom_id2name(contact.geom2)
                in self.gripper.contact_geoms()
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        block_pos = np.array(self.sim.data.body_xpos[self.object_body_ids["block"]])
        hole_pos = np.array(self.sim.data.body_xpos[self.object_body_ids["hole"]])
        result = self.hole.in_box(
            position=hole_pos, 
            object_position=block_pos, 
            # object_size=self.obj_size,
        )
        # if (not self.grid.in_grid(self.sim.data.body_xpos[self.sim.model.body_name2id("block")]-[0.16 + self.table_full_size[0] / 2, 0, 0.0], self.obj_size)):
        #     result = False
        return result

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to block
        if self.gripper_visualization:
            # get distance to cube
            block_site_id = self.sim.model.site_name2id("block")
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[block_site_id]
                    - self.sim.data.get_site_xpos("grip_site")
                )
            )

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba


class SawyerFitPushLongBar(SawyerFit):
    """Pushing task with object fitting."""

    def _get_default_initializer(self):
        initializer = SequentialCompositeSampler()
        initializer.sample_on_top(
            "hole",
            surface_name="table",
            x_range=(0.0, 0.0),
            y_range=(0.0, 0.0),
            z_rotation=0.,
            ensure_object_boundary_in_range=False,
        )
        initializer.sample_on_top(
            "block",
            surface_name="hole",
            x_range=[-0.1, -0.1],
            y_range=[-0.1, -0.1],
            z_rotation=0.,
            ensure_object_boundary_in_range=False,
        )
        return initializer

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """
        ret = {}

        # (low, high, number of grid points for this dimension)
        hole_x_bounds = (0., 0., 1)
        hole_y_bounds = (0., 0., 1)
        hole_z_rot_bounds = (0., 0., 1)
        hole_z_offset = 0.
        ret["hole"] = [hole_x_bounds, hole_y_bounds, hole_z_rot_bounds, hole_z_offset]

        block_x_bounds = (-0.1, -0.1, 1)
        block_y_bounds = (-0.1, -0.1, 1)
        block_z_rot_bounds = (0., 0. * np.pi, 1)
        block_z_offset = 0.
        ret["block"] = [block_x_bounds, block_y_bounds, block_z_rot_bounds, block_z_offset]

        return ret

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SawyerEnv._load_model(self)
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = LegoArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        TOLERANCE = 1.03
        self.obj_size = np.array([0.05, 0.015, 0.015])
        self.hole_size = TOLERANCE * self.obj_size

        piece = BoxObject(
            size=self.obj_size,
            rgba=[1, 0, 0, 1],
        )

        # self.hole = BoundingObject(
        #     size=[0.4, 0.4, 0.02],
        #     hole_size=self.hole_size, 
        #     joint=[],
        #     rgba=[0, 0, 1, 1],
        #     hole_rgba=[0, 1, 0, 1],
        # )

        # hole pattern to prop bar upright
        pattern = np.ones((3, 6, 6))
        pattern[0][1:-1, 1:-1] = 0.
        pattern[1][1:-1, 1:-1] = 0.
        pattern[2] = np.zeros((6, 6))
        unit_size = [0.004, 0.004, 0.015]
        self.hole = BoundingPatternObject(
            unit_size=unit_size,
            pattern=pattern,
            size=[0.4, 0.4, 0.05],
            hole_location=[0.3, 0],
            joint=[],
            # rgba=[0, 0, 1, 1],
            rgba=[0.627, 0.627, 0.627, 1],
            hole_rgba=[0, 1, 0, 1],
            pattern_rgba=[0, 1, 1, 1],
        )

        self.mujoco_objects = OrderedDict([
            ("block", piece), 
            ("hole", self.hole),
        ])

        # reset initial joint positions (gets reset in sim during super() call in _reset_internal)
        self.init_qpos = np.array([0.00, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        self.init_qpos += np.random.randn(self.init_qpos.shape[0]) * 0.02

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _pre_action(self, action, policy_step=None):
        """
        Last gripper dimensions of action are ignored.
        """
        # close gripper
        # action[-self.gripper.dof:] = 1.
        super()._pre_action(action, policy_step=policy_step)

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        return False


class SawyerThreading(SawyerFit):
    """Threading task."""

    def _get_default_initializer(self):
        initializer = SequentialCompositeSampler()

        # NOTE: this z-offset accounts for small errors with placement (problem for slide joints)
        initializer.sample_on_top(
            "hole",
            surface_name="table",
            x_range=(-0.1, 0.15),
            y_range=(-0.15, -0.15),
            z_rotation=(-np.pi / 3., 0.),
            z_offset=0.001, 
            ensure_object_boundary_in_range=False,
        )
        initializer.sample_on_top(
            "block",
            surface_name="table",
            x_range=[-0.2, -0.2],
            y_range=[0.2, 0.2],
            z_rotation=(np.pi / 2.),
            ensure_object_boundary_in_range=False,
        )
        return initializer

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """
        ret = {}

        # (low, high, number of grid points for this dimension)
        hole_x_bounds = (-0.1, 0.15, 3)
        hole_y_bounds = (-0.15, -0.15, 1)
        hole_z_rot_bounds = (-np.pi / 3., 0., 3)
        hole_z_offset = 0.001
        ret["hole"] = [hole_x_bounds, hole_y_bounds, hole_z_rot_bounds, hole_z_offset]

        block_x_bounds = (-0.2, -0.2, 1)
        block_y_bounds = (0.2, 0.2, 1)
        block_z_rot_bounds = (np.pi / 2., np.pi / 2., 1)
        block_z_offset = 0.
        ret["block"] = [block_x_bounds, block_y_bounds, block_z_rot_bounds, block_z_offset]

        return ret

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SawyerEnv._load_model(self)
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = LegoArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        self.obj_size = np.array([0.02, 0.1, 0.02])

        # make density light so object is not too heavy to pick up
        piece = BoxObject(
            size=self.obj_size,
            rgba=[1, 0, 0, 1],
            density=0.1,
        )

        # cross-section hole
        pattern = np.zeros((3, 3, 1))
        pattern[0] = [[1], [1], [1]]
        pattern[1] = [[1], [0], [1]]
        pattern[2] = [[1], [1], [1]]

        # 2D slide and hinge joints for hole
        slide_joint1 = dict(
            pos="0 0 0",
            axis="1 0 0",
            type="slide",
            limited="false",
            damping="0.5",
        )
        slide_joint2 = dict(
            pos="0 0 0",
            axis="0 1 0",
            type="slide",
            limited="false",
            damping="0.5",
        )
        hinge_joint = dict(
            pos="0 0 0",
            axis="0 0 1",
            type="hinge",
            limited="false",
            damping="0.005",
        )
        joints = [slide_joint1, slide_joint2, hinge_joint]

        self.hole = BoxPatternObject(
            unit_size=[0.025, 0.01, 0.025],
            pattern=pattern,
            joint=joints,
            rgba=[0, 1, 0, 1],
        )
        self.hole_size = np.array(self.hole.total_size)

        self.mujoco_objects = OrderedDict([
            ("block", piece), 
            ("hole", self.hole),
        ])

        # reset initial joint positions (gets reset in sim during super() call in _reset_internal)
        self.init_qpos = np.array([0.00, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        self.init_qpos += np.random.randn(self.init_qpos.shape[0]) * 0.02

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _check_success(self):
        """
        Returns True if task has been completed.
        """

        # just check if the center of the block and the hole are close enough
        block_pos = np.array(self.sim.data.body_xpos[self.object_body_ids["block"]])
        hole_pos = np.array(self.sim.data.body_xpos[self.object_body_ids["hole"]])
        radius = self.hole_size[1]
        return (np.linalg.norm(block_pos - hole_pos) < radius)

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        pass


class SawyerThreadingPrecise(SawyerThreading):
    """Threading task."""

    def _get_default_initializer(self):
        initializer = SequentialCompositeSampler()

        # NOTE: this z-offset accounts for small errors with placement (problem for slide joints)
        initializer.sample_on_top(
            "hole",
            surface_name="table",
            x_range=(0., 0.),
            y_range=(-0.15, -0.15),
            z_rotation=(-np.pi / 6., -np.pi / 6.),
            z_offset=0.001, 
            ensure_object_boundary_in_range=False,
        )
        initializer.sample_on_top(
            "block",
            surface_name="table",
            x_range=[-0.2, -0.2],
            y_range=[0.2, 0.2],
            z_rotation=(-np.pi / 2.),
            ensure_object_boundary_in_range=False,
        )
        return initializer

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """
        ret = {}

        # (low, high, number of grid points for this dimension)
        hole_x_bounds = (0., 0., 1)
        hole_y_bounds = (-0.15, -0.15, 1)
        hole_z_rot_bounds = (-np.pi / 6., -np.pi / 6., 1)
        hole_z_offset = 0.001
        ret["hole"] = [hole_x_bounds, hole_y_bounds, hole_z_rot_bounds, hole_z_offset]

        block_x_bounds = (-0.2, -0.2, 1)
        block_y_bounds = (0.2, 0.2, 1)
        block_z_rot_bounds = (-np.pi / 2., -np.pi / 2., 1)
        block_z_offset = 0.
        ret["block"] = [block_x_bounds, block_y_bounds, block_z_rot_bounds, block_z_offset]

        return ret

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SawyerEnv._load_model(self)
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = LegoArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # make a skinny threading object with a large handle
        thread_size = [0.005, 0.06, 0.005]
        handle_size = [0.02, 0.02, 0.02]
        geom_sizes = [
            thread_size,
            handle_size,
        ]
        geom_locations = [
            # thread geom needs to be offset from boundary in (x, z)
            [(handle_size[0] - thread_size[0]), 0., (handle_size[2] - thread_size[2])],
            # handle geom needs to be offset in y
            [0., 2. * thread_size[1], 0.],
        ]
        geom_names = ["thread", "handle"]
        geom_rgbas = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ]
        # make the thread low friction to ensure easy insertion
        geom_frictions = [
            [0.3, 5e-3, 1e-4],
            None,
        ]

        piece = CompositeBoxObject(
            total_size=[0.02, 0.08, 0.02],
            geom_locations=geom_locations,
            geom_sizes=geom_sizes,
            geom_names=geom_names,
            geom_rgbas=geom_rgbas,
            geom_frictions=geom_frictions,
            rgba=None,
            # density=0.1,
        )

        # big square with small hole
        unit_size = [0.008, 0.005, 0.008]
        pattern = np.ones((9, 9, 1))
        pattern[4][4] = 0
        solref = [0.02, 1.]
        solimp = [0.9, 0.95, 0.001]
        friction = [0.3, 5e-3, 1e-4] # low friction for easy insertion

        # 2D slide and hinge joints for hole
        slide_joint1 = dict(
            pos="0 0 0",
            axis="1 0 0",
            type="slide",
            limited="false",
            damping="0.5",
        )
        slide_joint2 = dict(
            pos="0 0 0",
            axis="0 1 0",
            type="slide",
            limited="false",
            damping="0.5",
        )
        hinge_joint = dict(
            pos="0 0 0",
            axis="0 0 1",
            type="hinge",
            limited="false",
            damping="0.001",
        )
        joints = [slide_joint1, slide_joint2, hinge_joint]

        self.hole = BoxPatternObject(
            unit_size=unit_size,
            pattern=pattern,
            joint=joints,
            rgba=[1, 0, 1, 1],
            solref=solref,
            solimp=solimp,
            friction=friction,
        )
        self.hole_size = np.array(self.hole.total_size)

        self.mujoco_objects = OrderedDict([
            ("block", piece), 
            ("hole", self.hole),
        ])

        # reset initial joint positions (gets reset in sim during super() call in _reset_internal)
        self.init_qpos = np.array([0.00, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        self.init_qpos += np.random.randn(self.init_qpos.shape[0]) * 0.02

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _check_success(self):
        """
        Returns True if task has been completed.
        """

        # just check if the center of the block and the hole are close enough
        block_pos = np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id("block_thread")])
        hole_pos = np.array(self.sim.data.body_xpos[self.object_body_ids["hole"]])
        radius = self.hole_size[1]
        return (np.linalg.norm(block_pos - hole_pos) < radius)

class SawyerThreadingRing(SawyerThreadingPrecise):
    """Threading task."""
    def __init__(
        self,
        use_post=True,
        **kwargs
    ):
        self.use_post = use_post
        super().__init__(**kwargs)

    def _get_default_initializer(self):
        initializer = SequentialCompositeSampler()

        # NOTE: this z-offset accounts for small errors with placement (problem for slide joints)
        initializer.sample_on_top(
            "hole",
            surface_name="table",
            x_range=(0., 0.),
            y_range=(-0.15, -0.15),
            z_rotation=(np.pi / 3., np.pi / 3.),
            z_offset=0.001, 
            ensure_object_boundary_in_range=False,
        )
        initializer.sample_on_top(
            "block",
            surface_name="table",
            x_range=[-0.2, -0.2],
            y_range=[0.2, 0.2],
            z_rotation=(-np.pi / 2.),
            ensure_object_boundary_in_range=False,
        )
        return initializer

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """
        ret = {}

        # (low, high, number of grid points for this dimension)
        hole_x_bounds = (0., 0., 1)
        hole_y_bounds = (-0.15, -0.15, 1)
        hole_z_rot_bounds = (np.pi / 3., np.pi / 3., 1)
        hole_z_offset = 0.001
        ret["hole"] = [hole_x_bounds, hole_y_bounds, hole_z_rot_bounds, hole_z_offset]

        block_x_bounds = (-0.2, -0.2, 1)
        block_y_bounds = (0.2, 0.2, 1)
        block_z_rot_bounds = (-np.pi / 2., -np.pi / 2., 1)
        block_z_offset = 0.
        ret["block"] = [block_x_bounds, block_y_bounds, block_z_rot_bounds, block_z_offset]

        return ret

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SawyerEnv._load_model(self)
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = LegoArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # make a skinny threading object with a large handle
        thread_size = [0.005, 0.06, 0.005]
        handle_size = [0.02, 0.02, 0.02]
        geom_sizes = [
            thread_size,
            handle_size,
        ]
        geom_locations = [
            # thread geom needs to be offset from boundary in (x, z)
            [(handle_size[0] - thread_size[0]), 0., (handle_size[2] - thread_size[2])],
            # handle geom needs to be offset in y
            [0., 2. * thread_size[1], 0.],
        ]
        geom_names = ["thread", "handle"]
        geom_rgbas = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ]
        # make the thread low friction to ensure easy insertion
        geom_frictions = [
            [0.3, 5e-3, 1e-4],
            None,
        ]

        piece = CompositeBoxObject(
            total_size=[0.02, 0.08, 0.02],
            geom_locations=geom_locations,
            geom_sizes=geom_sizes,
            geom_names=geom_names,
            geom_rgbas=geom_rgbas,
            geom_frictions=geom_frictions,
            rgba=None,
            density=100,
            # density=0.1,
        )

        ### small thin ring with tripod ###
        total_size = [0.05, 0.05, 0.1]

        # first get the geoms necessary to make the thin ring at the top
        ring_color = [0, 1, 0, 1]
        unit_size = [0.005, 0.002, 0.002]
        pattern = np.ones((6, 1, 6))
        for i in range(1, 5):
            pattern[i][0][1:5] = np.zeros(4)
            
        # make ring low friction for easy insertion
        ring_friction = [0.3, 5e-3, 1e-4] 
        ring_geom_args = BoxPatternObject._geoms_from_init(None, unit_size, pattern, rgba=None, friction=ring_friction)
        self.num_ring_geoms = len(ring_geom_args["geom_locations"])
        ring_geom_args["geom_rgbas"] = [ring_color for _ in range(self.num_ring_geoms)]
        ring_geom_args["geom_types"] = ["box" for _ in range(self.num_ring_geoms)]
        ring_geom_args["geom_names"] = ["ring_{}".format(i) for i in range(self.num_ring_geoms)]
        self.ring_size = [
            unit_size[0] * pattern.shape[1], unit_size[1] * pattern.shape[2], unit_size[2] * pattern.shape[0],
        ]

        # add in an offset for where the ring is located relative to the (0, 0, 0) corner
        ring_offset = [
            total_size[0] - self.ring_size[0], 
            total_size[1] - self.ring_size[1], 
            2. * (total_size[2] - self.ring_size[2]),
        ]
        for i in range(self.num_ring_geoms):
            ring_geom_args["geom_locations"][i][0] += ring_offset[0]
            ring_geom_args["geom_locations"][i][1] += ring_offset[1]
            ring_geom_args["geom_locations"][i][2] += ring_offset[2]

        # make the capsule tripod
        num_tripod_geoms = 3
        tripod_color = [1, 0, 1, 1]
        capsule_r = 0.01
        capsule_h = 0.03
        tripod_geom_args = {
            "geom_types" : ["capsule" for _ in range(num_tripod_geoms)],
            "geom_rgbas" : [tripod_color for _ in range(num_tripod_geoms)],
            "geom_sizes" : [[capsule_r, capsule_h] for _ in range(num_tripod_geoms)],
            "geom_names" : ["tripod_{}".format(i) for i in range(num_tripod_geoms)],
        }
        tripod_geom_args["geom_locations"] = [
            [0., 0., 0.],
            [0., 2. * total_size[1] - 2. * capsule_r, 0.],
            [2. * total_size[0] - 2. * capsule_r, total_size[1] - capsule_r, 0.],
        ]
        tripod_geom_args["geom_frictions"] = [None for _ in range(num_tripod_geoms)]

        # make a mounted base and a post
        num_additional_geoms = 2
        additional_color = [1, 0, 1, 1]
        base_thickness = 0.005
        post_size = 0.005
        additional_geom_args = {
            "geom_types" : ["box" for _ in range(num_additional_geoms)],
            "geom_rgbas" : [additional_color for _ in range(num_additional_geoms)],
            "geom_names" : ["additional_{}".format(i) for i in range(num_additional_geoms)],
        }
        additional_geom_args["geom_sizes"] = [
            [total_size[0], total_size[1], base_thickness],
            [post_size, post_size, total_size[2] - self.ring_size[2] - base_thickness - capsule_r - capsule_h],
        ]
        additional_geom_args["geom_locations"] = [
            [0., 0., 2. * (capsule_r + capsule_h)],
            [total_size[0] - post_size, total_size[1] - post_size, 2. * (capsule_r + capsule_h + base_thickness)]
        ]
        additional_geom_args["geom_frictions"] = [None for _ in range(num_additional_geoms)]

        if not self.use_post:
            # remove the post
            additional_geom_args = { k : additional_geom_args[k][:1] for k in additional_geom_args }

        geom_args = { k : ring_geom_args[k] + tripod_geom_args[k] + additional_geom_args[k] for k in ring_geom_args }

        # # NOTE: this lower value of solref allows the thin hole wall to avoid penetration through it
        solref = [0.001, 1]
        solimp = [0.9, 0.95, 0.001]
        joints = None

        # small thin ring with tripod
        self.hole = CompositeObject(
            total_size=total_size,
            joint=joints,
            rgba=None,
            density=100.,
            solref=solref,
            solimp=solimp,
            **geom_args,
        )
        self.hole_size = np.array(self.hole.total_size)

        self.mujoco_objects = OrderedDict([
            ("block", piece), 
            ("hole", self.hole),
        ])

        # reset initial joint positions (gets reset in sim during super() call in _reset_internal)
        self.init_qpos = np.array([0.00, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        self.init_qpos += np.random.randn(self.init_qpos.shape[0]) * 0.02

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        block_pos = np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id("block_thread")])

        # ring position is average of all the surrounding ring geom positions
        ring_pos = np.zeros(3)
        for i in range(self.num_ring_geoms):
            ring_pos += np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id("hole_ring_{}".format(i))])
        ring_pos /= self.num_ring_geoms

        # radius should be the ring size, since we want to check that the bar is within the ring
        radius = self.ring_size[1]

        # check if the center of the block and the hole are close enough
        return (np.linalg.norm(block_pos - ring_pos) < radius)


class SawyerCircus(SawyerThreadingRing):
    """Threading task."""
    def __init__(
        self,
        **kwargs
    ):
        # assert("use_post" not in kwargs)
        # kwargs["use_post"] = False
        super().__init__(**kwargs)

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """
        ret = super()._grid_bounds_for_eval_mode()

        # (low, high, number of grid points for this dimension)
        hole_x_bounds = (0.0, 0.15, 9)
        hole_y_bounds = (-0.15, -0.15, 1)
        hole_z_rot_bounds = (np.pi / 3., np.pi / 3., 1)
        hole_z_offset = 0.001
        ret["hole"] = [hole_x_bounds, hole_y_bounds, hole_z_rot_bounds, hole_z_offset]

        return ret


class SawyerCircusTest(SawyerCircus):
    """Threading task."""
    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """
        ret = super()._grid_bounds_for_eval_mode()

        # augment old spacing by half grid width to ensure no overlap in grid points
        old_spacing = (ret["hole"][0][1] - ret["hole"][0][0]) / ret["hole"][0][2]
        offset = old_spacing / 2.
        ret["hole"][0] = (ret["hole"][0][0] + offset, ret["hole"][0][1] + offset, ret["hole"][0][2])
        return ret

