from collections import OrderedDict
import numpy as np
from copy import deepcopy

from robosuite.utils.mjcf_utils import range_to_uniform_grid
from robosuite.utils.transform_utils import convert_quat
import robosuite.utils.env_utils as EU
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.objects.interactive_objects import MomentaryButtonObject, MaintainedButtonObject
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import TableTopTask, UniformRandomSampler, RoundRobinSampler, TableTopVisualTask, \
    SequentialCompositeSampler
from robosuite.controllers import load_controller_config
import os


class SawyerLift(SawyerEnv):
    """
    This class corresponds to the lifting task for the Sawyer robot arm.
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
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                z_rotation=None,
            )

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

    def _get_placement_initializer_for_eval_mode(self):
        """
        Sets a placement initializer that is used to initialize the
        environment into a fixed set of known task instances.
        This is for reproducibility in policy evaluation.
        """

        assert(self.eval_mode)

        # set up placement grid by getting bounds per dimension and then
        # using meshgrid to get all combinations
        x_bounds, y_bounds, z_rot_bounds = self._grid_bounds_for_eval_mode()
        x_grid = range_to_uniform_grid(a=x_bounds[0], b=x_bounds[1], n=x_bounds[2])
        y_grid = range_to_uniform_grid(a=y_bounds[0], b=y_bounds[1], n=y_bounds[2])
        z_rotation = range_to_uniform_grid(a=z_rot_bounds[0], b=z_rot_bounds[1], n=z_rot_bounds[2])
        grid = np.meshgrid(x_grid, y_grid, z_rotation)
        x_grid = grid[0].ravel()
        y_grid = grid[1].ravel()
        z_rotation = grid[2].ravel()
        grid_length = x_grid.shape[0]

        round_robin_period = grid_length
        if self.perturb_evals:
            # sample 100 rounds of perturbations and then sampler will repeat
            round_robin_period *= 100

            # perturbation size should be half the grid spacing
            x_pos_perturb_size = ((x_bounds[1] - x_bounds[0]) / x_bounds[2]) / 2.
            y_pos_perturb_size = ((y_bounds[1] - y_bounds[0]) / y_bounds[2]) / 2.
            z_rot_perturb_size = ((z_rot_bounds[1] - z_rot_bounds[0]) / z_rot_bounds[2]) / 2.

        # assign grid locations for the full round robin schedule
        final_x_grid = np.zeros(round_robin_period)
        final_y_grid = np.zeros(round_robin_period)
        final_z_grid = np.zeros(round_robin_period)
        for t in range(round_robin_period):
            g_ind = t % grid_length
            x, y, z = x_grid[g_ind], y_grid[g_ind], z_rotation[g_ind]
            if self.perturb_evals:
                x += np.random.uniform(low=-x_pos_perturb_size, high=x_pos_perturb_size)
                y += np.random.uniform(low=-y_pos_perturb_size, high=y_pos_perturb_size)
                z += np.random.uniform(low=-z_rot_perturb_size, high=z_rot_perturb_size)
            final_x_grid[t], final_y_grid[t], final_z_grid[t] = x, y, z

        self.placement_initializer = RoundRobinSampler(
            x_range=final_x_grid,
            y_range=final_y_grid,
            ensure_object_boundary_in_range=False,
            z_rotation=final_z_grid
        )

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = (-0.03, 0.03, 3)
        y_bounds = (-0.03, 0.03, 3)
        # z_rot_bounds = (1., 1., 1)
        z_rot_bounds = (0., 2. * np.pi, 3)
        return x_bounds, y_bounds, z_rot_bounds

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # initialize objects of interest
        cube = BoxObject(
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
        )
        self.mujoco_objects = OrderedDict([("cube", cube)])

        # reset initial joint positions (gets reset in sim during super() call in _reset_internal)
        self.init_qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        self.init_qpos += np.random.randn(self.init_qpos.shape[0]) * 0.02

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
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
        self.cube_body_id = self.sim.model.body_name2id("cube")
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        self.cube_geom_id = self.sim.model.geom_name2id("cube")

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

            # reaching reward
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            touch_left_finger = False
            touch_right_finger = False
            for i in range(self.sim.data.ncon):
                c = self.sim.data.contact[i]
                if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.cube_geom_id:
                    touch_left_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.cube_geom_id:
                    touch_right_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            if touch_left_finger and touch_right_finger:
                reward += 0.25

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
            # position and rotation of object
            cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
            cube_quat = convert_quat(
                np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw"
            )
            di["cube_pos"] = cube_pos
            di["cube_quat"] = cube_quat

            gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["gripper_to_cube"] = gripper_site_pos - cube_pos

            di["object-state"] = np.concatenate(
                [cube_pos, cube_quat, di["gripper_to_cube"]]
            )

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
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_full_size[2]

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.04

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.gripper_visualization:
            # get distance to cube
            cube_site_id = self.sim.model.site_name2id("cube")
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[cube_site_id]
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


### Some new environments... ###

class SawyerLiftPosition(SawyerLift):
    """
    Cube is initialized with a constant z-rotation of 0.
    If using OSC control, force control to be position-only.
    """
    def __init__(
        self,
        **kwargs
    ):
        assert("placement_initializer" not in kwargs)
        kwargs["placement_initializer"] = UniformRandomSampler(
            x_range=[-0.03, 0.03],
            y_range=[-0.03, 0.03],
            ensure_object_boundary_in_range=False,
            z_rotation=0.
        )
        if kwargs["controller_config"]["type"] == "EE_POS_ORI":
            kwargs["controller_config"]["type"] = "EE_POS"
        super(SawyerLiftPosition, self).__init__(**kwargs)

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = (-0.03, 0.03, 3)
        y_bounds = (-0.03, 0.03, 3)
        z_rot_bounds = (0., 0., 1)
        return x_bounds, y_bounds, z_rot_bounds


class SawyerLiftWidePositionInit(SawyerLift):
    """
    Cube is initialized with a wider set of positions, but with
    a fixed z-rotation of 1 radian.
    """
    def __init__(
        self,
        **kwargs
    ):
        assert("placement_initializer" not in kwargs)
        kwargs["placement_initializer"] = UniformRandomSampler(
            x_range=[-0.1, 0.1],
            y_range=[-0.1, 0.1],
            ensure_object_boundary_in_range=False,
            z_rotation=True
        )
        super(SawyerLiftWidePositionInit, self).__init__(**kwargs)

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = (-0.1, 0.1, 3)
        y_bounds = (-0.1, 0.1, 3)
        z_rot_bounds = (1., 1., 1)
        return x_bounds, y_bounds, z_rot_bounds


class SawyerLiftWideInit(SawyerLift):
    """
    Cube is initialized with a wider set of positions with
    uniformly random z-rotations.
    """
    def __init__(
        self,
        **kwargs
    ):
        assert("placement_initializer" not in kwargs)
        kwargs["placement_initializer"] = UniformRandomSampler(
            x_range=[-0.1, 0.1],
            y_range=[-0.1, 0.1],
            ensure_object_boundary_in_range=False,
            z_rotation=None
        )
        super(SawyerLiftWideInit, self).__init__(**kwargs)

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = (-0.1, 0.1, 3)
        y_bounds = (-0.1, 0.1, 3)
        z_rot_bounds = (0., 2. * np.pi, 3)
        return x_bounds, y_bounds, z_rot_bounds


class SawyerLiftSmallGrid(SawyerLift):
    """
    Cube is initialized on a grid of points.
    """
    def __init__(
        self,
        **kwargs
    ):

        # set up uniform 3x3 grid
        GRID_SIZE = 0.05
        x_grid = np.linspace(-GRID_SIZE, GRID_SIZE, num=3)
        y_grid = np.linspace(-GRID_SIZE, GRID_SIZE, num=3)
        grid = np.meshgrid(x_grid, y_grid)
        self.x_grid = grid[0].ravel()
        self.y_grid = grid[1].ravel()

        # remember when we have interaction
        self._has_interaction = False

        assert("placement_initializer" not in kwargs)
        kwargs["placement_initializer"] = RoundRobinSampler(
            x_range=self.x_grid,
            y_range=self.y_grid,
            ensure_object_boundary_in_range=False,
            z_rotation=np.zeros_like(self.x_grid)
        )
        super(SawyerLiftSmallGrid, self).__init__(**kwargs)

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = (-0.075, 0.075, 3)
        y_bounds = (-0.075, 0.075, 3)
        z_rot_bounds = (0., 0., 1)
        return x_bounds, y_bounds, z_rot_bounds

    def reset(self):
        self._has_interaction = False
        return super(SawyerLiftSmallGrid, self).reset()

    def step(self, action):
        if not self._has_interaction:
            # this is the first step call of the episode
            self.placement_initializer.increment_counter()
        self._has_interaction = True
        return super(SawyerLiftSmallGrid, self).step(action)


class SawyerLiftLargeGrid(SawyerLift):
    """
    Cube is initialized on a grid of points.
    """
    def __init__(
        self,
        **kwargs
    ):

        # set up uniform 3x3 grid
        GRID_SIZE = 0.1
        x_grid = np.linspace(-GRID_SIZE, GRID_SIZE, num=3)
        y_grid = np.linspace(-GRID_SIZE, GRID_SIZE, num=3)
        grid = np.meshgrid(x_grid, y_grid)
        self.x_grid = grid[0].ravel()
        self.y_grid = grid[1].ravel()

        # remember when we have interaction
        self._has_interaction = False

        assert("placement_initializer" not in kwargs)
        kwargs["placement_initializer"] = RoundRobinSampler(
            x_range=self.x_grid,
            y_range=self.y_grid,
            ensure_object_boundary_in_range=False,
            z_rotation=np.zeros_like(self.x_grid)
        )
        super(SawyerLiftLargeGrid, self).__init__(**kwargs)

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = (-0.1, 0.1, 3)
        y_bounds = (-0.1, 0.1, 3)
        z_rot_bounds = (0., 0., 1)
        return x_bounds, y_bounds, z_rot_bounds

    def reset(self):
        self._has_interaction = False
        return super(SawyerLiftLargeGrid, self).reset()

    def step(self, action):
        if not self._has_interaction:
            # this is the first step call of the episode
            self.placement_initializer.increment_counter()
        self._has_interaction = True
        return super(SawyerLiftLargeGrid, self).step(action)


class SawyerLiftPositionTarget(SawyerLift):
    """
    Goal-driven placing environment
    """
    def __init__(
        self,
        target_color=(0, 1, 0, 0.3),
        hide_target=True,
        goal_tolerance=0.05,
        goal_radius_low=0.1,
        goal_radius_high=0.2,
        **kwargs
    ):

        self._target_rgba = target_color
        self._goal_render_segmentation = None
        self._goal_tolerance = goal_tolerance
        self._goal_radius_low = goal_radius_low
        self._goal_radius_high = goal_radius_high
        self._goal_grid = None
        self._hide_target = hide_target

        self._target_name = 'cube_target'
        self._object_name = 'cube'
        self.interactive_objects = {}

        if 'placement_initializer' not in kwargs:
            kwargs["placement_initializer"] = UniformRandomSampler(
                x_range=[-0.1, 0.1],
                y_range=[-0.1, 0.1],
                ensure_object_boundary_in_range=False,
                z_rotation=0.
            )
        if kwargs["controller_config"]["type"] == "EE_POS_ORI":
            kwargs["controller_config"]["type"] = "EE_POS"
        super().__init__(**kwargs)

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions,
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = (-0.03, 0.03, 3)
        y_bounds = (-0.03, 0.03, 3)
        z_rot_bounds = (0., 0., 1)
        return x_bounds, y_bounds, z_rot_bounds

    def _load_model(self):
        super()._load_model()

        cube = BoxObject(
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.020, 0.020, 0.020],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
        )
        self.mujoco_objects = OrderedDict([(self._object_name, cube)])

        # target visual object
        target_size = np.array(self.mujoco_objects[self._object_name].size)
        target = BoxObject(
            size_min=target_size,
            size_max=target_size,
            rgba=self._target_rgba,
        )
        self.visual_objects = OrderedDict([(self._target_name, target)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopVisualTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.visual_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()
        self.model.place_visual()

    def _get_reference(self):
        super()._get_reference()

        self.object_body_id = self.sim.model.body_name2id(self._object_name)
        self.target_body_id = self.sim.model.body_name2id(self._target_name)
        target_qpos = self.sim.model.get_joint_qpos_addr(self._target_name + '_0')
        target_qvel = self.sim.model.get_joint_qvel_addr(self._target_name + '_0')
        self._ref_target_pos_low, self._ref_target_pos_high = target_qpos
        self._ref_target_vel_low, self._ref_target_vel_high = target_qvel

    def _get_placement_initializer_for_eval_mode(self):
        super(SawyerLiftPositionTarget, self)._get_placement_initializer_for_eval_mode()

        num_circles = 3
        num_circle_angles = 9

        num_placements = num_circles * num_circle_angles
        self._goal_grid = np.zeros((num_placements, 3))  # [radius, angle, z_rotation]
        r_list = np.linspace(self._goal_radius_low, self._goal_radius_high, num=num_circles)
        a_list = np.linspace(0, np.pi * (2 - (2 / (num_circle_angles + 1))), num=num_circle_angles)
        grid = np.meshgrid(r_list, a_list)
        self._goal_grid[:, 0] = grid[0].ravel()
        self._goal_grid[:, 1] = grid[1].ravel()
        self._goal_grid = np.tile(self._goal_grid, (100, 1))

        assert(self.placement_initializer.num_grid <= self._goal_grid.shape[0])

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        self._goal_dict = None
        self._goal_render_segmentation = None
        self._has_interaction = False

        super()._reset_internal()

        ### TODO: unclear why objects are placed twice... ###
        # reset positions of objects
        self.model.place_objects()
        self.model.place_visual()

        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        # init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

        # for now, place target randomly in a radius of 0.2 around current cube pos

        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        if self._goal_grid is not None:
            radius, angle, z_rot = self._goal_grid[self.placement_initializer.counter]
        else:
            angle = np.random.uniform(0., 2. * np.pi)
            radius = np.random.uniform(self._goal_radius_low, self._goal_radius_high)
        assert(radius > self._goal_tolerance)

        target_pos = cube_pos.copy()
        target_pos[0] += radius * np.cos(angle)
        target_pos[1] += radius * np.sin(angle)
        self._set_target(pos=target_pos)

        if self._hide_target:
            self.hide_target()

    def step(self, action):
        if not self._has_interaction:
            # this is the first step call of the episode
            self.placement_initializer.increment_counter()
        self._has_interaction = True
        result = super(SawyerLiftPositionTarget, self).step(action)
        if self._hide_target:
            self.hide_target()
        return result

    def _pre_action(self, action, policy_step=None):
        result = super()._pre_action(action, policy_step=policy_step)

        # gravity compensation for target object
        self.sim.data.qfrc_applied[
                self._ref_target_vel_low : self._ref_target_vel_high
            ] = self.sim.data.qfrc_bias[
                self._ref_target_vel_low : self._ref_target_vel_high
            ]
        return result

    def _set_target(self, pos, quat=None):
        """
        Set the target position and quaternion.
        Quaternion should be (x, y, z, w).
        """
        EU.set_body_pose(self.sim, self._target_name, pos=pos, quat=quat)
        self.sim.forward()

    def hide_target(self):
        """make the target transparent for rendering"""
        self.sim.model.geom_rgba[EU.bodyid2geomids(self.sim, self.target_body_id)[0]][-1] = 0.0

    def show_target(self):
        self.sim.model.geom_rgba[EU.bodyid2geomids(self.sim, self.target_body_id)[0]][-1] = 0.3

    @property
    def target_pos(self):
        return np.array(self.sim.data.body_xpos[self.target_body_id])

    def _get_observation(self):
        """
        Add in target location into observation.
        """
        ### TODO: handle this in batchRL appropriately ###
        di = super()._get_observation()
        di["goal"] = self.target_pos
        return di

    def set_goal(self, obs):
        # do nothing
        pass

    def _set_goal_rendering(self, _):
        pass

    def _set_state_to_goal(self):
        """Set the environment to a goal state"""
        # set cube to target position
        tgt_pos = self.sim.data.body_xpos[self.target_body_id]
        tgt_quat = self.sim.data.body_xquat[self.target_body_id]
        EU.set_body_pose(self.sim, self._object_name, pos=tgt_pos, quat=tgt_quat)
        self.sim.forward()

    def _get_goal(self):
        """
        Get goal observation by moving object to the target, get obs, and move back.
        :return: observation dict with goal
        """
        # avoid generating goal obs every time
        if self._goal_dict is not None:
            return self._goal_dict

        with EU.world_saved(self.sim):
            self._set_state_to_goal()
            self._goal_dict = deepcopy(self._get_observation())

        return self._goal_dict

    def render_with_visual_target(self, width, height, camera_name):
        self.show_target()
        im = self.sim.render(height=height, width=width, camera_name=camera_name)[::-1].copy()
        self.hide_target()
        return im

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        return np.linalg.norm(cube_pos - self.target_pos) < self._goal_tolerance


class SawyerPositionTargetPress(SawyerLiftPositionTarget):
    """
    Goal-driven placing environment
    """
    def __init__(self, **kwargs):
        super(SawyerPositionTargetPress, self).__init__(**kwargs)

    def _load_model(self):
        super()._load_model()

        slide_joint = dict(
            pos="0 0 0",
            axis="0 0 1",
            type="slide",
            springref="1",
            limited="true",
            stiffness="0.5",
            range="-0.1 0",
            damping="1"
        )
        button = CylinderObject(rgba=(1, 0, 0, 1), size=[0.03, 0.01], joint=[slide_joint])

        self.mujoco_objects["button"] = button
        self.interactive_objects = OrderedDict([("button", None)])

        sampler = SequentialCompositeSampler()
        self.placement_initializer.setup(
            OrderedDict([("cube", self.mujoco_objects["cube"])]),
            table_size=self.mujoco_arena.table_full_size,
            table_top_offset=self.mujoco_arena.table_top_abs
        )
        sampler.append_sampler('cube', self.placement_initializer)
        button_sampler = UniformRandomSampler(
            x_range=[-0.1, 0.1],
            y_range=[-0.1, 0.1],
            ensure_object_boundary_in_range=False,
            z_rotation=0.
        )
        button_sampler.setup(
            mujoco_objects=OrderedDict([("button", button)]),
            table_size=self.mujoco_arena.table_full_size,
            table_top_offset=self.mujoco_arena.table_top_abs
        )
        sampler.append_sampler('button', button_sampler)

        # task includes arena, robot, and objects of interest
        self.model = TableTopVisualTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            visual_objects=self.visual_objects,
            initializer=sampler,
        )
        self.model.place_objects()
        self.model.place_visual()

    def _get_reference(self):
        super()._get_reference()

        self.object_body_id = self.sim.model.body_name2id("cube")
        button = MaintainedButtonObject(self.sim, body_id=self.sim.model.body_name2id("button"), on_rgba=(1, 0, 0, 1))

        def color_trigger(sim, body_id, color):
            for gid in EU.bodyid2geomids(sim, body_id):
                self.sim.model.geom_rgba[gid] = color

        button.add_on_state_funcs(
            cond={button.state_name: True},
            func=lambda: color_trigger(sim=self.sim, body_id=self.object_body_id, color=(0, 1, 0, 1))
        )

        button.add_on_state_funcs(
            cond={button.state_name: False},
            func=lambda: color_trigger(sim=self.sim, body_id=self.object_body_id, color=(1, 0, 0, 1))
        )

        self.interactive_objects['button'] = button

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        ret = super()._reset_internal()
        for _, o in self.interactive_objects.items():
            o.reset()
        return ret

    def step(self, action):
        for _, o in self.interactive_objects.items():
            o.step(sim_step=self.timestep)

        return super().step(action)

    def _set_state_to_goal(self):
        ret = super()._set_state_to_goal()
        self.interactive_objects['button'].activate()
        return ret

    def _get_goal(self):
        """
        Get goal observation by moving object to the target, get obs, and move back.
        :return: observation dict with goal
        """
        # avoid generating goal obs every time
        if self._goal_dict is not None:
            return self._goal_dict

        with EU.world_saved(self.sim):
            self._set_state_to_goal()
            self._goal_dict = deepcopy(self._get_observation())

        for k, v in self.interactive_objects.items():
            v.reset()

        return self._goal_dict

    def _check_success(self):
        loc_success = super()._check_success()
        state_name = self.interactive_objects['button'].state_name
        but_success = self.interactive_objects['button'].satisfies({state_name: True})
        return loc_success and but_success


class SawyerPositionPress(SawyerPositionTargetPress):
    def _set_state_to_goal(self):
        self.interactive_objects['button'].activate()

    def _check_success(self):
        state_name = self.interactive_objects['button'].state_name
        return self.interactive_objects['button'].satisfies({state_name: True})


class SawyerPositionTarget(SawyerPositionTargetPress):
    def _set_state_to_goal(self):
        SawyerLiftPositionTarget._set_state_to_goal(self)

    def _check_success(self):
        return SawyerLiftPositionTarget._check_success(self)
