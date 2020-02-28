from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import TableTopTask, UniformRandomSampler
import hjson
import os


class SawyerLift(SawyerEnv):
    """
    This class corresponds to the lifting task for the Sawyer robot arm.
    """

    def __init__(
        self,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        gripper_visualization=False,
        use_indicator_object=False,
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
        use_default_task_config=True,
        task_config_file=None,
        use_default_controller_config=True,
        controller_config_file=None,
        controller='joint_velocity',
        **kwargs
    ):
        """
        Args:

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

            use_default_task_config (bool): True if using default configuration file
                for remaining environment parameters. Default is true

            task_config_file (str): filepath to configuration file to be
                used for remaining environment parameters (taken relative to head of robosuite repo).

            use_default_controller_config (bool): True if using default configuration file
                for remaining environment parameters. Default is true

            controller_config_file (str): filepath to configuration file to be
                used for remaining environment parameters (taken relative to head of robosuite repo).

            controller (str): Can be 'position', 'position_orientation', 'joint_velocity', 'joint_impedance', or
                'joint_torque'. Specifies the type of controller to be used for dynamic trajectories

            controller_config_file (str): filepath to the corresponding controller config file that contains the
                associated controller parameters

            #########
            **kwargs includes additional params that may be specified and will override values found in
            the configuration files
        """

        # Load the parameter configuration files
        if use_default_controller_config == True:
            controller_filepath = os.path.join(os.path.dirname(__file__), '..',
                                               'scripts/config/controller_config.hjson')
        else:
            controller_filepath = controller_config_file

        if use_default_task_config == True:
            task_filepath = os.path.join(os.path.dirname(__file__), '..',
                                         'scripts/config/Lift_task_config.hjson')
        else:
            task_filepath = task_config_file

        try:
            with open(task_filepath) as f:
                task = hjson.load(f)
                # Load additional arguments from kwargs and override the prior config-file loaded ones
                for key, value in kwargs.items():
                    if key in task:
                        task[key] = value
        except FileNotFoundError:
            print("Env Config file '{}' not found. Please check filepath and try again.".format(task_filepath))

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                z_rotation=True,
            )

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
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
            controller_config_file=controller_filepath,
            controller=controller,
            **kwargs
        )

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
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # initialize objects of interest
        cube = BoxObject(
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
        )
        self.mujoco_objects = OrderedDict([("cube", cube)])

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

        # reset positions of objects
        self.model.place_objects()

        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

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
        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

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
        if kwargs["controller"] == "position_orientation":
            kwargs["controller"] = "position"
        super(SawyerLiftPosition, self).__init__(**kwargs)

class SawyerLiftRotation(SawyerLift):
    """
    Cube is initialized with uniform z-rotations instead of fixed z-rotation
    of 1 radian.
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
            z_rotation=None
        )
        super(SawyerLiftRotation, self).__init__(**kwargs)


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

class RoundRobinSampler(UniformRandomSampler):
    """Places all objects according to grid and round robin between grid points."""

    def __init__(
        self,
        x_range=None,
        y_range=None,
        ensure_object_boundary_in_range=True,
        z_rotation=None,
    ):
        # x_range, y_range, and z_rotation should all be lists of values to rotate between
        assert(len(x_range) == len(y_range))
        assert(len(z_rotation) == len(y_range))
        self._counter = 0
        self.num_grid = len(x_range)

        super(RoundRobinSampler, self).__init__(
            x_range=x_range, 
            y_range=y_range, 
            ensure_object_boundary_in_range=ensure_object_boundary_in_range, 
            z_rotation=z_rotation,
        )

    def increment_counter(self):
        """
        Useful for moving on to next placement in the grid.
        """
        self._counter = (self._counter + 1) % self.num_grid

    def decrement_counter(self):
        """
        Useful to reverting to the last placement in the grid.
        """
        self._counter -= 1
        if self._counter < 0:
            self._counter = self.num_grid - 1

    def sample_x(self, object_horizontal_radius):
        minimum = self.x_range[self._counter]
        maximum = self.x_range[self._counter]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_y(self, object_horizontal_radius):
        minimum = self.y_range[self._counter]
        maximum = self.y_range[self._counter]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_quat(self):
        rot_angle = self.z_rotation[self._counter]
        return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]


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

    def reset(self):
        self._has_interaction = False
        return super(SawyerLiftSmallGrid, self).reset()

    def step(self, action):
        if not self._has_interaction:
            # this is the first step call of the episode
            self.placement_initializer.increment_counter()
        self._has_interaction = True
        return super(SawyerLiftSmallGrid, self).step(action)

class SawyerLiftSmallGridFull(SawyerLift):
    """
    Cube is initialized in a box.
    """
    def __init__(
        self,
        **kwargs
    ):

        # remember when we have interaction
        self._has_interaction = False


        GRID_SIZE = 0.05
        assert("placement_initializer" not in kwargs)

        self.reproducible = kwargs.get("reproducible", False)
        if self.reproducible:

            # probably don't need more than 1000 points to cover the grid finely enough
            NUM_EVALS = 1000
            SEED = 0

            np.random.seed(SEED)
            self.x_grid = np.random.uniform(low=-GRID_SIZE, high=GRID_SIZE, size=NUM_EVALS)
            self.y_grid = np.random.uniform(low=-GRID_SIZE, high=GRID_SIZE, size=NUM_EVALS)
            kwargs["placement_initializer"] = RoundRobinSampler(
                x_range=self.x_grid,
                y_range=self.y_grid,
                ensure_object_boundary_in_range=False,
                z_rotation=np.zeros_like(self.x_grid)
            )
        else:
            kwargs["placement_initializer"] = UniformRandomSampler(
                x_range=[-GRID_SIZE, GRID_SIZE],
                y_range=[-GRID_SIZE, GRID_SIZE],
                ensure_object_boundary_in_range=False,
                z_rotation=0.
            )
        super(SawyerLiftSmallGridFull, self).__init__(**kwargs)

    def reset(self):
        self._has_interaction = False
        return super(SawyerLiftSmallGridFull, self).reset()

    def step(self, action):
        if not self._has_interaction and self.reproducible:
            # this is the first step call of the episode
            self.placement_initializer.increment_counter()
        self._has_interaction = True
        return super(SawyerLiftSmallGridFull, self).step(action)

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

    def reset(self):
        self._has_interaction = False
        return super(SawyerLiftLargeGrid, self).reset()

    def step(self, action):
        if not self._has_interaction:
            # this is the first step call of the episode
            self.placement_initializer.increment_counter()
        self._has_interaction = True
        return super(SawyerLiftLargeGrid, self).step(action)


class SawyerLiftLargeGridFull(SawyerLift):
    """
    Cube is initialized in a box.
    """
    def __init__(
        self,
        **kwargs
    ):

        # remember when we have interaction
        self._has_interaction = False


        GRID_SIZE = 0.1
        assert("placement_initializer" not in kwargs)

        self.reproducible = kwargs.get("reproducible", False)
        if self.reproducible:

            # probably don't need more than 1000 points to cover the grid finely enough
            NUM_EVALS = 1000
            SEED = 0

            np.random.seed(SEED)
            self.x_grid = np.random.uniform(low=-GRID_SIZE, high=GRID_SIZE, size=NUM_EVALS)
            self.y_grid = np.random.uniform(low=-GRID_SIZE, high=GRID_SIZE, size=NUM_EVALS)
            self.z_rotation_grid = np.random.uniform(low=0., high=(2. * np.pi), size=NUM_EVALS)
            kwargs["placement_initializer"] = RoundRobinSampler(
                x_range=self.x_grid,
                y_range=self.y_grid,
                ensure_object_boundary_in_range=False,
                z_rotation=self.z_rotation_grid
            )
        else:
            kwargs["placement_initializer"] = UniformRandomSampler(
                x_range=[-GRID_SIZE, GRID_SIZE],
                y_range=[-GRID_SIZE, GRID_SIZE],
                ensure_object_boundary_in_range=False,
                z_rotation=None
            )
        super(SawyerLiftLargeGridFull, self).__init__(**kwargs)

    def reset(self):
        self._has_interaction = False
        return super(SawyerLiftLargeGridFull, self).reset()

    def step(self, action):
        if not self._has_interaction and self.reproducible:
            # this is the first step call of the episode
            self.placement_initializer.increment_counter()
        self._has_interaction = True
        return super(SawyerLiftLargeGridFull, self).step(action)

