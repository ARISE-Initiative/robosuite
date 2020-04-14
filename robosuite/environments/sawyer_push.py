from collections import OrderedDict
import numpy as np

from robosuite.environments.sawyer import SawyerEnv
from robosuite.environments.sawyer_lift import SawyerLift

import robosuite.utils.env_utils as EU
import robosuite.utils.control_utils as CU
import robosuite.utils.transform_utils as TU
from robosuite.utils.mjcf_utils import bounds_to_grid

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import TableTopTask, UniformRandomSampler, RoundRobinSampler, TableTopVisualTask, \
    TableTopMergedTask, SequentialCompositeSampler


class SawyerPush(SawyerLift):
    """
    This class corresponds to the pushing task for the Sawyer robot arm.

    NOTE: The table is the same as the lifting task, with one important difference.
    We make the first friction coefficient (corresponding to translational motion)
    very small in order to enable sliding objects. In this way, the object friction
    value determines how hard it is to push objects (since MuJoCo will take a maximum
    over the two values in contact).
    """

    def __init__(
        self,
        controller_config=None,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(0.01, 5e-3, 1e-4), 
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
        target_color=(0, 1, 0, 0.3),
        hide_target=False,
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

        self._object_name = 'object'
        self._target_name = 'target'
        self._target_rgba = target_color
        self._hide_target = hide_target
        if self._hide_target:
            self._target_rgba = (0., 0., 0., 0.,)

        # object placement initializer
        if placement_initializer is None:
            placement_initializer = self._get_default_initializer()

        super().__init__(
            controller_config=controller_config,
            gripper_type=gripper_type,
            table_full_size=table_full_size,
            table_friction=table_friction,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_shaping=reward_shaping,
            placement_initializer=placement_initializer,
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
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
            eval_mode=eval_mode,
            perturb_evals=perturb_evals,
            camera_real_depth=camera_real_depth,
            camera_segmentation=camera_segmentation,
        )

    def _get_default_initializer(self):
        initializer = SequentialCompositeSampler()
        initializer.sample_on_top(
            self._object_name,
            surface_name="table",
            x_range=(-0.13, -0.13),
            y_range=(-0.05, 0.05),
            z_rotation=(0.0, 0.0),
            ensure_object_boundary_in_range=False,
        )
        initializer.sample_on_top(
            self._target_name,
            surface_name="table",
            x_range=(0.17, 0.17),
            y_range=(-0.05, 0.05),
            z_rotation=(0.0, 0.0),
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

        ordered_object_names = [self._object_name, self._target_name]
        bounds = self._grid_bounds_for_eval_mode()
        initializer = SequentialCompositeSampler(round_robin_all_pairs=True)

        for name in ordered_object_names:
            if self.perturb_evals:
                # perturbation sizes should be half the grid spacing
                perturb_sizes = [((b[1] - b[0]) / b[2]) / 2. for b in bounds[name]]
            else:
                perturb_sizes = [None for b in bounds[name]]

            grid = bounds_to_grid(bounds[name])
            sampler = RoundRobinSampler(
                x_range=grid[0],
                y_range=grid[1],
                ensure_object_boundary_in_range=False,
                z_rotation=grid[2],
                x_perturb=perturb_sizes[0],
                y_perturb=perturb_sizes[1],
                z_rotation_perturb=perturb_sizes[2],
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
        # x_bounds = (-0.16, -0.1, 3)
        x_bounds = (-0.13, -0.13, 1)
        y_bounds = (-0.05, 0.05, 3)
        z_rot_bounds = (0., 0., 1)
        ret[self._object_name] = [x_bounds, y_bounds, z_rot_bounds]

        goal_x_bounds = (0.17, 0.17, 1)
        goal_y_bounds = (-0.05, 0.05, 3)
        goal_z_rot_bounds = (0., 0., 1)
        ret[self._target_name] = [goal_x_bounds, goal_y_bounds, goal_z_rot_bounds]

        return ret

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SawyerEnv._load_model(self)

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
        obj = BoxObject(
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            friction=[0.3, 5e-3, 1e-4], # NOTE: make friction low for sliding
        )
        self.mujoco_objects = OrderedDict([(self._object_name, obj)])

        # target visual object
        target_size = np.array(self.mujoco_objects[self._object_name].size)
        target = BoxObject(
            size_min=target_size,
            size_max=target_size,
            rgba=self._target_rgba,
        )
        self.visual_objects = OrderedDict([(self._target_name, target)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            mujoco_objects=self.mujoco_objects,
            visual_objects=self.visual_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        SawyerEnv._get_reference(self)

        self.object_body_id = self.sim.model.body_name2id(self._object_name)
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

        self.target_body_id = self.sim.model.body_name2id(self._target_name)
        target_qpos = self.sim.model.get_joint_qpos_addr(self._target_name + '_0')
        target_qvel = self.sim.model.get_joint_qvel_addr(self._target_name + '_0')
        self._ref_target_pos_low, self._ref_target_pos_high = target_qpos
        self._ref_target_vel_low, self._ref_target_vel_high = target_qvel

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _pre_action(self, action, policy_step=None):
        """
        Last gripper dimensions of action are ignored.
        """

        # close gripper
        action[-self.gripper.dof:] = 1.

        super()._pre_action(action, policy_step=policy_step)

        # gravity compensation for target object
        self.sim.data.qfrc_applied[
                self._ref_target_vel_low : self._ref_target_vel_high
            ] = self.sim.data.qfrc_bias[
                self._ref_target_vel_low : self._ref_target_vel_high
            ]

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

            # reaching reward for object close to target
            object_pos = self.sim.data.body_xpos[self.object_body_id]
            dist = np.linalg.norm(object_pos[:2] - self.target_pos[:2])
            reaching_reward = 1. - np.tanh(10.0 * dist)
            reward += reaching_reward

        return reward

    def _set_target(self, pos, quat=None):
        """
        Set the target position and quaternion.
        Quaternion should be (x, y, z, w).
        """
        EU.set_body_pose(self.sim, self._target_name, pos=pos, quat=quat)
        self.sim.forward()

    @property
    def target_pos(self):
        return np.array(self.sim.data.body_xpos[self.target_body_id])

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
        di = SawyerEnv._get_observation(self)

        # add in target position
        if self.use_object_obs:
            # position and rotation of object
            object_pos = np.array(self.sim.data.body_xpos[self.object_body_id])
            object_quat = TU.convert_quat(
                np.array(self.sim.data.body_xquat[self.object_body_id]), to="xyzw"
            )
            di["object_pos"] = object_pos
            di["object_quat"] = object_quat

            gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["gripper_to_object"] = gripper_site_pos - object_pos

            di["object-state"] = np.concatenate(
                [object_pos, object_quat, di["gripper_to_object"], self.target_pos]
            )
        return di

    def _check_success(self):
        """
        Returns True if task has been completed.
        """

        # successful if object within close range of target
        object_pos = np.array(self.sim.data.body_xpos[self.object_body_id])
        target_pos = np.array(self.sim.data.body_xpos[self.target_body_id])
        pos_dist = np.abs(object_pos - target_pos)

        # axis-angle representation of delta rotation - interpret the angle as rotation distance
        object_rot = np.array(self.sim.data.body_xmat[self.object_body_id]).reshape(3, 3)
        target_rot = np.array(self.sim.data.body_xmat[self.target_body_id]).reshape(3, 3)
        _, rot_dist = TU.vec2axisangle(CU.orientation_error(object_rot, target_rot))

        # rotation angle tolerance corresponds to about 5 degrees of error
        return (pos_dist[0] <= 0.01) and (pos_dist[1] <= 0.01) and (rot_dist <= 0.08)


### Some new environments... ###

class SawyerPushPosition(SawyerPush):
    """
    Cube is initialized with a constant z-rotation of 0.
    If using OSC control, force control to be position-only.
    """
    def __init__(
        self,
        **kwargs
    ):
        # assert("placement_initializer" not in kwargs)
        # kwargs["placement_initializer"] = UniformRandomSampler(
        #     x_range=[-0.16, -0.1],
        #     y_range=[-0.03, 0.03],
        #     ensure_object_boundary_in_range=False,
        #     z_rotation=0.
        # )
        if kwargs["controller_config"]["type"] == "EE_POS_ORI":
            kwargs["controller_config"]["type"] = "EE_POS"
        super(SawyerPushPosition, self).__init__(**kwargs)

class SawyerPushPuck(SawyerPush):
    """
    Pushing with a puck object that can move in 2D on top of the table.
    """
    def _load_model(self):
        SawyerEnv._load_model(self)

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

        ### cylinder with 2 slide joints###
        slide_joint1 = dict(
            pos="0 0 0",
            axis="1 0 0",
            type="slide",
            limited="false",
        )
        slide_joint2 = dict(
            pos="0 0 0",
            axis="0 1 0",
            type="slide",
            limited="false",
        )
        joints = [slide_joint1, slide_joint2]
        obj = CylinderObject(
            size=[0.04, 0.01],
            rgba=(1, 0, 0, 1),
            friction=[0.3, 5e-3, 1e-4], # NOTE: make friction low for sliding
            joint=joints,
        )

        ### cylinder ###
        # obj = CylinderObject(
        #     size=[0.04, 0.02],
        #     rgba=(1, 0, 0, 1),
        #     friction=[0.3, 5e-3, 1e-4], # NOTE: make friction low for sliding
        #     solref=[0.001, 1], # NOTE: added to make sure puck can't sink into table much
        #     # solimp=[0.998, 0.998, 0.001], 
        # )

        self.mujoco_objects = OrderedDict([(self._object_name, obj)])

        # target visual object
        target_size = np.array(self.mujoco_objects[self._object_name].size)
        target = CylinderObject(
            size=target_size,
            rgba=self._target_rgba,
        )
        self.visual_objects = OrderedDict([(self._target_name, target)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            mujoco_objects=self.mujoco_objects,
            visual_objects=self.visual_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

class SawyerPushWideBar(SawyerPush):
    """
    Pushing with a wide bar object.
    """
    def _load_model(self):
        SawyerEnv._load_model(self)

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

        ### wide bar ###
        obj = BoxObject(
            size=[0.02, 0.1, 0.01],
            # size=[0.01, 0.1, 0.01],
            rgba=[1, 0, 0, 1],
            friction=[0.3, 5e-3, 1e-4], # NOTE: make friction low for sliding
        )

        self.mujoco_objects = OrderedDict([(self._object_name, obj)])

        # target visual object
        target_size = np.array(self.mujoco_objects[self._object_name].size)
        target = BoxObject(
            size_min=target_size,
            size_max=target_size,
            rgba=self._target_rgba,
        )
        self.visual_objects = OrderedDict([(self._target_name, target)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            mujoco_objects=self.mujoco_objects,
            visual_objects=self.visual_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

class SawyerPushLongBar(SawyerPush):
    """
    Pushing with a long bar object.
    """
    def _load_model(self):
        SawyerEnv._load_model(self)

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

        ### long bar ###
        obj = BoxObject(
            # size=[0.05, 0.01, 0.01],
            size=[0.05, 0.015, 0.01],
            rgba=[1, 0, 0, 1],
            friction=[0.3, 5e-3, 1e-4], # NOTE: make friction low for sliding
        )

        self.mujoco_objects = OrderedDict([(self._object_name, obj)])

        # target visual object
        target_size = np.array(self.mujoco_objects[self._object_name].size)
        target = BoxObject(
            size_min=target_size,
            size_max=target_size,
            rgba=self._target_rgba,
        )
        self.visual_objects = OrderedDict([(self._target_name, target)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            mujoco_objects=self.mujoco_objects,
            visual_objects=self.visual_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()


