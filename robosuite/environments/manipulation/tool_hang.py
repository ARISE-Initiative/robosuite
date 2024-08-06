from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_robot_env import SingleRobotEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import HookFrame, RatchetingWrenchObject, StandWithMount
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.sim_utils import check_contact


class ToolHang(SingleRobotEnv):
    """
    This class corresponds to the tool hang task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
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

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

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

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model

        Made some aspects easier than the real world task:
            - increase base thickness for stand
            - increase mount width to 1.2 cm
            - add hole visualization
            - reduce hook height on stand a little
            - reduce tool ends height a little
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.4837275266036987, 0.2505579098815722, 1.2639379055124524],
            quat=[0.39713290333747864, 0.27807527780532837, 0.5016612410545349, 0.7164464592933655],
        )

        # Add sideview
        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.4837275266036987, 0.2505579098815722, 1.2139379055124524],
            quat=[0.39713290333747864, 0.27807527780532837, 0.5016612410545349, 0.7164464592933655],
        )

        # Create stand, frame, and tool
        self.stand_args = dict(
            name="stand",
            size=(
                (12.0 / 100.0),
                (14.0 / 100.0),
                (16.0 / 100.0),
            ),  # 14 cm x 12 cm base, with 16 cm height (in real world we cut the 32 cm height stand in half as well)
            mount_location=(0.0, (4.5 / 100.0)),  # 2.5 cm from right edge, so 4.5 cm to the right
            mount_width=(1.2 / 100.0),  # 1.2 cm thickness for rod cavity
            wall_thickness=(0.1 / 100.0),  # about 0.1-0.2 cm thickness for walls
            base_thickness=(1 / 100.0),  # increased thickness to 1 cm (different from real)
            initialize_on_side=False,
            add_hole_vis=True,
            density=50000.0,
            solref=(0.02, 1.0),
            solimp=(0.998, 0.998, 0.001),
        )
        self.stand = StandWithMount(**self.stand_args)

        self.frame_args = dict(
            name="frame",
            frame_length=(9.5 / 100.0),  # 9.5 cm wide
            frame_height=(18.0 / 100.0),  # 18 cm tall (in real world we cut the physical 36 cm rod in half as well)
            frame_thickness=(0.75 / 100.0),  # 0.75 cm thick
            hook_height=(1.2 / 100.0),  # lowered to 1.2 cm tall (instead of 1.7 cm in real world)
            grip_location=((9.0 - 3.0) / 100.0)
            - (0.75 / 200.0),  # move up by half height of frame minus half height of grip minus half thickness
            grip_size=((2.54 / 200.0), (6.35 / 200.0)),  # 6.35 cm length, 2.54 cm thick
            tip_size=(
                (2.54 / 200.0),
                (0.2 / 200.0),
                (0.65 / 200.0),
                (1.905 / 100.0),
            ),  # 1-inch cylinder, 0.65 inch solder tip
            density=500.0,
            solref=(0.02, 1.0),
            solimp=(0.998, 0.998, 0.001),
        )
        self.frame = HookFrame(**self.frame_args)

        self.real_tool_args = dict(
            name="tool",
            handle_size=(
                (16.5 / 200.0),
                (1.75 / 200.0),
                (0.32 / 200.0),
            ),  # 16.5 cm length, 1.75 cm width, 0.32 cm thick (1.5 cm with foam)
            outer_radius_1=(3.5 / 200.0),  # larger hole 3.5 cm outer diameter
            inner_radius_1=(2.1 / 200.0),  # reduced larger hole 2.1 cm inner diameter (from real world 2.3 cm)
            height_1=(0.7 / 200.0),  # 0.7 cm height
            outer_radius_2=(3.0 / 200.0),  # smaller hole 3 cm outer diameter
            inner_radius_2=(2.0 / 200.0),  # smaller hole 2 cm outer diameter
            height_2=(0.7 / 200.0),  # 0.7 cm height
            ngeoms=8,
            grip_size=((3 / 200.0), (8.0 / 200.0)),  # 8 cm length, 3 cm thick
            density=2000.0,
            solref=(0.02, 1.0),
            solimp=(0.998, 0.998, 0.001),
            friction=(0.95, 0.3, 0.1),
        )

        self.tool_args = self.real_tool_args
        self.tool = RatchetingWrenchObject(**self.tool_args)

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.stand, self.frame, self.tool],
        )

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # Pre-define settings for each object's placement
        objects = [self.stand, self.frame, self.tool]
        x_centers = [-self.table_full_size[0] * 0.1, -self.table_full_size[0] * 0.05, self.table_full_size[0] * 0.05]
        y_centers = [0.0, -self.table_full_size[1] * 0.3, -self.table_full_size[1] * 0.25]
        x_tols = [0.0, 0.02, 0.02]
        y_tols = [0.0, 0.02, 0.02]
        rot_centers = [0, (-np.pi / 2) + (np.pi / 6), (-np.pi / 2) - (np.pi / 9.0)]
        rot_tols = [0.0, np.pi / 18, np.pi / 18.0]
        rot_axes = ["z", "y", "z"]
        z_offsets = [
            0.001,
            (self.frame_args["frame_thickness"] - self.frame_args["frame_height"]) / 2.0
            + 0.001
            + (self.stand_args["base_thickness"] / 2.0)
            + (self.frame_args["grip_size"][1]),
            0.001,
        ]
        if ("tip_size" in self.frame_args) and (self.frame_args["tip_size"] is not None):
            z_offsets[1] -= self.frame_args["tip_size"][0] + 2.0 * self.frame_args["tip_size"][3]
        for obj, x, y, x_tol, y_tol, r, r_tol, r_axis, z_offset in zip(
            objects, x_centers, y_centers, x_tols, y_tols, rot_centers, rot_tols, rot_axes, z_offsets
        ):
            # Create sampler for this object and add it to the sequential sampler
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{obj.name}ObjectSampler",
                    mujoco_objects=obj,
                    x_range=[x - x_tol, x + x_tol],
                    y_range=[y - y_tol, y + y_tol],
                    rotation=[r - r_tol, r + r_tol],
                    rotation_axis=r_axis,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.table_offset,
                    z_offset=z_offset,
                )
            )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = dict(
            stand=self.sim.model.body_name2id(self.stand.root_body),
            frame=self.sim.model.body_name2id(self.frame.root_body),
            tool=self.sim.model.body_name2id(self.tool.root_body),
        )

        # Important sites:
        #   tool_hole1_center - for checking hanging
        #   frame_hang_site, frame_mount_site, frame_intersection_site - for orienting the hook, and checking hanging
        #   stand_mount_site - for checking that stand base is upright
        self.obj_site_id = dict(
            tool_hole1_center=self.sim.model.site_name2id("tool_hole1_center"),  # center of one end of wrench
            # tool_hole2_center=self.sim.model.site_name2id("tool_hole2_center"), # center of other end of wrench
            frame_hang_site=self.sim.model.site_name2id("frame_hang_site"),  # end of frame where hanging takes place
            frame_mount_site=self.sim.model.site_name2id(
                "frame_mount_site"
            ),  # bottom of frame that needs to be inserted into base
            frame_intersection_site=self.sim.model.site_name2id("frame_intersection_site"),  # corner of frame
            stand_mount_site=self.sim.model.site_name2id(
                "stand_mount_site"
            ),  # where frame needs to be inserted into stand
        )
        if ("tip_size" in self.frame_args) and (self.frame_args["tip_size"] is not None):
            self.obj_site_id["frame_tip_site"] = self.sim.model.site_name2id("frame_tip_site")  # tip site for insertion

        # Important geoms:
        #   stand_base - for checking that stand base is upright
        #   stand wall geoms - for checking rod insertion into stand
        #   tool hole geoms - for checking insertion
        self.obj_geom_id = dict(
            stand_base=self.sim.model.geom_name2id("stand_base"),  # bottom of stand
        )
        for i in range(4):
            self.obj_geom_id["stand_wall_{}".format(i)] = self.sim.model.geom_name2id("stand_wall{}".format(i))
        for i in range(self.tool_args["ngeoms"]):
            self.obj_geom_id["tool_hole1_hc_{}".format(i)] = self.sim.model.geom_name2id("tool_hole1_hc_{}".format(i))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # for conversion to relative gripper frame for each gripper
            def get_world_pose_grippers(modality, arm_prefix, name):
                @sensor(modality=modality)
                def fn(obs_cache):
                    return (
                        T.pose_inv(T.pose2mat((obs_cache[f"{arm_prefix}eef_pos"], obs_cache[f"{arm_prefix}eef_quat"])))
                        if f"{arm_prefix}eef_pos" in obs_cache and f"{arm_prefix}eef_quat" in obs_cache
                        else np.eye(4)
                    )

                fn.__name__ = name
                return fn

            arm_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors = [
                get_world_pose_grippers(modality, arm_prefix=arm_prefix, name=f"world_pose_in_{arm_prefix}_gripper")
                for arm_prefix in arm_prefixes
            ]
            names = [fn.__name__ for fn in sensors]
            actives = [False] * len(sensors)

            # Add absolute and relative pose for each object
            obj_names = ["base", "frame", "tool"]
            query_names = ["stand_base", "frame_intersection_site", "tool"]
            query_types = ["geom", "site", "body"]
            for i in range(len(obj_names)):
                obj_sensors, obj_sensor_names = self._create_obj_sensors(
                    obj_name=obj_names[i], modality=modality, query_name=query_names[i], query_type=query_types[i]
                )
                sensors += obj_sensors
                names += obj_sensor_names
                actives += [True] * len(obj_sensors)

            # Key boolean checks
            @sensor(modality=modality)
            def frame_is_assembled(obs_cache):
                return [float(self._check_frame_assembled())]

            @sensor(modality=modality)
            def tool_on_frame(obs_cache):
                return [float(self._check_tool_on_frame())]

            sensors += [frame_is_assembled, tool_on_frame]
            names += [frame_is_assembled.__name__, tool_on_frame.__name__]
            actives += [True, True]

            # Create observables
            for name, s, active in zip(names, sensors, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=active,
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object", query_name=None, query_type="body"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for (used for naming observations)
            modality (str): Modality to assign to all sensors
            query_name (str): Name to query mujoco for the pose attributes of this object - if None, use @obj_name
            query_type (str): Either "body", "geom", or "site" - type of mujoco sensor that will be queried for pose

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        if query_name is None:
            query_name = obj_name

        assert query_type in ["body", "geom", "site"]
        if query_type == "body":
            id_lookup = self.obj_body_id
            pos_lookup = self.sim.data.body_xpos
            mat_lookup = self.sim.data.body_xmat
        elif query_type == "geom":
            id_lookup = self.obj_geom_id
            pos_lookup = self.sim.data.geom_xpos
            mat_lookup = self.sim.data.geom_xmat
        else:
            id_lookup = self.obj_site_id
            pos_lookup = self.sim.data.site_xpos
            mat_lookup = self.sim.data.site_xmat

        ### TODO: this was slightly modified from pick-place - do we want to move this into utils to share it? ###
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(pos_lookup[id_lookup[query_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.mat2quat(np.array(mat_lookup[id_lookup[query_name]]).reshape(3, 3))

        def get_obj_to_eefs_pos(arm_prefix, name):
            @sensor(modality=modality)
            def fn(obs_cache):
                # Immediately return default value if cache is empty
                if any(
                    [
                        name not in obs_cache
                        for name in [f"{obj_name}_pos", f"{obj_name}_quat", f"world_pose_in_{arm_prefix}_gripper"]
                    ]
                ):
                    return np.zeros(3)
                obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
                rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache[f"world_pose_in_{arm_prefix}_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"{obj_name}_to_{arm_prefix}eef_quat"] = rel_quat
                return rel_pos

            fn.__name__ = name
            return fn

        def get_objs_to_eefs_quat(arm_prefix, name):
            @sensor(modality=modality)
            def fn(obs_cache):
                return (
                    obs_cache[f"{obj_name}_to_{arm_prefix}eef_quat"]
                    if f"{obj_name}_to_{arm_prefix}eef_quat" in obs_cache
                    else np.zeros(4)
                )

            fn.__name__ = name
            return fn

        arm_prefixes = self._get_arm_prefixes(self.robots[0])
        sensors = [get_obj_to_eefs_pos(arm_prefix, f"{obj_name}_to_{arm_prefix}eef_pos") for arm_prefix in arm_prefixes]
        sensors += [
            get_objs_to_eefs_quat(arm_prefix, f"{obj_name}_to_{arm_prefix}eef_quat") for arm_prefix in arm_prefixes
        ]
        names = [fn.__name__ for fn in sensors]
        sensors += [obj_pos, obj_quat]
        names += [f"{obj_name}_pos", f"{obj_name}_quat"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.tool)

    def _check_success(self):
        """
        Check if tool is hung on frame correctly and frame is assembled coorectly as well.

        Returns:
            bool: True if tool is hung on frame correctly
        """
        return self._check_frame_assembled() and self._check_tool_on_frame()

    def _check_frame_assembled(self):
        """
        Check if the frame has been assembled correctly. This checks the following things:
            (1) the base is upright
            (2) the end of the hook frame is close enough to the base
            (3) the hook frame is between the walls of the base
        """

        # position of base
        base_pos = self.sim.data.geom_xpos[self.obj_geom_id["stand_base"]]

        # check (1): the base is upright. Just take the vector between two locations on the base shaft, and check
        #            that the angle to the z-axis is small, by computing the angle between that unit vector and
        #            the z-axis. Recall that for two unit vectors, the arccosine of the dot product gives the angle.
        vec_along_base_shaft = self.sim.data.site_xpos[self.obj_site_id["stand_mount_site"]] - base_pos
        vec_along_base_shaft = vec_along_base_shaft / np.linalg.norm(vec_along_base_shaft)
        angle_to_z_axis = np.abs(np.arccos(vec_along_base_shaft[2]))
        base_shaft_is_vertical = angle_to_z_axis < np.pi / 18.0  # less than 10 degrees

        # check (2): the end of the hook frame is close enough to the base. Just check the distance
        if "frame_tip_site" in self.obj_site_id:
            bottom_hook_pos = self.sim.data.site_xpos[self.obj_site_id["frame_tip_site"]]
        else:
            bottom_hook_pos = self.sim.data.site_xpos[self.obj_site_id["frame_mount_site"]]
        insertion_dist = np.linalg.norm(bottom_hook_pos - base_pos)
        # insertion_tolerance = (self.frame_args["frame_thickness"] / 2.)
        insertion_tolerance = 0.05  # NOTE: this was manually tuned
        bottom_is_close_enough = insertion_dist < insertion_tolerance

        # check (3): the hook frame is in between the walls of the base. Take the geom positions of opposing base walls
        #            and check that they are on opposite sides of the line defined by the hook frame.

        # normalized vector that points along the frame hook
        hook_endpoint = self.sim.data.site_xpos[self.obj_site_id["frame_mount_site"]]
        frame_hook_vec = self.sim.data.site_xpos[self.obj_site_id["frame_intersection_site"]] - hook_endpoint
        frame_hook_length = np.linalg.norm(frame_hook_vec)
        frame_hook_vec = frame_hook_vec / frame_hook_length

        # geom wall position vectors relative to base position
        geom_positions = [
            self.sim.data.geom_xpos[self.obj_geom_id["stand_wall_{}".format(i)]] - hook_endpoint for i in range(4)
        ]

        # take cross product of each point against the line, and then dot the result to see if
        # the sign is positive or negative. If it is positive, then they are on the same side
        # (visualize with right-hand-rule to see this)
        rod_is_between_stand_walls = all(
            [
                np.dot(np.cross(geom_positions[0], frame_hook_vec), np.cross(geom_positions[2], frame_hook_vec)) < 0,
                np.dot(np.cross(geom_positions[1], frame_hook_vec), np.cross(geom_positions[3], frame_hook_vec)) < 0,
            ]
        )

        return base_shaft_is_vertical and (bottom_is_close_enough and rod_is_between_stand_walls)

    def _check_tool_on_frame(self):
        """
        Check if the tool has been hung on the frame correctly. This checks the following things:
            (1) the robot is not touching the tool (it is hanging on its own)
            (2) the tool hole is making contact with the frame hook
            (3) the tool hole is close to the line defined by the frame hook
            (4) either end of the tool hole are on opposite sides of the frame hook
            (5) the tool hole is inserted far enough into the frame hook
        """

        # check (1): robot is not touching the tool
        robot_grasp_geoms = [
            self.robots[0].gripper[arm].important_geoms["left_fingerpad"] for arm in self.robots[0].gripper
        ]

        robot_grasp_geoms += [
            self.robots[0].gripper[arm].important_geoms["right_fingerpad"] for arm in self.robots[0].gripper
        ]

        robot_and_tool_contact = False
        for g_group in robot_grasp_geoms:
            if check_contact(self.sim, g_group, self.tool.contact_geoms):
                robot_and_tool_contact = True
                break

        # check (2): the tool hole is making contact with the frame hook
        all_tool_hole_geoms = ["tool_hole1_hc_{}".format(i) for i in range(self.tool_args["ngeoms"])]
        frame_hook_geom = "frame_horizontal_frame"
        frame_and_tool_hole_contact = check_contact(self.sim, all_tool_hole_geoms, frame_hook_geom)

        # check (3): compute distance from tool hole center to the line defined by the frame hook

        # normalized vector that points along the frame hook
        hook_endpoint = self.sim.data.site_xpos[self.obj_site_id["frame_hang_site"]]
        frame_hook_vec = self.sim.data.site_xpos[self.obj_site_id["frame_intersection_site"]] - hook_endpoint
        frame_hook_length = np.linalg.norm(frame_hook_vec)
        frame_hook_vec = frame_hook_vec / frame_hook_length

        # compute orthogonal projection of tool hole point to get distance to frame hook line
        # (see https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation)
        tool_hole_center = self.sim.data.site_xpos[self.obj_site_id["tool_hole1_center"]]
        tool_hole_vec = tool_hole_center - hook_endpoint
        tool_hole_dot = np.dot(tool_hole_vec, frame_hook_vec)
        tool_hole_proj = tool_hole_dot * frame_hook_vec
        tool_hole_ortho_proj = tool_hole_vec - tool_hole_proj
        dist_to_frame_hook_line = np.linalg.norm(tool_hole_ortho_proj)

        # distance needs to be less than the difference between the inner tool hole radius and the half-length of the frame hook box geom
        tool_hole_is_close_enough = dist_to_frame_hook_line < (
            self.tool_args["inner_radius_1"] - (self.frame_args["frame_thickness"] / 2.0)
        )

        # check (4): take two opposite geoms around the tool hole, and check that they are on opposite sides of the frame hook line
        #            to guarantee that insertion has taken place
        g2_id = self.tool_args["ngeoms"] // 2  # get geom opposite geom 0
        g1_pos = self.sim.data.geom_xpos[self.obj_geom_id["tool_hole1_hc_0"]]
        g2_pos = self.sim.data.geom_xpos[self.obj_geom_id["tool_hole1_hc_{}".format(g2_id)]]

        # take cross product of each point against the line, and then dot the result to see if
        # the sign is positive or negative. If it is positive, then they are on the same side
        # (visualize with right-hand-rule to see this)
        g1_vec = g1_pos - hook_endpoint
        g2_vec = g2_pos - hook_endpoint
        tool_is_between_hook = np.dot(np.cross(g1_vec, frame_hook_vec), np.cross(g2_vec, frame_hook_vec)) < 0

        # check (5): check if tool insertion is far enough - check this by computing normalized distance of projection along frame hook line.
        #            We ensure that it's at least 5% inserted along the length of the frame hook.
        normalized_dist_along_frame_hook_line = tool_hole_dot / frame_hook_length
        tool_is_inserted_far_enough = (normalized_dist_along_frame_hook_line > 0.05) and (
            normalized_dist_along_frame_hook_line < 1.0
        )

        return all(
            [
                (not robot_and_tool_contact),
                frame_and_tool_hole_contact,
                tool_hole_is_close_enough,
                tool_is_between_hook,
                tool_is_inserted_far_enough,
            ]
        )
