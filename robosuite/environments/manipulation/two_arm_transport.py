from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import BoxObject, HammerObject, TransportGroup
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class TwoArmTransport(TwoArmEnv):
    """
    This class corresponds to the transport task for two robot arms, requiring a payload to be transported from an
    initial bin into a target bin, while removing trash from the target bin to a trash bin.

    Args:
        robots (str or list of str): Specification for specific robot(s)
            Note: Must be either 2 robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment if two robots inputted. Can be either:

            :`'parallel'`: Sets up the two robots next to each other on the -x side of the table
            :`'opposed'`: Sets up the two robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" "opposed" if two robots are used.

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

        tables_boundary (3-tuple): x, y, and z dimensions of the table bounds. Two tables will be created at the edges of
            this boundary

        table_friction (3-tuple): the three mujoco friction parameters for
            each table.

        bin_size (3-tuple): (x,y,z) dimensions of bins to use

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

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

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
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        tables_boundary=(0.8, 1.2, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        bin_size=(0.3, 0.3, 0.15),
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
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        ep_meta={},
    ):
        # settings for table top
        self.tables_boundary = tables_boundary
        self.table_full_size = np.array(tables_boundary)
        self.table_full_size[1] *= 0.25  # each table size will only be a fraction of the full boundary
        self.table_friction = table_friction
        self.table_offsets = np.zeros((2, 3))
        self.table_offsets[0, 1] = self.tables_boundary[1] * -3 / 8  # scale y offset
        self.table_offsets[1, 1] = self.tables_boundary[1] * 3 / 8  # scale y offset
        self.table_offsets[:, 2] = 0.8  # scale z offset
        self.bin_size = np.array(bin_size)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.height_threshold = 0.1  # threshold above the table surface which the payload is considered lifted

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
            lite_physics=lite_physics,
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
            seed=seed,
            ep_meta=ep_meta,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided when the payload is in the target bin and the trash is in the trash
                bin

        Un-normalized max-wise components if using reward shaping:

            # TODO!

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        # Initialize reward
        reward = 0

        # use a shaping reward if specified
        if self.reward_shaping:
            # TODO! So we print a warning and force sparse rewards
            print(f"\n\nWarning! No dense reward current implemented for this task. Forcing sparse rewards\n\n")
            self.reward_shaping = False

        # Else this is the sparse reward setting
        else:
            # Provide reward if payload is in target bin and trash is in trash bin
            if self._check_success():
                reward = 1.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation, offset in zip(self.robots, (np.pi / 2, -np.pi / 2), (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    xpos += np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.6, 0.6)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = MultiTableArena(
            table_offsets=self.table_offsets,
            table_rots=0,
            table_full_sizes=self.table_full_size,
            table_frictions=self.table_friction,
            has_legs=True,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.8894354364730311, -3.481824231498976e-08, 1.7383813133506494],
            quat=[0.6530981063842773, 0.2710406184196472, 0.27104079723358154, 0.6530979871749878],
        )

        # TODO: Add built-in method into TwoArmEnv so we have an elegant way of automatically adding extra cameras to all these envs
        # Add shoulder cameras
        mujoco_arena.set_camera(
            camera_name="shouldercamera0",
            pos=[0.4430096057365183, -1.0697399743660143, 1.3639950119362048],
            quat=[0.804057240486145, 0.5531665086746216, 0.11286306381225586, 0.18644218146800995],
        )
        mujoco_arena.set_camera(
            camera_name="shouldercamera1",
            pos=[-0.40900713993039983, 0.9613722572245062, 1.3084072951772754],
            quat=[0.15484197437763214, 0.12077208608388901, -0.5476858019828796, -0.8133130073547363],
        )

        # Add relevant materials
        # Textures to use
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # initialize objects of interest
        payload = HammerObject(
            name="payload",
            handle_radius=0.015,
            handle_length=0.20,
            handle_density=150.0,
            handle_friction=4.0,
            head_density_ratio=1.5,
            rng=self.rng,
        )
        trash = BoxObject(name="trash", size=[0.02, 0.02, 0.02], material=redwood, rng=self.rng)
        self.transport = TransportGroup(
            name="transport",
            payload=payload,
            trash=trash,
            bin_size=self.bin_size,
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.transport.objects.values()),
        )

        # Create placement initializer
        self._get_placement_initializer()

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # Pre-define settings for each object's placement
        object_names = ["start_bin", "lid", "payload", "target_bin", "trash", "trash_bin"]
        table_nums = [0, 0, 0, 1, 1, 1]
        x_centers = [
            self.table_full_size[0] * 0.25,
            0,  # gets overridden anyways
            0,  # gets overridden anyways
            -self.table_full_size[0] * 0.25,
            0,  # gets overridden anyways
            self.table_full_size[0] * 0.25,
        ]
        pos_tol = 0.005
        rot_centers = [0, 0, np.pi / 2, 0, 0, 0]
        rot_tols = [0, 0, np.pi / 6, 0, 0.3 * np.pi, 0]
        rot_axes = ["z", "z", "y", "z", "z", "z"]
        for obj_name, x, r, r_tol, r_axis, table_num in zip(
            object_names, x_centers, rot_centers, rot_tols, rot_axes, table_nums
        ):
            # Get name and table
            obj = self.transport.objects[obj_name]
            table_pos = self.table_offsets[table_num]
            # Create sampler for this object and add it to the sequential sampler
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{obj_name}ObjectSampler",
                    mujoco_objects=obj,
                    x_range=[x - pos_tol, x + pos_tol],
                    y_range=[-pos_tol, pos_tol],
                    rotation=[r - r_tol, r + r_tol],
                    rotation_axis=r_axis,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=table_pos,
                    z_offset=0.001,
                    rng=self.rng,
                )
            )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"

            # position and rotation of payload
            @sensor(modality=modality)
            def payload_pos(obs_cache):
                return np.array(self.transport.payload_pos)

            @sensor(modality=modality)
            def payload_quat(obs_cache):
                return np.array(self.transport.payload_quat)

            # position and rotation of trash
            @sensor(modality=modality)
            def trash_pos(obs_cache):
                return np.array(self.transport.trash_pos)

            @sensor(modality=modality)
            def trash_quat(obs_cache):
                return np.array(self.transport.trash_quat)

            # position and rotation of lid handle
            @sensor(modality=modality)
            def lid_handle_pos(obs_cache):
                return np.array(self.transport.lid_handle_pos)

            @sensor(modality=modality)
            def lid_handle_quat(obs_cache):
                return np.array(self.transport.lid_handle_quat)

            # bin positions
            @sensor(modality=modality)
            def target_bin_pos(obs_cache):
                return np.array(self.transport.target_bin_pos)

            @sensor(modality=modality)
            def trash_bin_pos(obs_cache):
                return np.array(self.transport.trash_bin_pos)

            # Key boolean checks
            @sensor(modality=modality)
            def payload_in_target_bin(obs_cache):
                return self.transport.payload_in_target_bin

            @sensor(modality=modality)
            def trash_in_trash_bin(obs_cache):
                return self.transport.trash_in_trash_bin

            sensors = [
                payload_pos,
                payload_quat,
                trash_pos,
                trash_quat,
                lid_handle_pos,
                lid_handle_quat,
                target_bin_pos,
                trash_bin_pos,
                payload_in_target_bin,
                trash_in_trash_bin,
            ]
            names = [s.__name__ for s in sensors]

            arm_sensor_fns = []
            if self.env_configuration == "single-robot":
                # If single-robot, we only have one robot. gripper 0 is always right and gripper 1 is always left
                pf0 = self.robots[0].robot_model.naming_prefix + "right_"
                pf1 = self.robots[0].robot_model.naming_prefix + "left_"
                prefixes = [pf0, pf1]
                arm_sensor_fns = [
                    self._get_obj_eef_sensor(full_pf, "payload_pos", f"gripper{i}_to_payload", modality)
                    for i, full_pf in enumerate(prefixes)
                ]
                arm_sensor_fns += [
                    self._get_obj_eef_sensor(pf0, "lid_handle_pos", "gripper0_to_lid_handle", modality),
                    self._get_obj_eef_sensor(pf1, "trash_pos", "gripper1_to_trash", modality),
                ]
            else:
                # If not single-robot, we have two robots. gripper 0 is always the first robot's gripper and
                # gripper 1 is always the second robot's gripper. However, must account for the fact that
                # each robot may have multiple arms/grippers
                robot_arm_prefixes = [self._get_arm_prefixes(robot, include_robot_name=False) for robot in self.robots]
                robot_full_prefixes = [self._get_arm_prefixes(robot, include_robot_name=True) for robot in self.robots]
                for i, (arm_prefixes, full_prefixes) in enumerate(zip(robot_arm_prefixes, robot_full_prefixes)):
                    arm_sensor_fns += [
                        self._get_obj_eef_sensor(full_pf, "payload_pos", f"{arm_pf}gripper{i}_to_payload", modality)
                        for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                    ]

                    # If this is the first robot, add lid handle sensor, else add trash sensor
                    if i == 0:
                        arm_sensor_fns += [
                            self._get_obj_eef_sensor(
                                full_pf, "lid_handle_pos", f"{arm_pf}gripper0_to_lid_handle", modality
                            )
                            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                        ]
                    else:
                        arm_sensor_fns += [
                            self._get_obj_eef_sensor(full_pf, "trash_pos", f"{arm_pf}gripper1_to_trash", modality)
                            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                        ]

            sensors += arm_sensor_fns
            names += [s.__name__ for s in arm_sensor_fns]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Update sim
        self.transport.update_sim(sim=self.sim)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Initialize placeholders that we'll need to override the payload, lid, and trash object locations
            start_bin_pos = None
            target_bin_pos = None

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # If this is toolbox or good bin, store their sampled positions
                if "start_bin" in obj.name and "lid" not in obj.name:
                    start_bin_pos = obj_pos
                elif "target_bin" in obj.name:
                    target_bin_pos = obj_pos
                # Else if this is either the lid, payload, or trash object,
                # we override their positions to match their respective containers' positions
                elif "lid" in obj.name:
                    obj_pos = (start_bin_pos[0], start_bin_pos[1], obj_pos[2] + self.transport.bin_size[2])
                elif "payload" in obj.name:
                    obj_pos = (
                        start_bin_pos[0],
                        start_bin_pos[1],
                        obj_pos[2] + self.transport.objects["start_bin"].wall_thickness,
                    )
                elif "trash" in obj.name and "bin" not in obj.name:
                    obj_pos = (
                        target_bin_pos[0],
                        target_bin_pos[1],
                        obj_pos[2] + self.transport.objects["target_bin"].wall_thickness,
                    )
                # Set the collision object joints
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _check_success(self):
        """
        Check if payload is in target in and trash is in trash bin

        Returns:
            bool: True if transport has been completed
        """
        return True if self.transport.payload_in_target_bin and self.transport.trash_in_trash_bin else False
