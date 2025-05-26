from collections import OrderedDict
from copy import deepcopy

import numpy as np

import robosuite.macros as macros
from robosuite.environments.base import MujocoEnv
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from robosuite.utils.observables import Observable, sensor


class RobotEnv(MujocoEnv):
    """
    Initializes a robot environment in Mujoco.

    Args:
        robots: Specification for specific robot(s) to be instantiated within this env

        env_configuration (str): Specifies how to position the robot(s) within the environment. Default is "default",
            which should be interpreted accordingly by any subclasses.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        mount_types (None or str or list of str): type of mount, used to instantiate mount models from mount factory.
            Default is "default", which is the default mount associated with the robot(s) the 'robots' specification.
            None results in no mount, and any other (valid) model overrides the default mount. Should either be
            single str if same mount type is to be used for all robots or else it should be a list of the same
            length as "robots" param

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

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str or list of str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse. When a list of strings is provided, it will render from multiple camera angles.

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

        robot_configs (list of dict): Per-robot configurations set from any subclass initializers.

        seed (int): environment seed. Default is None, where environment is unseeded, ie. random

    Raises:
        ValueError: [Camera obs require offscreen renderer]
        ValueError: [Camera name must be specified to use camera obs]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        base_types="default",
        controller_configs=None,
        initialization_noise=None,
        use_camera_obs=True,
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
        camera_segmentations=None,
        robot_configs=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # First, verify that correct number of robots are being inputted
        self.env_configuration = env_configuration
        self._check_robot_configuration(robots)

        # Robot
        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]
        self.num_robots = len(robots)
        self.robot_names = robots
        self.robots = self._input2list(None, self.num_robots)
        self._action_dim = None

        # Robot base
        base_types = self._input2list(base_types, self.num_robots)

        # Composite Controller
        controller_configs = self._input2list(controller_configs, self.num_robots)

        # Initialization Noise
        initialization_noise = self._input2list(initialization_noise, self.num_robots)

        # Observations -- Ground truth = object_obs, Image data = camera_obs
        self.use_camera_obs = use_camera_obs

        # Camera / Rendering Settings
        self.has_offscreen_renderer = has_offscreen_renderer
        self.camera_names = (
            list(camera_names) if type(camera_names) is list or type(camera_names) is tuple else [camera_names]
        )
        self.num_cameras = len(self.camera_names)

        self.camera_heights = self._input2list(camera_heights, self.num_cameras)
        self.camera_widths = self._input2list(camera_widths, self.num_cameras)
        self.camera_depths = self._input2list(camera_depths, self.num_cameras)
        self.camera_segmentations = self._input2list(camera_segmentations, self.num_cameras)
        # We need to parse camera segmentations more carefully since it may be a nested list
        seg_is_nested = False
        for i, camera_s in enumerate(self.camera_segmentations):
            if isinstance(camera_s, list) or isinstance(camera_s, tuple):
                seg_is_nested = True
                break
        camera_segs = deepcopy(self.camera_segmentations)
        for i, camera_s in enumerate(self.camera_segmentations):
            if camera_s is not None:
                self.camera_segmentations[i] = self._input2list(camera_s, 1) if seg_is_nested else deepcopy(camera_segs)

        # sanity checks for camera rendering
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Error: Camera observations require an offscreen renderer!")
        if self.use_camera_obs and self.camera_names is None:
            raise ValueError("Must specify at least one camera name when using camera obs")

        # Robot configurations -- update from subclass configs
        if robot_configs is None:
            robot_configs = [{} for _ in range(self.num_robots)]
        self.robot_configs = [
            dict(
                **{
                    "composite_controller_config": controller_configs[idx],
                    "base_type": base_types[idx],
                    "initialization_noise": initialization_noise[idx],
                    "control_freq": control_freq,
                    "lite_physics": lite_physics,
                },
                **robot_config,
            )
            for idx, robot_config in enumerate(robot_configs)
        ]

        # Run superclass init
        super().__init__(
            has_renderer=has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def visualize(self, vis_settings):
        """
        In addition to super call, visualizes robots.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "robots" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)
        # Loop over robots to visualize them independently
        for robot in self.robots:
            robot.visualize(vis_settings=vis_settings)

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment

        Returns:
            set: All components that can be individually visualized for this environment
        """
        vis_set = super()._visualizations
        vis_set.add("robots")
        return vis_set

    @property
    def action_spec(self):
        """
        Action space (low, high) for this environment

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        low, high = [], []
        for robot in self.robots:
            lo, hi = robot.action_limits
            low, high = np.concatenate([low, lo]), np.concatenate([high, hi])
        return low, high

    @property
    def action_dim(self):
        """
        Size of the action space

        Returns:
            int: Action space dimension
        """
        return self._action_dim

    @staticmethod
    def _input2list(inp, length):
        """
        Helper function that converts an input that is either a single value or a list into a list

        Args:
            inp (None or str or list): Input value to be converted to list
            length (int): Length of list to broadcast input to

        Returns:
            list: input @inp converted into a list of length @length
        """
        # convert to list if necessary
        return list(inp) if type(inp) is list or type(inp) is tuple else [inp for _ in range(length)]

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Load robots
        self._load_robots()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Setup robot-specific references as well (note: requires resetting of sim for robot first)
        for robot in self.robots:
            robot.reset_sim(self.sim)
            robot.setup_references()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Loops through all robots and grabs their corresponding
        observables to add to the procedurally generated dict of observables

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        # Loop through all robots and grab their observables, adding it to the proprioception modality
        for robot in self.robots:
            robot_obs = robot.setup_observables()
            observables.update(robot_obs)

        # Loop through cameras and update the observations if using camera obs
        if self.use_camera_obs:
            # Create sensor information
            sensors = []
            names = []
            for cam_name, cam_w, cam_h, cam_d, cam_segs in zip(
                self.camera_names,
                self.camera_widths,
                self.camera_heights,
                self.camera_depths,
                self.camera_segmentations,
            ):

                # Add cameras associated to our arrays
                cam_sensors, cam_sensor_names = self._create_camera_sensors(
                    cam_name, cam_w=cam_w, cam_h=cam_h, cam_d=cam_d, cam_segs=cam_segs, modality="image"
                )
                sensors += cam_sensors
                names += cam_sensor_names

            # If any the camera segmentations are not None, then we shrink all the sites as a hacky way to
            # prevent them from being rendered in the segmentation mask
            if not all(seg is None for seg in self.camera_segmentations):
                self.sim.model.site_size[:, :] = 1.0e-8

            # Create observables for these cameras
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _create_camera_sensors(self, cam_name, cam_w, cam_h, cam_d, cam_segs, modality="image"):
        """
        Helper function to create sensors for a given camera. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.
        Args:
            cam_name (str): Name of camera to create sensors for
            cam_w (int): Width of camera
            cam_h (int): Height of camera
            cam_d (bool): Whether to create a depth sensor as well
            cam_segs (None or list): Type of segmentation(s) to use, where each entry can be the following:
                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            modality (str): Modality to assign to all sensors
        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given camera
                names (list): array of corresponding observable names
        """
        # Make sure we get correct convention
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        # Create sensor information
        sensors = []
        names = []

        # Add camera observables to the dict
        rgb_sensor_name = f"{cam_name}_image"
        depth_sensor_name = f"{cam_name}_depth"
        segmentation_sensor_name = f"{cam_name}_segmentation"

        @sensor(modality=modality)
        def camera_rgb(obs_cache):
            img = self.sim.render(
                camera_name=cam_name,
                width=cam_w,
                height=cam_h,
                depth=cam_d,
            )
            if cam_d:
                rgb, depth = img
                obs_cache[depth_sensor_name] = np.expand_dims(depth[::convention], axis=-1)
                return rgb[::convention]
            else:
                return img[::convention]

        sensors.append(camera_rgb)
        names.append(rgb_sensor_name)

        if cam_d:

            @sensor(modality=modality)
            def camera_depth(obs_cache):
                return obs_cache[depth_sensor_name] if depth_sensor_name in obs_cache else np.zeros((cam_h, cam_w, 1))

            sensors.append(camera_depth)
            names.append(depth_sensor_name)

        if cam_segs is not None:
            # Define mapping we'll use for segmentation
            for cam_s in cam_segs:
                seg_sensor, seg_sensor_name = self._create_segementation_sensor(
                    cam_name=cam_name,
                    cam_w=cam_w,
                    cam_h=cam_h,
                    cam_s=cam_s,
                    seg_name_root=segmentation_sensor_name,
                    modality=modality,
                )

                sensors.append(seg_sensor)
                names.append(seg_sensor_name)

        return sensors, names

    def _create_segementation_sensor(self, cam_name, cam_w, cam_h, cam_s, seg_name_root, modality="image"):
        """
        Helper function to create sensors for a given camera. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            cam_name (str): Name of camera to create sensors for
            cam_w (int): Width of camera
            cam_h (int): Height of camera
            cam_s (None or list): Type of segmentation to use, should be the following:
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level
            seg_name_root (str): Sensor name root to assign to this sensor

            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                camera_segmentation (function): Generated sensor function for this segmentation sensor
                name (str): Corresponding sensor name
        """
        # Make sure we get correct convention
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        if cam_s == "instance":
            name2id = {inst: i for i, inst in enumerate(list(self.model.instances_to_ids.keys()))}
            mapping = {idn: name2id[inst] for idn, inst in self.model.geom_ids_to_instances.items()}
        elif cam_s == "class":
            name2id = {cls: i for i, cls in enumerate(list(self.model.classes_to_ids.keys()))}
            mapping = {idn: name2id[cls] for idn, cls in self.model.geom_ids_to_classes.items()}
        else:  # element
            # No additional mapping needed
            mapping = None

        @sensor(modality=modality)
        def camera_segmentation(obs_cache):
            seg = self.sim.render(
                camera_name=cam_name,
                width=cam_w,
                height=cam_h,
                depth=False,
                segmentation=True,
            )
            seg = np.expand_dims(seg[::convention, :, 1], axis=-1)
            # Map raw IDs to grouped IDs if we're using instance or class-level segmentation
            if mapping is not None:
                seg = (
                    np.fromiter(map(lambda x: mapping.get(x, -1), seg.flatten()), dtype=np.int32).reshape(
                        cam_h, cam_w, 1
                    )
                    + 1
                )
            return seg

        name = f"{seg_name_root}_{cam_s}"

        return camera_segmentation, name

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # Run superclass reset functionality
        super()._reset_internal()

        # Reset action dim
        self._action_dim = 0

        # Reset robot and update action space dimension along the way
        for robot in self.robots:
            robot.reset(deterministic=self.deterministic_reset, rng=self.rng)
            self._action_dim += robot.action_dim

        # Update cameras if appropriate
        if self.use_camera_obs:
            temp_names = []
            for cam_name in self.camera_names:
                if "all-" in cam_name:
                    # We need to add all robot-specific camera names that include the key after the tag "all-"
                    start_idx = len(temp_names) - 1
                    key = cam_name.replace("all-", "")
                    for robot in self.robots:
                        for robot_cam_name in robot.robot_model.cameras:
                            if key in robot_cam_name:
                                temp_names.append(robot_cam_name)
                    # We also need to broadcast the corresponding values from each camera dimensions as well
                    end_idx = len(temp_names) - 1
                    self.camera_widths = (
                        self.camera_widths[:start_idx]
                        + [self.camera_widths[start_idx]] * (end_idx - start_idx)
                        + self.camera_widths[(start_idx + 1) :]
                    )
                    self.camera_heights = (
                        self.camera_heights[:start_idx]
                        + [self.camera_heights[start_idx]] * (end_idx - start_idx)
                        + self.camera_heights[(start_idx + 1) :]
                    )
                    self.camera_depths = (
                        self.camera_depths[:start_idx]
                        + [self.camera_depths[start_idx]] * (end_idx - start_idx)
                        + self.camera_depths[(start_idx + 1) :]
                    )
                else:
                    # We simply add this camera to the temp_names
                    temp_names.append(cam_name)
            # Lastly, replace camera names with the updated ones
            self.camera_names = temp_names

    def _pre_action(self, action, policy_step=False):
        """
        Overrides the superclass method to control the robot(s) within this enviornment using their respective
        controllers using the passed actions and gripper control.

        Args:
            action (np.array): The control to apply to the robot(s). Note that this should be a flat 1D array that
                encompasses all actions to be distributed to each robot if there are multiple. For each section of the
                action space assigned to a single robot, the first @self.robots[i].controller.control_dim dimensions
                should be the desired controller actions and if the robot has a gripper, the next
                @self.robots[i].gripper.dof dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """
        # Verify that the action is the correct dimension
        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )

        # Update robot joints based on controller actions
        cutoff = 0
        for idx, robot in enumerate(self.robots):
            robot_action = action[cutoff : cutoff + robot.action_dim]
            robot.control(robot_action, policy_step=policy_step)
            cutoff += robot.action_dim

    def _load_robots(self):
        """
        Instantiates robots and stores them within the self.robots attribute
        """
        # Loop through robots and instantiate Robot object for each
        for idx, (name, config) in enumerate(zip(self.robot_names, self.robot_configs)):
            # Create the robot instance
            self.robots[idx] = ROBOT_CLASS_MAPPING[name](robot_type=name, idn=idx, **config)
            # Now, load the robot models
            self.robots[idx].load_model()

    def reward(self, action):
        """
        Runs superclass method by default
        """
        return super().reward(action)

    def _check_success(self):
        """
        Runs superclass method by default
        """
        return super()._check_success()

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure inputted robots and the corresponding requested task/configuration combo is legal.
        Should be implemented in every specific task module

        Args:
            robots (str or list of str): Inputted requested robots at the task-level environment
        """
        raise NotImplementedError
