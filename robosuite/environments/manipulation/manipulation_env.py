import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.robot_env import RobotEnv
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot, MobileBaseRobot
from robosuite.utils.observables import Observable, sensor


class ManipulationEnv(RobotEnv):
    """
    Initializes a manipulation-specific robot environment in Mujoco.

    Args:
        robots: Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)

        env_configuration (str): Specifies how to position the robot(s) within the environment. Default is "default",
            which should be interpreted accordingly by any subclasses.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        gripper_types (None or str or list of str): type of gripper, used to instantiate
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

        use_camera_obs (bool): if True, every observation includes rendered image(s)

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

        control_freq (float): how many control signals to receive in every second. This sets the abase of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            This feature is set to False by default to preserve backward compatibility.

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

        seed (int): environment seed. Default is None, where environment is unseeded, ie. random

    Raises:
        ValueError: [Camera obs require offscreen renderer]
        ValueError: [Camera name must be specified to use camera obs]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        composite_controller_configs=None,
        base_types="default",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=False,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # Robot info
        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]
        num_robots = len(robots)

        # Gripper
        gripper_types = self._input2list(gripper_types, num_robots)

        # Robot configurations to pass to super call
        robot_configs = [
            {
                "gripper_type": gripper_types[idx],
            }
            for idx in range(num_robots)
        ]

        # Run superclass init
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            composite_controller_configs=composite_controller_configs,
            base_types=base_types,
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
            robot_configs=robot_configs,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment

        Returns:
            set: All components that can be individually visualized for this environment
        """
        vis_set = super()._visualizations
        vis_set.add("grippers")
        return vis_set

    def _get_obj_eef_sensor(self, prefix, obj_key, fn_name, modality):
        """
        Creates a sensor function that returns the relative position between the object specified by @obj_key
        and the end effector specified by @prefix.

        Args:
            prefix (str): Prefix for the arm to which the end effector belongs
            obj_key (str): Key to access the object's position in the observation cache
            fn_name (str): Name to assign to the sensor function
            modality (str): Modality for the sensor

        Returns:
            function: Sensor function that returns the relative position between the object and the end effector
        """

        @sensor(modality)
        def sensor_fn(obs_cache):
            return (
                obs_cache[obj_key] - obs_cache[f"{prefix}eef_pos"]
                if obj_key in obs_cache and f"{prefix}eef_pos" in obs_cache
                else np.zeros(3)
            )

        sensor_fn.__name__ = fn_name
        return sensor_fn

    def _get_world_pose_in_gripper_sensor(self, prefix, fn_name, modality):
        """
        Creates a sensor function that returns the inverse pose of the gripper.

        Args:
            prefix (str): Prefix for the arm to which the gripper belongs
            fn_name (str): Name to assign to the sensor function
            modality (str): Modality for the sensor

        Returns:
            function: Sensor function that returns the relative pose between the world and the gripper
        """

        @sensor(modality=modality)
        def fn(obs_cache):
            return (
                T.pose_inv(T.pose2mat((obs_cache[f"{prefix}eef_pos"], obs_cache[f"{prefix}eef_quat"])))
                if f"{prefix}eef_pos" in obs_cache and f"{prefix}eef_quat" in obs_cache
                else np.eye(4)
            )

        fn.__name__ = fn_name
        return fn

    def _get_rel_obj_eef_sensor(self, prefix, obj_key, fn_name, new_key_prefix, modality):
        """
        Creates a sensor function that returns the relative position between the object specified by @obj_key
        and populates the observation cache with relative quaternion. This sensor function uses the robots
        gripper's inverse pose which should be in the observation cache.


        Args:
            prefix (str): Prefix used to access the robot arm's inverse pose in the observation cache
            obj_key (str): Key to access the object's position/quaternion in the observation cache
            fn_name (str): Name to assign to the sensor function
            new_key_prefix (str): Prefix to use for the new key in the observation cache
            modality (str): Modality for the sensor

        Returns:
            function: Sensor function that returns the relative position between the object and the end effector
        """

        @sensor(modality=modality)
        def fn(obs_cache):
            # Immediately return default value if cache is empty
            if any(
                [
                    name not in obs_cache
                    for name in [f"{obj_key}_pos", f"{obj_key}_quat", f"world_pose_in_{prefix}gripper"]
                ]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_key}_pos"], obs_cache[f"{obj_key}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache[f"world_pose_in_{prefix}gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_key}_to_{new_key_prefix}eef_quat"] = rel_quat
            return rel_pos

        fn.__name__ = fn_name
        return fn

    def _get_obj_eef_rel_quat_sensor(self, prefix, obj_key, fn_name, modality):
        """
        Creates a sensor function that returns the relative quaternion between the object specified by @obj_key
        and the end effector specified by @prefix.

        Args:
            prefix (str): Prefix for the arm to which the end effector belongs
            obj_key (str): Key to access the object's quaternion in the observation cache
            fn_name (str): Name to assign to the sensor function
            modality (str): Modality for the sensor

        Returns:
            function: Sensor function that returns the relative quaternion between the object and the end effector
        """

        @sensor(modality)
        def sensor_fn(obs_cache):
            return (
                obs_cache[f"{obj_key}_to_{prefix}eef_quat"]
                if f"{obj_key}_to_{prefix}eef_quat" in obs_cache
                else np.zeros(4)
            )

        sensor_fn.__name__ = fn_name
        return sensor_fn

    def _check_grasp(self, gripper, object_geoms):
        """
        Checks whether the specified gripper as defined by @gripper is grasping the specified object in the environment.
        If multiple grippers are specified, will return True if at least one gripper is grasping the object.

        By default, this will return True if at least one geom in both the "left_fingerpad" and "right_fingerpad" geom
        groups are in contact with any geom specified by @object_geoms. Custom gripper geom groups can be
        specified with @gripper as well.

        Args:
            gripper (GripperModel or str or list of str or list of list of str or dict): If a MujocoModel, this is specific
                gripper to check for grasping (as defined by "left_fingerpad" and "right_fingerpad" geom groups). Otherwise,
                this sets custom gripper geom groups which together define a grasp. This can be a string
                (one group of single gripper geom), a list of string (multiple groups of single gripper geoms) or a
                list of list of string (multiple groups of multiple gripper geoms), or a dictionary in the case
                where the robot has multiple arms/grippers. At least one geom from each group must be in contact
                with any geom in @object_geoms for this method to return True.
            object_geoms (str or list of str or MujocoModel): If a MujocoModel is inputted, will check for any
                collisions with the model's contact_geoms. Otherwise, this should be specific geom name(s) composing
                the object to check for contact.

        Returns:
            bool: True if the gripper is grasping the given object
        """
        # Convert object, gripper geoms into standardized form
        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms

        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        elif isinstance(gripper, dict):
            assert all([isinstance(gripper[arm], GripperModel) for arm in gripper]), "Invalid gripper dict format!"
            return any([self._check_grasp(gripper[arm], object_geoms) for arm in gripper])
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.check_contact(g_group, o_geoms):
                return False
        return True

    def _gripper_to_target(self, gripper, target, target_type="body", return_distance=False):
        """
        Calculates the (x,y,z) Cartesian distance (target_pos - gripper_pos) from the specified @gripper to the
        specified @target. If @return_distance is set, will return the Euclidean (scalar) distance instead.
        If the @gripper is a dict, will return the minimum distance across all grippers.

        Args:
            gripper (MujocoModel or dict): Gripper model to update grip site rgb
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
            return_distance (bool): If set, will return Euclidean distance instead of Cartesian distance

        Returns:
            np.array or float: (Cartesian or Euclidean) distance from gripper to target
        """
        if isinstance(gripper, dict):
            assert all([isinstance(gripper[arm], GripperModel) for arm in gripper]), "Invalid gripper dict format!"
            # get the min distance to the target if there are multiple arms
            if return_distance:
                return min(
                    [self._gripper_to_target(gripper[arm], target, target_type, return_distance) for arm in gripper]
                )
            else:
                return min(
                    [self._gripper_to_target(gripper[arm], target, target_type, return_distance) for arm in gripper],
                    key=lambda x: np.linalg.norm(x),
                )

        # Get gripper and target positions
        gripper_pos = self.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        # If target is MujocoModel, grab the correct body as the target and find the target position
        if isinstance(target, MujocoModel):
            target_pos = self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            target_pos = self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            target_pos = self.sim.data.get_site_xpos(target)
        else:
            target_pos = self.sim.data.get_geom_xpos(target)
        # Calculate distance
        diff = target_pos - gripper_pos
        # Return appropriate value
        return np.linalg.norm(diff) if return_distance else diff

    def _visualize_gripper_to_target(self, gripper, target, target_type="body"):
        """
        Colors the grip visualization site proportional to the Euclidean distance to the specified @target.
        Colors go from red --> green as the gripper gets closer. If a dict of grippers is given, will visualize
        all grippers to the target.

        Args:
            gripper (MujocoModel or dict): Gripper model to update grip site rgb
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
        """
        if isinstance(gripper, dict):
            assert all([isinstance(gripper[arm], GripperModel) for arm in gripper]), "Invalid gripper dict format!"
            for arm in gripper:
                self._visualize_gripper_to_target(gripper[arm], target, target_type)
            return
        # Get gripper and target positions
        gripper_pos = self.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        # If target is MujocoModel, grab the correct body as the target and find the target position
        if isinstance(target, MujocoModel):
            target_pos = self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            target_pos = self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            target_pos = self.sim.data.get_site_xpos(target)
        else:
            target_pos = self.sim.data.get_geom_xpos(target)
        # color the gripper site appropriately based on (squared) distance to target
        dist = np.sum(np.square((target_pos - gripper_pos)))
        max_dist = 0.1
        scaled = (1.0 - min(dist / max_dist, 1.0)) ** 15
        rgba = np.zeros(3)
        rgba[0] = 1 - scaled
        rgba[1] = scaled
        self.sim.model.site_rgba[self.sim.model.site_name2id(gripper.important_sites["grip_site"])][:3] = rgba

    def _get_arm_prefixes(self, robot, include_robot_name=True):
        """
        Returns the naming prefixes for the robot arms in the environment used to access proprioceptive information.
        By convention, if there is only one arm, it does not include  the arm name. Otherwise, returns
        a list of prefixes for each arm containing the robot's naming prefix (if include_robot_name is set) and the arm name.

        Args:
            robot (RobotModel): Robot model to extract prefixes from

        Returns:
            list: List of prefixes for the robot arms
        """

        name_pf = robot.robot_model.naming_prefix if include_robot_name else ""
        if len(robot.arms) == 1:
            return [name_pf]

        prefixes = [f"{name_pf}{arm}_" for arm in robot.arms]
        return prefixes

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure inputted robots and the corresponding requested task/configuration combo is legal.
        Should be implemented in every specific task module

        Args:
            robots (str or list of str): Inputted requested robots at the task-level environment
        """
        # Make sure all inputted robots are a manipulation robot
        if type(robots) is str:
            robots = [robots]
        for robot in robots:
            assert issubclass(ROBOT_CLASS_MAPPING[robot], FixedBaseRobot) or issubclass(
                ROBOT_CLASS_MAPPING[robot], MobileBaseRobot
            ), "Only manipulator robots supported for manipulation environment!"
