import numpy as np

from robosuite.environments.robot_env import RobotEnv
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
from robosuite.robots import ROBOT_CLASS_MAPPING, Manipulator


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

        mount_types (None or str or list of str): type of mount, used to instantiate mount models from mount factory.
            Default is "default", which is the default mount associated with the robot(s) the 'robots' specification.
            None results in no mount, and any other (valid) model overrides the default mount. Should either be
            single str if same mount type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        gripper_types (None or str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initial_qpos (None or tuple or list of tuples): If set, sets custom initial qpos values for robot models.
            Default of None corresponds to defalt initial qpos values defined by robot model.

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
        ValueError: [Camera obs require offscreen renderer]
        ValueError: [Camera name must be specified to use camera obs]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        mount_types="default",
        gripper_types="default",
        initial_qpos=None,
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        optimize_physics=False,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        seed=None,
    ):
        # Robot info
        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]
        num_robots = len(robots)

        # Gripper
        gripper_types = self._input2list(gripper_types, num_robots)

        # Initial qpos
        if initial_qpos is not None:
            if initial_qpos[0] is None or type(initial_qpos[0]) is list or type(initial_qpos[0]) is tuple:
                initial_qpos = list(initial_qpos)
            else:
                initial_qpos = [initial_qpos for _ in range(num_robots)]

        # Robot configurations to pass to super call
        robot_configs = [
            {
                "gripper_type": gripper_types[idx],
                "initial_qpos": initial_qpos,
            }
            for idx in range(num_robots)
        ]

        # Run superclass init
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=mount_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            optimize_physics=optimize_physics,
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

    def _check_grasp(self, gripper, object_geoms):
        """
        Checks whether the specified gripper as defined by @gripper is grasping the specified object in the environment.

        By default, this will return True if at least one geom in both the "left_fingerpad" and "right_fingerpad" geom
        groups are in contact with any geom specified by @object_geoms. Custom gripper geom groups can be
        specified with @gripper as well.

        Args:
            gripper (GripperModel or str or list of str or list of list of str): If a MujocoModel, this is specific
            gripper to check for grasping (as defined by "left_fingerpad" and "right_fingerpad" geom groups). Otherwise,
                this sets custom gripper geom groups which together define a grasp. This can be a string
                (one group of single gripper geom), a list of string (multiple groups of single gripper geoms) or a
                list of list of string (multiple groups of multiple gripper geoms). At least one geom from each group
                must be in contact with any geom in @object_geoms for this method to return True.
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

        Args:
            gripper (MujocoModel): Gripper model to update grip site rgb
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
            return_distance (bool): If set, will return Euclidean distance instead of Cartesian distance

        Returns:
            np.array or float: (Cartesian or Euclidean) distance from gripper to target
        """
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
        Colors go from red --> green as the gripper gets closer.

        Args:
            gripper (MujocoModel): Gripper model to update grip site rgb
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
        """
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
            assert issubclass(
                ROBOT_CLASS_MAPPING[robot], Manipulator
            ), "Only manipulator robots supported for manipulation environment!"
