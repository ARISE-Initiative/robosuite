import random
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import BinsArena
from robosuite.models.objects import (
    BreadObject,
    BreadVisualObject,
    CanObject,
    CanVisualObject,
    CerealObject,
    CerealVisualObject,
    MilkObject,
    MilkVisualObject,
)

# --- MODIFIED IMPORT BLOCK ---
from robosuite.models.objects import (
    BottleObject,
    LemonObject,
    SquareNutObject,
    RoundNutObject,
)
# --- END OF MODIFIED BLOCK ---

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class PickPlaceClutter(ManipulationEnv):
    """
    This class corresponds to the pick place task for a single robot arm.

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

        bin1_pos (3-tuple): Absolute cartesian coordinates of the bin initially holding the objects

        bin2_pos (3-tuple): Absolute cartesian coordinates of the goal bin

        z_offset (float): amount of z offset for initializing objects in bin

        z_rotation (float, tuple, or None): if provided, controls the range of z-rotation initialization
            for the objects

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        single_object_mode (int): specifies which version of the task to do. Note that
            the observations change accordingly.

            :`0`: corresponds to the full task with all types of objects.

            :`1`: corresponds to an easier task with only one type of object initialized
               on the table with every reset. The type is randomized on every reset.

            :`2`: corresponds to an easier task with only one type of object initialized
               on the table with every reset. The type is kept constant and will not
               change between resets.

        object_type (string): if provided, should be one of "milk", "bread", "cereal",
            or "can". Determines which type of object will be spawned on every
            environment reset. Only used if @single_object_mode is 2.

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
        AssertionError: [Invalid object type specified]
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.39, 0.49, 0.82),
        table_friction=(1, 0.005, 0.0001),
        bin1_pos=(0.1, -0.25, 0.8),
        bin2_pos=(0.1, 0.28, 0.8),
        z_offset=0.0,
        z_rotation=None,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        single_object_mode=0,
        object_type=None,
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
    ):
        # task settings
        self.single_object_mode = single_object_mode
        self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        self.object_id_to_sensors = {}  # Maps object id to sensor names for that object
        self.obj_names = ["Milk", "Bread", "Cereal", "Can"]
        if object_type is not None:
            assert object_type in self.object_to_id.keys(), "invalid @object_type argument - choose one of {}".format(
                list(self.object_to_id.keys())
            )
            self.object_id = self.object_to_id[object_type]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # settings for bin position
        self.bin1_pos = np.array(bin1_pos)
        self.bin2_pos = np.array(bin2_pos)
        self.z_offset = z_offset  # z offset for initializing items in bin
        self.z_rotation = z_rotation  # z rotation for initializing items in bin

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
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

          - a discrete reward of 1.0 per object if it is placed in its correct bin

        Un-normalized components if using reward shaping, where the maximum is returned if not solved:

          - Reaching: in [0, 0.1], proportional to the distance between the gripper and the closest object
          - Grasping: in {0, 0.35}, nonzero if the gripper is grasping an object
          - Lifting: in {0, [0.35, 0.5]}, nonzero only if object is grasped; proportional to lifting height
          - Hovering: in {0, [0.5, 0.7]}, nonzero only if object is lifted; proportional to distance from object to bin

        Note that a successfully completed task (object in bin) will return 1.0 per object irregardless of whether the
        environment is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 4.0 (or 1.0 if only a single object is
        being used) as well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_in_bins)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
            if self.single_object_mode == 0:
                reward /= 4.0
        return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already in the correct bins
        active_objs = []
        for i, obj in enumerate(self.objects):
            if self.objects_in_bins[i]:
                continue
            active_objs.append(obj)

        # reaching reward governed by distance to closest object
        r_reach = 0.0
        if active_objs:
            # get reaching reward via minimum distance to a target object
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=active_obj.root_body,
                    target_type="body",
                    return_distance=True,
                )
                for active_obj in active_objs
            ]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        # grasping reward for touching any objects of interest
        r_grasp = (
            int(
                self._check_grasp(
                    gripper=self.robots[0].gripper,
                    object_geoms=[g for active_obj in active_objs for g in active_obj.contact_geoms],
                )
            )
            * grasp_mult
        )

        # lifting reward for picking up an object
        r_lift = 0.0
        if active_objs and r_grasp > 0.0:
            z_target = self.bin2_pos[2] + 0.25
            object_z_locs = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name] for active_obj in active_objs]][
                :, 2
            ]
            z_dists = np.maximum(z_target - object_z_locs, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (lift_mult - grasp_mult)

        # hover reward for getting object above bin
        r_hover = 0.0
        if active_objs:
            target_bin_ids = [self.object_to_id[active_obj.name.lower()] for active_obj in active_objs]
            # segment objects into left of the bins and above the bins
            object_xy_locs = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name] for active_obj in active_objs]][
                :, :2
            ]
            y_check = (
                np.abs(object_xy_locs[:, 1] - self.target_bin_placements[target_bin_ids, 1]) < self.bin_size[1] / 4.0
            )
            x_check = (
                np.abs(object_xy_locs[:, 0] - self.target_bin_placements[target_bin_ids, 0]) < self.bin_size[0] / 4.0
            )
            objects_above_bins = np.logical_and(x_check, y_check)
            objects_not_above_bins = np.logical_not(objects_above_bins)
            dists = np.linalg.norm(self.target_bin_placements[target_bin_ids, :2] - object_xy_locs, axis=1)
            # objects to the left get r_lift added to hover reward,
            # those on the right get max(r_lift) added (to encourage dropping)
            r_hover_all = np.zeros(len(active_objs))
            r_hover_all[objects_above_bins] = lift_mult + (1 - np.tanh(10.0 * dists[objects_above_bins])) * (
                hover_mult - lift_mult
            )
            r_hover_all[objects_not_above_bins] = r_lift + (1 - np.tanh(10.0 * dists[objects_not_above_bins])) * (
                hover_mult - lift_mult
            )
            r_hover = np.max(r_hover_all)

        return r_reach, r_grasp, r_lift, r_hover

    def not_in_bin(self, obj_pos, bin_id):

        bin_x_low = self.bin2_pos[0]
        bin_y_low = self.bin2_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= self.bin_size[0] / 2
        if bin_id < 2:
            bin_y_low -= self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0] / 2
        bin_y_high = bin_y_low + self.bin_size[1] / 2

        res = True
        if (
            bin_x_low < obj_pos[0] < bin_x_high
            and bin_y_low < obj_pos[1] < bin_y_high
            and self.bin2_pos[2] < obj_pos[2] < self.bin2_pos[2] + 0.1
        ):
            res = False
        return res

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # can sample anywhere in bin
        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

        # each object should just be sampled in the bounds of the bin (with some tolerance)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-bin_x_half, bin_x_half],
                y_range=[-bin_y_half, bin_y_half],
                rotation=self.z_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=self.z_offset,
            )
        )
        
        # --- SAMPLER 2: For the CLUTTER objects ---
        # (This is the new block you need to add)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ClutterObjectSampler", # Note: A different name is good practice
                mujoco_objects=self.clutter_objects, # <-- This one places the clutter
                x_range=[-bin_x_half, bin_x_half],
                y_range=[-bin_y_half, bin_y_half],
                rotation=self.z_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,  # This is critical!
                reference_pos=self.bin1_pos,
                z_offset=self.z_offset,
            )
        )
        # --- END OF ADDED BLOCK ---

        # each visual object should just be at the center of each target bin
        index = 0
        for vis_obj in self.visual_objects:

            # get center of target bin
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if index == 0 or index == 2:
                bin_x_low -= self.bin_size[0] / 2
            if index < 2:
                bin_y_low -= self.bin_size[1] / 2
            bin_x_high = bin_x_low + self.bin_size[0] / 2
            bin_y_high = bin_y_low + self.bin_size[1] / 2
            bin_center = np.array(
                [
                    (bin_x_low + bin_x_high) / 2.0,
                    (bin_y_low + bin_y_high) / 2.0,
                ]
            )

            # placement is relative to object bin, so compute difference and send to placement initializer
            rel_center = bin_center - self.bin1_pos[:2]

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{vis_obj.name}ObjectSampler",
                    mujoco_objects=vis_obj,
                    x_range=[rel_center[0], rel_center[0]],
                    y_range=[rel_center[1], rel_center[1]],
                    rotation=0.0,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.bin1_pos,
                    z_offset=self.bin2_pos[2] - self.bin1_pos[2],
                )
            )
            index += 1

    def _construct_visual_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.visual_objects = []
        for vis_obj_cls, obj_name in zip(
            (MilkVisualObject, BreadVisualObject, CerealVisualObject, CanVisualObject),
            self.obj_names,
        ):
            vis_name = "Visual" + obj_name
            vis_obj = vis_obj_cls(name=vis_name)
            self.visual_objects.append(vis_obj)

    def _construct_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.objects = []
        for obj_cls, obj_name in zip(
            (MilkObject, BreadObject, CerealObject, CanObject),
            self.obj_names,
        ):
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)
    
    # --- MODIFIED: New function to construct clutter objects ---
    def _construct_clutter_objects(self):
        """
        Function to load clutter objects.
        """
        self.clutter_objects = [
            BottleObject(name="ClutterBottle"),
            LemonObject(name="ClutterLemon"),
            #SquareNutObject(name="ClutterSquareNut"),
            #RoundNutObject(name="ClutterRoundNut"),
        ]
# --- END OF MODIFIED BLOCK ---

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["bins"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = BinsArena(
            bin1_pos=self.bin1_pos, table_full_size=self.table_full_size, table_friction=self.table_friction
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # store some arena attributes
        self.bin_size = mujoco_arena.table_full_size

        # make objects
        self._construct_visual_objects()
        self._construct_objects()
        self._construct_clutter_objects() # Call to new function

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            #mujoco_objects=self.visual_objects + self.objects,
            mujoco_objects=self.visual_objects + self.objects + self.clutter_objects,
        )

        # Generate placement initializer
        self._get_placement_initializer()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # object-specific ids
        #for obj in self.visual_objects + self.objects:
        for obj in self.visual_objects + self.objects + self.clutter_objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.objects))

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.objects), 3))
        for i, obj in enumerate(self.objects):
            bin_id = i
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2.0
            if bin_id < 2:
                bin_y_low -= self.bin_size[1] / 2.0
            bin_x_low += self.bin_size[0] / 4.0
            bin_y_low += self.bin_size[1] / 4.0
            self.target_bin_placements[i, :] = [bin_x_low, bin_y_low, self.bin2_pos[2]]

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

            # Reset obj sensor mappings
            self.object_id_to_sensors = {}

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors = [
                self._get_world_pose_in_gripper_sensor(full_pf, f"world_pose_in_{arm_pf}gripper", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [fn.__name__ for fn in sensors]
            actives = [False] * len(sensors)
            enableds = [True] * len(sensors)

            for i, obj in enumerate(self.objects):
                # Create object sensors
                using_obj = self.single_object_mode == 0 or self.object_id == i
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                enableds += [using_obj] * len(obj_sensor_names)
                actives += [using_obj] * len(obj_sensor_names)
                self.object_id_to_sensors[i] = obj_sensor_names

            if self.single_object_mode == 1:
                # This is randomly sampled object, so we need to include object id as observation
                @sensor(modality=modality)
                def obj_id(obs_cache):
                    return self.object_id

                sensors.append(obj_id)
                names.append("obj_id")
                enableds.append(True)
                actives.append(True)

            # Create observables
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active,
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
        full_prefixes = self._get_arm_prefixes(self.robots[0])

        sensors = [
            self._get_rel_obj_eef_sensor(arm_pf, obj_name, f"{obj_name}_to_{full_pf}eef_pos", full_pf, modality)
            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
        ]
        sensors += [
            self._get_obj_eef_rel_quat_sensor(full_pf, obj_name, f"{obj_name}_to_{full_pf}eef_quat", modality)
            for full_pf in full_prefixes
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
                # Set the visual object body locations
                if "visual" in obj.name.lower():
                    self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                    self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                else:
                    # Set the collision object joints
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Set the bins to the desired position
        self.sim.model.body_pos[self.sim.model.body_name2id("bin1")] = self.bin1_pos
        self.sim.model.body_pos[self.sim.model.body_name2id("bin2")] = self.bin2_pos

        # Move objects out of the scene depending on the mode
        obj_names = {obj.name for obj in self.objects}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(obj_names))
            for obj_type, i in self.object_to_id.items():
                if obj_type.lower() in self.obj_to_use.lower():
                    self.object_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.objects[self.object_id].name
        if self.single_object_mode in {1, 2}:
            obj_names.remove(self.obj_to_use)
            self.clear_objects(list(obj_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.object_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active object, else False
                    self._observables[name].set_enabled(i == self.object_id)
                    self._observables[name].set_active(i == self.object_id)

    def _check_success(self):
        """
        Check if all objects have been successfully placed in their corresponding bins.

        Returns:
            bool: True if all objects are placed correctly
        """
        # remember objects that are in the correct bins
        for i, obj in enumerate(self.objects):
            obj_str = obj.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = min(
                [
                    np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - obj_pos)
                    for arm in self.robots[0].arms
                ]
            )
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_in_bins[i] = int((not self.not_in_bin(obj_pos, i)) and r_reach < 0.6)

        # returns True if a single object is in the correct bin
        if self.single_object_mode in {1, 2}:
            return np.sum(self.objects_in_bins) > 0

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.objects)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest object.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest object
        if vis_settings["grippers"]:
            # if the robot has multiple arms color each arm independently based on its closest object
            for arm in self.robots[0].arms:
                # find closest object
#                dists = [
#                    self._gripper_to_target(
#                        gripper=self.robots[0].gripper[arm],
#                        target=obj.root_body,
#                        target_type="body",
#                        return_distance=True,
#                    )
#                    for obj in self.objects
#                ]
#                closest_obj_id = np.argmin(dists)
#                # Visualize the distance to this target
#                self._visualize_gripper_to_target(
#                    gripper=self.robots[0].gripper[arm],
#                    target=self.objects[closest_obj_id].root_body,
#                    target_type="body",
#                )
                # find closest object
                dists_task = [
                    self._gripper_to_target(
                        gripper=self.robots[0].gripper[arm],
                        target=obj.root_body,
                        target_type="body",
                        return_distance=True,
                    )
                    for obj in self.objects
                ]

                # --- MODIFIED: Also check distance to clutter objects ---
                dists_clutter = [
                    self._gripper_to_target(
                        gripper=self.robots[0].gripper[arm],
                        target=obj.root_body,
                        target_type="body",
                        return_distance=True,
                    )
                    for obj in self.clutter_objects
                ]

                all_dists = dists_task + dists_clutter
                all_objects = self.objects + self.clutter_objects
                # --- END OF MODIFIED BLOCK ---

                closest_obj_id = np.argmin(all_dists)
                # Visualize the distance to this target
                self._visualize_gripper_to_target(
                    gripper=self.robots[0].gripper[arm],
                    target=all_objects[closest_obj_id].root_body,
                    target_type="body",
                )


class PickPlaceSingle(PickPlaceClutter):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=1, **kwargs)


class PickPlaceMilk(PickPlaceClutter):
    """
    Easier version of task - place one milk into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="milk", **kwargs)


class PickPlaceBread(PickPlaceClutter):
    """
    Easier version of task - place one bread into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="bread", **kwargs)


class PickPlaceCereal(PickPlaceClutter):
    """
    Easier version of task - place one cereal into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="cereal", **kwargs)


class PickPlaceCan(PickPlaceClutter):
    """
    Easier version of task - place one can into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="can", **kwargs)
