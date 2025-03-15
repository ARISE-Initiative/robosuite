import random
from collections import OrderedDict

import numpy as np
from copy import deepcopy

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArenaDoubleBinsA, TableArenaDoubleBinsB, TableArenaDoubleBinsC, TableArenaDoubleBinsD, TableArenaDoubleBinsDa, TableArenaDoubleBinsE
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler,SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat

from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, add_material

from robosuite.models.objects import (
    BreadObject,
    AppleJuiceObject,
    CoffeeObject,
    LimoncelloObject,
    StrawberryJuiceObject,
    JuiceGrayObject,
    BeerObject,
    SodaObject,
    TeaObject,
    MilkBlueObject,
    MilkGreenObject,
    MilkBlackObject,
    MilkGrayObject,
    MilkRedObject,
    OatsMilkObject,
    ChocolateMilkObject,
    LemonObject,
    CanObject,
)

MAX_OBJ_NUMS = 3
class CleanUpMedium(SingleArmEnv):
    """
    This class corresponds to the cleaning-up task for a single robot arm.

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

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

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
        table_friction=(0.6, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
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
        left_bin_obj_ids=None,
        right_bin_obj_ids=None,
        left_mat_obj_ids=None,
        right_mat_obj_ids=None,
        left_mat_center=None,
        right_mat_center=None,
	    num_objs=3,
        table_class_name=None,
    ):
        self.num_pnp_objs = 2
        self.num_push_objs = 1
        assert self.num_pnp_objs + self.num_push_objs == num_objs

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.num_objs = num_objs
        self.objs_idx = [o for o in range(self.num_objs)]
        if left_bin_obj_ids is not None:
            self.objs_idx = sorted(left_bin_obj_ids + right_bin_obj_ids + left_mat_obj_ids + right_mat_obj_ids)
            assert len(self.objs_idx) == self.num_objs
            for _obj_ids_list in [left_bin_obj_ids, right_bin_obj_ids, left_mat_obj_ids, right_mat_obj_ids]:
                for _id in range(len(_obj_ids_list)):
                    for _obj_id in range(len(self.objs_idx)):
                        if self.objs_idx[_obj_id] == _obj_ids_list[_id]:
                            _obj_ids_list[_id] = _obj_id
                            break
        self.left_bin_obj_ids = left_bin_obj_ids
        self.right_bin_obj_ids = right_bin_obj_ids
        self.left_mat_obj_ids = left_mat_obj_ids
        self.right_mat_obj_ids = right_mat_obj_ids

        self.left_mat_center = left_mat_center
        self.right_mat_center = right_mat_center
        self.table_class_name = table_class_name

        self.eef_bounds = np.array([
                [-0.28, -0.32, 0.80],
                [0.135, 0.32, 1.05]
            ])

        self.data_eef_bounds = np.array([
            [-0.26, -0.31, 0.80],
            [0.13, 0.31, 1.05]
        ])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
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
        if self._check_success():
            return 1.0
        return 0.0

    def _load_objects(self):
        pass

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self.table_class_name(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self._load_objects()

        self._get_placement_initializer()

        mujoco_objects = self.objs

        # Create placement initializer
        # if self.placement_initializer is not None:
        #     self.placement_initializer.reset()
        #     self.placement_initializer.add_objects(cubes)
        # else:


        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )

    def _get_placement_initializer(self):
        pass

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.table_body_id = self.sim.model.body_name2id("table")

        self.pnp_obj_body_ids, self.push_obj_body_ids, self.obj_body_ids = [], [], []
        for i in range(self.num_objs):
            obj = self.objs[i]
            id = self.sim.model.body_name2id(obj.root_body)
            self.obj_body_ids.append(id)

        for pnp_obj in self.pnp_objs:
            id = self.sim.model.body_name2id(pnp_obj.root_body)
            self.pnp_obj_body_ids.append(id)

        for push_obj in self.push_objs:
            id = self.sim.model.body_name2id(push_obj.root_body)
            self.push_obj_body_ids.append(id)

        self.bins_body_id = [self.sim.model.body_name2id("bin0"), self.sim.model.body_name2id("bin1")]

    @property
    def obj_positions(self):
        _obj_positions = [
            self.sim.data.body_xpos[self.obj_body_ids[i]].copy()
            for i in range(self.num_objs)
        ]
        return _obj_positions

    @property
    def obj_quats(self):
        _obj_quats = [
            convert_quat(
                np.array(self.sim.data.body_xquat[self.obj_body_ids[i]]), to="xyzw"
            )
            for i in range(self.num_objs)
        ]
        return _obj_quats

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

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            @sensor(modality=modality)
            def obj_pos(obs_cache):
                return np.array(self.obj_positions).flatten()

            @sensor(modality=modality)
            def obj_quat(obs_cache):
                return np.array(self.obj_quats).flatten()

            @sensor(modality=modality)
            def obj_ind(obs_cache):
                assert self.num_objs <= MAX_OBJ_NUMS
                objs_ind = np.zeros((self.num_objs, MAX_OBJ_NUMS))
                for i in range(self.num_objs):
                    objs_ind[i][self.objs_idx[i]] = 1
                return objs_ind.flatten()

            @sensor(modality=modality)
            def object_centric(obs_cache):
                _objs_pos = np.array(self.obj_positions).copy().reshape(-1, 3)
                nobject = _objs_pos.shape[0]
                gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
                gripper_to_objects_pos = np.stack([_obj_pos - gripper_site_pos for _obj_pos in _objs_pos])
                assert len(gripper_to_objects_pos) == nobject
                relative_objs_pos = []
                for i in range(nobject):
                    for j in range(i + 1, nobject):
                        relative_objs_pos.append(_objs_pos[j] - _objs_pos[i])
                assert len(relative_objs_pos) == (nobject - 1) * nobject / 2
                relative_objs_pos = np.array(relative_objs_pos)
                return np.concatenate([gripper_to_objects_pos.flatten(), relative_objs_pos.flatten()])

            # @sensor(modality=modality)
            # def obj_state(obs_cache):
            #     obj_pos = np.array(self.obj_positions).flatten()
            #     obj_quat = np.array(self.obj_quats).flatten()
            #     return np.concatenate([obj_pos, obj_quat])

            sensors = [obj_pos, obj_quat, object_centric, obj_ind]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def in_bin(self, obj_pos, bin_id):
        # bins_pos = [np.array(self.sim.data.body_xpos[bin_body_id]) for bin_body_id in self.bins_body_id]
        bin_pos = np.array(self.sim.data.body_xpos[self.bins_body_id[bin_id]])
        if abs(obj_pos[0] - bin_pos[0]) < 0.12 \
                and abs(obj_pos[1] - bin_pos[1]) < 0.07 \
                and obj_pos[2] < self.table_offset[2] + 0.09:
            return True
        return False

    def _all_in_bins(self):
        for i in self.left_bin_obj_ids:
            obj_pos = self.sim.data.body_xpos[self.obj_body_ids[i]]
            if not self.in_bin(obj_pos, 0):
                return False
        for i in self.right_bin_obj_ids:
            obj_pos = self.sim.data.body_xpos[self.obj_body_ids[i]]
            if not self.in_bin(obj_pos, 1):
                return False
        return True

    def _all_on_mats(self):
        for i in self.left_mat_obj_ids:
            obj_pos = self.sim.data.body_xpos[self.obj_body_ids[i]]
            target_pos_xy = self.left_mat_center
            d_push = np.linalg.norm(obj_pos[:2] - target_pos_xy)
            if d_push > 0.07:
                return False

        for i in self.right_mat_obj_ids:
            obj_pos = self.sim.data.body_xpos[self.obj_body_ids[i]]
            target_pos_xy = self.right_mat_center
            d_push = np.linalg.norm(obj_pos[:2] - target_pos_xy)
            if d_push > 0.07:
                return False
        return True

    def _check_success(self):
        if not self._all_in_bins():
            return False
        if not self._all_on_mats():
            return False
        return True

    def _get_env_info(self, action):
        env_info = {}
        # env_info['success_pnp'] = self._check_success_pnp()
        # env_info['success_push'] = self._check_success_push()
        env_info['success'] = self._check_success()

        return env_info

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

        # if vis_settings["grippers"]:
        #     self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)

    @property
    def _has_gripper_contact(self):
        return np.linalg.norm(self.robots[0].ee_force) > 20

    def _get_skill_info(self):
        pnp_obj_pos_list = [self.sim.data.body_xpos[self.pnp_obj_body_ids[i]].copy() for i in range(self.num_pnp_objs)]
        push_obj_pos_list = [self.sim.data.body_xpos[self.push_obj_body_ids[i]].copy() for i in range(self.num_push_objs)]

        pos_info = {}

        pos_info['grasp'] = pnp_obj_pos_list  # grasp target positions
        pos_info['push'] = push_obj_pos_list  # push target positions
        pos_info['reach'] = None
        # pos_info['place'] = None
        pos_info['place'] = []
        for bin_id in range(2):
            _place_pos = [0., 0., 0.95]
            _place_pos[:2] = np.array(self.sim.data.body_xpos[self.bins_body_id[bin_id]])[:2]
            pos_info['place'].append(_place_pos)

        info = {}
        for k in pos_info:
            info[k + '_pos'] = pos_info[k]

        info['grasped_obj'] = [
            self._check_grasp(gripper=self.robots[0].gripper, object_geoms=obj) for obj in self.objs]

        info['gripper_contact'] = self._has_gripper_contact

        return info

class CleanUpMediumSmallInit(CleanUpMedium):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(**kwargs)

    def _load_objects(self):
        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }

        obj_texture_lst = ["WoodBlue", "WoodGreen", "WoodRed"]
        obj_material_list = []
        for i in range(len(obj_texture_lst)):
            obj_material_list.append(
                CustomMaterial(
                    texture=obj_texture_lst[i],
                    tex_name="obj{}_tex".format(i),
                    mat_name="obj{}_mat".format(i),
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                ))

        obj_size_list = [
            np.array([0.04, 0.022, 0.033]) * 0.7,
            np.array([0.03, 0.03, 0.04]) * 0.7,
            np.array([0.0350, 0.0425, 0.02]) * 1.2
        ]
        assert len(obj_texture_lst) == len(obj_size_list) == len(obj_material_list) == MAX_OBJ_NUMS
        self.pnp_objs, self.push_objs, self.objs = [], [], []
        for i in self.objs_idx:
            if self.num_objs > 1:
                color = 0.25 + 0.75 * i / (self.num_objs - 1)
            else:
                color = 1.0
            obj_size = obj_size_list[i]
            obj = BoxObject(
                name="obj_{}".format(i),
                size_min=obj_size,
                size_max=obj_size,
                rgba=[color, 0, 0, 1],
                material=obj_material_list[i],
                # solimp=[0.998, 0.998, 0.001],
            )
            self.objs.append(obj)
            if i < self.num_pnp_objs:
                self.pnp_objs.append(obj)
            else:
                self.push_objs.append(obj)

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler_0",
                mujoco_objects=self.objs[0],
                x_range=[-0.17, -0.07],
                y_range=[-0.15, -0.05],
                rotation=(-np.pi / 4., np.pi / 4.),
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler_1",
                mujoco_objects=self.objs[1],
                x_range=[-0.05, 0.05],
                y_range=[-0.15, -0.05],
                rotation=(-np.pi / 4., np.pi / 4.),
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler_2",
                mujoco_objects=self.objs[2],
                x_range=[-0.12, 0.05],
                y_range=[-0.03, 0.09],
                rotation=(-np.pi / 4., np.pi / 4.),
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))

class CleanUpMediumSmallInitA(CleanUpMediumSmallInit):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(table_class_name=TableArenaDoubleBinsA,
                         left_mat_center=np.array([-0.135, 0.29]),
                         right_mat_center=np.array([0.075, 0.29]),
                         **kwargs)

class CleanUpMediumSmallInitB(CleanUpMediumSmallInit):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(table_class_name=TableArenaDoubleBinsB,
                         left_mat_center=np.array([-0.135, -0.29]),
                         right_mat_center=np.array([0.075, -0.29]),
                         **kwargs)

class CleanUpMediumSmallInitC(CleanUpMediumSmallInit):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(table_class_name=TableArenaDoubleBinsC,
                         left_mat_center=np.array([0.14, -0.1]),
                         right_mat_center=np.array([0.14, 0.1]),
                         **kwargs)

class CleanUpMediumSmallInitD(CleanUpMediumSmallInit):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(table_class_name=TableArenaDoubleBinsD,
                         left_mat_center=np.array([0.14, -0.1]),
                         right_mat_center=np.array([0.14, 0.1]),
                         **kwargs)

class CleanUpMediumSmallInitDa(CleanUpMediumSmallInit):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(table_class_name=TableArenaDoubleBinsDa,
                         left_mat_center=np.array([0.14, -0.1]),
                         right_mat_center=np.array([0.14, 0.1]),
                         **kwargs)

class CleanUpMediumSmallInitE(CleanUpMediumSmallInit):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(table_class_name=TableArenaDoubleBinsE,
                         left_mat_center=np.array([0.14, -0.1]),
                         right_mat_center=np.array([0.14, 0.1]),
                         **kwargs)

class CleanUpMediumSmallInitA1(CleanUpMediumSmallInitA):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

class CleanUpMediumSmallInitAReal1(CleanUpMediumSmallInitA):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

    def _load_objects(self):
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }

        self.pnp_objs = self.push_objs = self.objs = []
        self.objs.append(CanObject(name="can"))

        obj_size_list = [
            np.array([0.04, 0.022, 0.033]) * 0.7,
            np.array([0.0375, 0.041, 0.02]) * 1.2
        ]

        obj_material_list = [
            CustomMaterial(
                texture="Cheese",
                tex_name="pnpobj_tex",
                mat_name="pnpobj_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib, ),
            CustomMaterial(
                texture="Cereal",
                tex_name="pushobj_tex",
                mat_name="pushobj_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib, )
        ]
        for i in range(1, 3):
            if self.num_objs > 1:
                color = 0.25 + 0.75 * i / (self.num_objs - 1)
            else:
                color = 1.0
            obj_size = obj_size_list[i-1]
            obj = BoxObject(
                name="obj_{}".format(i),
                size_min=obj_size,
                size_max=obj_size,
                rgba=[color, 0, 0, 1],
                material=obj_material_list[i-1],
                # solimp=[0.998, 0.998, 0.001],
            )
            self.objs.append(obj)

class CleanUpMediumSmallInitB1(CleanUpMediumSmallInitB):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

class CleanUpMediumSmallInitBReal1(CleanUpMediumSmallInitB):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

    def _load_objects(self):
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }

        self.pnp_objs = self.push_objs = self.objs = []
        self.objs.append(ColaObject(name="cola"))

        obj_size_list = [
            np.array([0.038, 0.025, 0.027]) * 0.7,
            np.array([0.0325, 0.045, 0.02]) * 1.2
        ]

        obj_material_list = [
            CustomMaterial(
                texture="Pancake",
                tex_name="pnpobj_tex",
                mat_name="pnpobj_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib, ),
            CustomMaterial(
                texture="Spam",
                tex_name="pushobj_tex",
                mat_name="pushobj_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib, )
        ]
        for i in range(1, 3):
            if self.num_objs > 1:
                color = 0.25 + 0.75 * i / (self.num_objs - 1)
            else:
                color = 1.0
            obj_size = obj_size_list[i-1]
            obj = BoxObject(
                name="obj_{}".format(i),
                size_min=obj_size,
                size_max=obj_size,
                rgba=[color, 0, 0, 1],
                material=obj_material_list[i-1],
                # solimp=[0.998, 0.998, 0.001],
            )
            self.objs.append(obj)

class CleanUpMediumSmallInitC1(CleanUpMediumSmallInitC):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

class CleanUpMediumSmallInitCReal1(CleanUpMediumSmallInitC):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

    def _load_objects(self):
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }

        self.pnp_objs = self.push_objs = self.objs = []
        for obj_cls, obj_name, obj_scale in zip(
                (AppleJuiceObject, StrawberryJuiceObject), ["apple_juice", "strawberry_juice"], [0.8, 1.0]
        ):
            obj = obj_cls(name=obj_name, scale=obj_scale)
            self.objs.append(obj)

        if self.num_objs > 1:
            color = 0.25 + 0.75 * 2 / (self.num_objs - 1)
        else:
            color = 1.0
        obj_size = np.array([0.038, 0.04, 0.02]) * 1.2
        obj_material = CustomMaterial(
            texture="Jello",
            tex_name="pushobj_tex",
            mat_name="pushobj_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        obj = BoxObject(
            name="obj_{}".format(2),
            size_min=obj_size,
            size_max=obj_size,
            rgba=[color, 0, 0, 1],
            material=obj_material,
        )
        self.objs.append(obj)

class CleanUpMediumSmallInitD1(CleanUpMediumSmallInitD):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

class CleanUpMediumSmallInitD2(CleanUpMediumSmallInitD):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0, 1],
                         right_bin_obj_ids=[],
                         left_mat_obj_ids=[],
                         right_mat_obj_ids=[2],
                         **kwargs)

class CleanUpMediumSmallInitD3(CleanUpMediumSmallInitD):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[],
                         right_bin_obj_ids=[0, 1],
                         left_mat_obj_ids=[],
                         right_mat_obj_ids=[2],
                         **kwargs)

class CleanUpMediumSmallInitDReal1(CleanUpMediumSmallInitD):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

    def _load_objects(self):
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }

        self.pnp_objs = self.push_objs = self.objs = []
        for obj_cls, obj_name in zip(
                (CanObject, MilkBlueObject), ["can", "milk"]
        ):
            obj = obj_cls(name=obj_name, scale=1.0)
            self.objs.append(obj)

        if self.num_objs > 1:
            color = 0.25 + 0.75 * 2 / (self.num_objs - 1)
        else:
            color = 1.0
        obj_size = np.array([0.0350, 0.0425, 0.02]) * 1.2
        obj_material = CustomMaterial(
            texture="Tea",
            tex_name="pushobj_tex",
            mat_name="pushobj_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        obj = BoxObject(
            name="obj_{}".format(2),
            size_min=obj_size,
            size_max=obj_size,
            rgba=[color, 0, 0, 1],
            material=obj_material,
        )
        self.objs.append(obj)

class CleanUpMediumSmallInitDObjectTrain(CleanUpMediumSmallInitD):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

    def _load_objects(self):
        """
        Loads an xml model, puts it in self.model
        """

        tex_attrib_list = [
            {"type": "cube", },
            {"type": "2d", }
        ]
        mat_attrib_list = [
            {
                "texrepeat": "3 3",
                "specular": "0.4",
                "shininess": "0.1",
            },
            {
                "texrepeat": "5 5",
                "reflectance": "0.7",
                "specular": "0.4",
                "shininess": "0.1",
            }
        ]
        box_material_list = [
            dict(texture="Cereal", tex_name="cereal", mat_name="MatCereal"),
            dict(texture="Spam", tex_name="spam", mat_name="MatSpam"),
            dict(texture="Cheese", tex_name="cheese", mat_name="MatCheese"),
            dict(texture="Jello", tex_name="jello", mat_name="MatJello"),
            dict(texture="Burger", tex_name="burger", mat_name="MatBurger"),
            dict(texture="Bean", tex_name="bean", mat_name="MatBean"),
            dict(texture="Chip", tex_name="chip", mat_name="MatChip"),
            dict(texture="Candy", tex_name="candy", mat_name="MatCandy"),
        ]
        object_list = dict(
            milk=[
                MilkBlueObject,
                MilkGreenObject,
                MilkBlackObject,
                MilkRedObject,
                MilkGrayObject,
                OatsMilkObject,
            ],
            can=[
                 CoffeeObject,
                 LimoncelloObject,
                 JuiceGrayObject,
                 BeerObject,
                 SodaObject,
                 CanObject,
                 TeaObject,
            ]
        )


        object_category_list = ["milk", "can", "box"]
        selected_obj_num = 3
        selected_obj_shape_list = random.choices(object_category_list, k=selected_obj_num)
        self.pnp_objs = self.push_objs = self.objs = []
        for i in range(len(selected_obj_shape_list)):
            if selected_obj_shape_list[i] in ["milk", "can"]:
                obj_category = selected_obj_shape_list[i]
                scale = random.uniform(0.8, 1.0)
                selected_object_idx = random.randint(0, len(object_list[obj_category]) - 1)
                obj = object_list[obj_category][selected_object_idx](name=obj_category + str(i), scale=scale)
            elif selected_obj_shape_list[i] in ["box"]:
                selected_texture_idx = random.randint(0, len(box_material_list) - 1)
                obj_material = CustomMaterial(
                    texture=box_material_list[selected_texture_idx]['texture'],
                    tex_name=box_material_list[selected_texture_idx]['tex_name'],
                    mat_name=box_material_list[selected_texture_idx]['mat_name'],
                    tex_attrib=tex_attrib_list[0],
                    mat_attrib=mat_attrib_list[0],
                )
                obj = BoxObject(
                    name="box_{}".format(i),
                    size_min=np.array([0.025, 0.014, 0.02]),
                    size_max=np.array([0.05, 0.05, 0.03]),
                    rgba=[1, 0, 0, 1],
                    material=obj_material,
                )
            else:
                raise NotImplementedError
            self.objs.append(obj)

class CleanUpMediumSmallInitDObjectTest(CleanUpMediumSmallInitD):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

    def _load_objects(self):
        """
        Loads an xml model, puts it in self.model
        """

        tex_attrib_list = [
            {"type": "cube", },
            {"type": "2d", }
        ]
        mat_attrib_list = [
            {
                "texrepeat": "3 3",
                "specular": "0.4",
                "shininess": "0.1",
            },
            {
                "texrepeat": "5 5",
                "reflectance": "0.7",
                "specular": "0.4",
                "shininess": "0.1",
            }
        ]
        box_material_list = [
            dict(texture="Jello", tex_name="jello", mat_name="MatJello"),
        ]
        object_list = dict(
            milk=[
                 MilkBlueObject,
            ],
            can=[
                 TeaObject,
            ]
        )


        selected_obj_shape_list = ["milk", "can", "box"]
        # selected_obj_num = 3
        # selected_obj_shape_list = random.choices(object_category_list, k=selected_obj_num)
        self.pnp_objs = self.push_objs = self.objs = []
        for i in range(len(selected_obj_shape_list)):
            if selected_obj_shape_list[i] in ["milk", "can"]:
                obj_category = selected_obj_shape_list[i]
                scale = 0.9
                selected_object_idx = random.randint(0, len(object_list[obj_category]) - 1)
                obj = object_list[obj_category][selected_object_idx](name=obj_category + str(i), scale=scale)
            elif selected_obj_shape_list[i] in ["box"]:
                selected_texture_idx = random.randint(0, len(box_material_list) - 1)
                obj_material = CustomMaterial(
                    texture=box_material_list[selected_texture_idx]['texture'],
                    tex_name=box_material_list[selected_texture_idx]['tex_name'],
                    mat_name=box_material_list[selected_texture_idx]['mat_name'],
                    tex_attrib=tex_attrib_list[0],
                    mat_attrib=mat_attrib_list[0],
                )
                obj = BoxObject(
                    name="box_{}".format(i),
                    size_min=np.array([0.0350, 0.0425, 0.02]) * 1.2,
                    size_max=np.array([0.0350, 0.0425, 0.02]) * 1.2,
                    rgba=[1, 0, 0, 1],
                    material=obj_material,
                )
            else:
                raise NotImplementedError
            self.objs.append(obj)

class CleanUpMediumSmallInitDa1(CleanUpMediumSmallInitDa):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0, 1],
                         right_bin_obj_ids=[],
                         left_mat_obj_ids=[],
                         right_mat_obj_ids=[2],
                         **kwargs)

class CleanUpMediumSmallInitE1(CleanUpMediumSmallInitE):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[],
                         right_bin_obj_ids=[0, 1],
                         left_mat_obj_ids=[],
                         right_mat_obj_ids=[2],
                         **kwargs)

class CleanUpMediumMediumInit(CleanUpMedium):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        obj_idx = [0, 1, 2]
        random.shuffle(obj_idx)
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler_{}".format(obj_idx[0]),
                mujoco_objects=self.objs[obj_idx[0]],
                x_range=[-0.19, -0.07],
                y_range=[-0.13, -0.03],
                rotation=(-np.pi / 4., np.pi / 4.),
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler_{}".format(obj_idx[1]),
                mujoco_objects=self.objs[obj_idx[1]],
                x_range=[-0.05, 0.07],
                y_range=[-0.14, -0.02],
                rotation=(-np.pi / 4., np.pi / 4.),
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler_{}".format(obj_idx[2]),
                mujoco_objects=self.objs[obj_idx[2]],
                x_range=[-0.12, 0.05],
                y_range=[0.01, 0.13],
                rotation=(-np.pi / 4., np.pi / 4.),
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))

class CleanUpMediumLargeInit(CleanUpMedium):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(left_bin_obj_ids=[0],
                         right_bin_obj_ids=[1],
                         left_mat_obj_ids=[2],
                         right_mat_obj_ids=[],
                         **kwargs)

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.objs,
                x_range=[-0.18, 0.10],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))
