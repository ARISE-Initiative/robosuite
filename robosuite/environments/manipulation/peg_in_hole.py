from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import BlockArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from robosuite.models.objects.xml_objects import PegObject

DEFAULT_PEG_IN_HOLE_CONFIG = {
    'large_hole': False,
    'd_weight': 1,
    't_weight': 5,
    'cos_weight': 1,
    'scale_by_cos': True,
    'scale_by_d': True,
    'cos_tanh_mult': 3.0,
    'd_tanh_mult': 15.0,
    't_tanh_mult': 7.5,
    'limit_init_ori': True,
    'lift_pos_offset': 0.3,
}

class PegInHole(SingleArmEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

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
        table_friction=(1., 5e-3, 1e-4),
        table_offset=(0, 0, 0.8),
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
        task_config=None,
        skill_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # Get config
        self.task_config = DEFAULT_PEG_IN_HOLE_CONFIG.copy()
        if task_config is not None:
            self.task_config.update(task_config)

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
            skill_config=skill_config,
        )

    def reward(self, action):
        r_reach, r_grasp, r_align, r_insert = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_grasp, r_align, r_insert)
        else:
            reward = r_insert

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def staged_rewards(self):
        reach_mult = 0.10
        grasp_mult = 0.15
        align_mult = 0.85

        gripper_peg_dist = self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.peg.important_sites["tip"],
            target_type="site",
            return_distance=True,
        )
        r_reach = (1 - np.tanh(10.0 * gripper_peg_dist)) * reach_mult

        r_grasp = int(self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.peg.contact_geoms if g == 'peg_tip_geom'])
        ) * grasp_mult

        # Orientation reward
        r_align = 0
        if r_grasp > 0.:
            t, d, cos = self._compute_orientation()

            d_w = self.task_config['d_weight']
            t_w = self.task_config['t_weight']
            cos_w = self.task_config['cos_weight']

            cos_tanh_mult = self.task_config['cos_tanh_mult']
            d_tanh_mult = self.task_config['d_tanh_mult']
            t_tanh_mult = self.task_config['t_tanh_mult']

            cos_dist = 1 - (cos + 1) / 2
            cos_rew = 1 - np.tanh(cos_tanh_mult * cos_dist)
            d_rew = 1 - np.tanh(d_tanh_mult * d)
            t_rew = 1 - np.tanh(t_tanh_mult * np.abs(t))

            scale_by_cos = self.task_config['scale_by_cos']
            scale_by_d = self.task_config['scale_by_d']
            if scale_by_cos:
                t_rew *= cos_rew
            if scale_by_d:
                t_rew *= d_rew
            if scale_by_cos:
                d_rew *= cos_rew

            r_align_sum = (cos_w * cos_rew) + (d_w * d_rew) + (t_w * t_rew)
            r_align_norm = r_align_sum / float(d_w + t_w + cos_w) # normalize sum to be between 0 and 1

            r_align = r_grasp + r_align_norm * (align_mult - grasp_mult)

        r_insert = self._check_success()

        return r_reach, r_grasp, r_align, r_insert

    def _get_env_info(self, action):
        info = super()._get_env_info(action)
        r_reach, r_grasp, r_align, r_insert = self.staged_rewards()
        t, d, cos = self._compute_orientation()
        info.update({
            'r_reach': r_reach / 0.10,
            'r_grasp': r_grasp / 0.15,
            'r_align': r_align / 0.85,
            'success': r_insert,
            'align_t_abs': np.abs(t),
            'align_d': d,
            'align_cos': cos,
        })
        return info

    def _get_skill_info(self):
        peg_tip_pos = self.sim.data.get_site_xpos(self.peg.important_sites["tip"])
        hole_pos = np.array(self.sim.data.body_xpos[self.block_body_id])
        lift_pos = hole_pos + [0.0, self.task_config['lift_pos_offset'], 0.0]

        pos_info = {}

        pos_info['grasp'] = [peg_tip_pos] # grasp target positions
        pos_info['push'] = [] # push target positions
        pos_info['reach'] = [lift_pos] # reach target positions

        info = {}
        for k in pos_info:
            info[k + '_pos'] = pos_info[k]

        return info

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        if self.task_config['large_hole']:
            xml_path="arenas/table_arena_block_large_hole.xml"
        else:
            xml_path="arenas/table_arena_block.xml"

        # load model for table top workspace
        mujoco_arena = BlockArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            xml=xml_path,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.peg = PegObject(name='peg')
        objs = [self.peg]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(objs)
        else:
            if self.task_config['limit_init_ori']:
                rotation =  [-np.pi/2 - np.pi*3/8, -np.pi/2 + np.pi*3/8]
            else:
                rotation = None

            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=objs,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation=rotation,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objs,
        )

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Additional object references from this env
        self.block_body_id = self.sim.model.body_name2id("block")
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)

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

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:

            `'robot-state'`: contains robot-centric information.

            `'object-state'`: requires @self.use_object_obs to be True. Contains object-centric information.

            `'image'`: requires @self.use_camera_obs to be True. Contains a rendered frame from the simulation.

            `'depth'`: requires @self.use_camera_obs and @self.camera_depth to be True.
            Contains a rendered depth map from the simulation

        Returns:
            OrderedDict: Observations from the environment
        """
        di = super()._get_observation()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix
            pr = self.robots[0].robot_model.naming_prefix

            # position and rotation of the first cube
            peg_pos = np.array(self.sim.data.body_xpos[self.peg_body_id])
            peg_quat = convert_quat(
                np.array(self.sim.data.body_xquat[self.peg_body_id]), to="xyzw"
            )
            di["peg_pos"] = peg_pos
            di["peg_quat"] = peg_quat

            # relative positions between gripper and cubes
            di[pr + "gripper_to_peg"] = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.peg.important_sites["tip"],
                target_type="site",
                return_distance=False,
            )

            di["hole_pos"] = np.array(self.sim.data.body_xpos[self.block_body_id])

            # Relative orientation parameters
            t, d, cos = self._compute_orientation()
            di["angle"] = cos
            di["t"] = t
            di["d"] = d

            di["object-state"] = np.concatenate(
                [
                    peg_pos,
                    peg_quat,
                    di[pr + "gripper_to_peg"],
                    di["hole_pos"],
                    [di["angle"]],
                    [di["t"]],
                    [di["d"]],
                ]
            )

        return di

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
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.peg)

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        t, d, cos = self._compute_orientation()

        return d < 0.05 and -0.05 <= t <= 0.05 and cos > 0.95

    def _compute_orientation(self):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.

        Returns:
            3-tuple:

                - (float): parallel distance
                - (float): perpendicular distance
                - (float): angle
        """
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]

        hole_pos = self.sim.data.body_xpos[self.block_body_id]
        hole_mat = self.sim.data.body_xmat[self.block_body_id]
        hole_mat.shape = (3, 3)

        v = peg_mat @ np.array([1, 0, 0]) #np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center = hole_pos #+ hole_mat @ np.array([0.1, 0, 0])

        hole_normal = hole_mat @ np.array([1, 0, 0]) #np.array([0, 0, 1])

        t = (center - peg_pos) @ hole_normal / (np.linalg.norm(hole_normal) ** 2)
        d = np.linalg.norm(np.cross(hole_normal, peg_pos - center)) / np.linalg.norm(hole_normal)

        return (
            t,
            d,
            np.dot(hole_normal, v) / np.linalg.norm(hole_normal) / np.linalg.norm(v),
        )