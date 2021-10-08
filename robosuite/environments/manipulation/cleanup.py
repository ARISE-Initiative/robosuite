from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils import RandomizationError

DEFAULT_CLEANUP_CONFIG = {
    'use_pnp_rew': True,
    'use_push_rew': True,
    'rew_type': 'sum',
    'num_pnp_objs': 1,
    'num_push_objs': 1,
    'shaped_push_rew': False,
    'push_scale_fac': 5.0,
}


class Cleanup(SingleArmEnv):
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
        self.task_config = DEFAULT_CLEANUP_CONFIG.copy()
        if task_config is not None:
            assert all([k in self.task_config for k in task_config])
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
        _, _, reward = self.reward_infos()
        return reward

    def reward_infos(self):
        rew_pnp = 0
        partial_rew_pnp = 0
        num_pnp_success = 0
        for i in range(self.task_config['num_pnp_objs']):
            r, g, l, h, b = self.pnp_staged_rewards(obj_id=i)

            if b == 1.0:
                rew_pnp += 1.0
                num_pnp_success += 1
            elif b == 0.0:
                partial_rew_pnp = max(partial_rew_pnp, max(r, g, l, h))
            else:
                raise ValueError

        if self.reward_shaping:
            rew_pnp += partial_rew_pnp

        rew_push = 0
        for i in range(self.task_config['num_push_objs']):
            r, p, d = self.push_staged_rewards(obj_id=i)
            rew_push += p
            if self.task_config['shaped_push_rew']:
                rew_push += r

        if self.task_config['use_pnp_rew'] and self.task_config['use_push_rew']:
            if self.task_config['rew_type'] == 'sum':
                reward = rew_pnp + rew_push
            elif self.task_config['rew_type'] == 'step':
                pnp_success = (num_pnp_success == self.task_config['num_pnp_objs'])
                reward = rew_pnp + float(pnp_success) * rew_push
            else:
                raise ValueError
        elif self.task_config['use_pnp_rew']:
            reward = rew_pnp
        elif self.task_config['use_push_rew']:
            reward = rew_push
        else:
            raise ValueError

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return rew_pnp, rew_push, reward

    def pnp_staged_rewards(self, obj_id=0):
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        obj_pos = self.sim.data.body_xpos[self.pnp_obj_body_ids[obj_id]]
        obj = self.pnp_objs[obj_id]

        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - obj_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult

        # grasping reward
        r_grasp = int(self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=obj)
        ) * grasp_mult

        r_lift = 0.
        r_hover = 0.
        if r_grasp > 0.:
            table_pos = np.array(self.sim.data.body_xpos[self.table_body_id])
            z_target = table_pos[2] + 0.15
            obj_z = obj_pos[2]
            z_dist = np.maximum(z_target - obj_z, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * z_dist)) * (
                    lift_mult - grasp_mult
            )

            bin_xy = np.array(self.sim.data.body_xpos[self.bin_body_id])[:2]
            obj_xy = obj_pos[:2]
            dist = np.linalg.norm(bin_xy - obj_xy)
            r_hover = r_lift + (1 - np.tanh(10.0 * dist)) * (
                    hover_mult - lift_mult
            )

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_bin = self.in_bin(obj_pos)

        return r_reach, r_grasp, r_lift, r_hover, r_bin

    def push_staged_rewards(self, obj_id=0):
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        obj_pos = self.sim.data.body_xpos[self.push_obj_body_ids[obj_id]]
        target_pos_xy = self.table_offset[:2] + np.array([-0.15, 0.15])
        d_push = np.linalg.norm(obj_pos[:2] - target_pos_xy)

        th = [0.08, 0.08, 0.04]
        d_reach = np.sum(
            np.clip(
                np.abs(gripper_site_pos - obj_pos) - th,
                0, None
            )
        )
        r_reach = (1 - np.tanh(10.0 * d_reach)) * 0.25

        r_push = 1 - np.tanh(self.task_config['push_scale_fac'] * d_push)
        return r_reach, r_push, d_push

    def in_bin(self, obj_pos):
        bin_pos = np.array(self.sim.data.body_xpos[self.bin_body_id])
        res = False
        if (
                abs(obj_pos[0] - bin_pos[0]) < 0.10
                and abs(obj_pos[1] - bin_pos[1]) < 0.15
                and obj_pos[2] < self.table_offset[2] + 0.05
        ):
            res = True
        return res

    def _get_env_info(self, action):
        info = super()._get_env_info(action)

        rews = dict(
            r_reach=[],
            r_grasp=[],
            r_lift=[],
            r_hover=[],
            r_bin=[],

            r_reach_push=[],
            r_push=[],
            d_push=[],
        )
        for i in range(self.task_config['num_pnp_objs']):
            r, g, l, h, b = self.pnp_staged_rewards(obj_id=i)
            rews['r_reach'].append(r / 0.1)
            rews['r_grasp'].append(g / 0.35)
            rews['r_lift'].append(l / 0.5)
            rews['r_hover'].append(h / 0.7)
            rews['r_bin'].append(b / 1.0)

        for i in range(self.task_config['num_push_objs']):
            r, p, d = self.push_staged_rewards(obj_id=i)
            rews['r_reach_push'].append(r / 0.25)
            rews['r_push'].append(p)
            rews['d_push'].append(d)

        for k in rews:
            info[k] = np.sum(rews[k])
            if k.startswith('d'):
                info[k + '_min'] = np.min(rews[k])
            else:
                info[k + '_max'] = np.max(rews[k])

        rew_pnp, rew_push, reward = self.reward_infos()
        info['rew_pnp'] = rew_pnp
        info['rew_push'] = rew_push
        info['rew'] = reward

        info['success_pnp'] = self._check_success_pnp()
        info['success_push'] = self._check_success_push()
        info['success'] = self._check_success()

        return info

    def _get_skill_info(self):
        pos_info = dict(
            grasp=[],
            push=[],
            reach=[],
        )

        bin_pos = self.sim.data.body_xpos[self.bin_body_id].copy()
        obj_positions = self.obj_positions
        num_pnp_objs = self.task_config['num_pnp_objs']

        pnp_objs = obj_positions[:num_pnp_objs]
        push_objs = obj_positions[num_pnp_objs:]

        drop_pos = bin_pos + [0, 0, 0.15]

        pos_info['grasp'] += pnp_objs
        pos_info['push'] += push_objs
        pos_info['reach'].append(drop_pos)

        info = {}
        for k in pos_info:
            info[k + '_pos'] = pos_info[k]

        return info
    
    @property
    def obj_positions(self):
        pnp_obj_positions = [
            self.sim.data.body_xpos[self.pnp_obj_body_ids[i]].copy()
            for i in range(self.task_config['num_pnp_objs'])
        ]
        push_obj_positions = [
            self.sim.data.body_xpos[self.push_obj_body_ids[i]].copy()
            for i in range(self.task_config['num_push_objs'])
        ]
        return pnp_obj_positions + push_obj_positions

    @property
    def obj_quats(self):
        pnp_obj_quats = [
            convert_quat(
                np.array(self.sim.data.body_xquat[self.pnp_obj_body_ids[i]]), to="xyzw"
            )
            for i in range(self.task_config['num_pnp_objs'])
        ]
        push_obj_quats = [
            convert_quat(
                np.array(self.sim.data.body_xquat[self.push_obj_body_ids[i]]), to="xyzw"
            )
            for i in range(self.task_config['num_push_objs'])
        ]
        return pnp_obj_quats + push_obj_quats

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
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
            xml="arenas/table_arena_box.xml",
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        pnpmaterial = CustomMaterial(
            texture="Spam",
            tex_name="pnpobj_tex",
            mat_name="pnpobj_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        pushmaterial = CustomMaterial(
            texture="Jello",
            tex_name="pushobj_tex",
            mat_name="pushobj_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.pnp_objs = []
        num_pnp_objs = self.task_config['num_pnp_objs']
        for i in range(num_pnp_objs):
            if num_pnp_objs > 1:
                color = 0.25 + 0.75 * i / (num_pnp_objs - 1)
            else:
                color = 1.0
            pnp_size = np.array([0.04, 0.022, 0.033]) * 0.75
            obj = BoxObject(
                name="obj_pnp_{}".format(i),
                size_min=pnp_size,
                size_max=pnp_size,
                rgba=[color, 0, 0, 1],
                material=pnpmaterial,
            )
            self.pnp_objs.append(obj)

        self.push_objs = []
        num_push_objs = self.task_config['num_push_objs']
        for i in range(num_push_objs):
            if num_push_objs > 1:
                color = 0.25 + 0.75 * i / (num_push_objs - 1)
            else:
                color = 1.0
            push_size = np.array([0.0350, 0.0425, 0.0125]) * 1.20
            obj = BoxObject(
                name="obj_push_{}".format(i),
                size_min=push_size,
                size_max=push_size,
                rgba=[0, color, 0, 1],
                material=pushmaterial,
            )
            self.push_objs.append(obj)

        objs = self.pnp_objs + self.push_objs
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(objs)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=objs,
                x_range=[0.0, 0.16],
                y_range=[-0.16, 0.16],
                rotation=None,
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
        self.table_body_id = self.sim.model.body_name2id("table")

        self.pnp_obj_body_ids = []
        for i in range(self.task_config['num_pnp_objs']):
            obj = self.pnp_objs[i]
            id = self.sim.model.body_name2id(obj.root_body)
            self.pnp_obj_body_ids.append(id)

        self.push_obj_body_ids = []
        for i in range(self.task_config['num_push_objs']):
            obj = self.push_objs[i]
            id = self.sim.model.body_name2id(obj.root_body)
            self.push_obj_body_ids.append(id)

        self.bin_body_id = self.sim.model.body_name2id("bin")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            while True:
                try:
                    object_placements = self.placement_initializer.sample()
                    sample_success = True
                except RandomizationError:
                    sample_success = False

                if sample_success:
                    break

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
            # position and rotation of the first obj
            obj_pos = np.array(self.obj_positions).flatten()
            obj_quat = np.array(self.obj_quats).flatten()
            di["obj_pos"] = obj_pos
            di["obj_quat"] = obj_quat

            di["object-state"] = np.concatenate(
                [
                    obj_pos,
                    obj_quat,
                ]
            )

        return di

    def _check_success_pnp(self):
        for i in range(self.task_config['num_pnp_objs']):
            _, _, _, _, b = self.pnp_staged_rewards(obj_id=i)
            if b < 1:
                return False
        return True

    def _check_success_push(self):
        for i in range(self.task_config['num_push_objs']):
            _, _, d = self.push_staged_rewards(obj_id=i)
            if d > 0.10:
                return False
        return True

    def _check_success(self):
        if self.task_config['use_pnp_rew']:
            if not self._check_success_pnp():
                return False

        if self.task_config['use_push_rew']:
            if not self._check_success_push():
                return False

        return True

    def _get_info_pnp(self, obj_id=0):
        pnp_obj_pos = self.sim.data.body_xpos[self.pnp_obj_body_ids[obj_id]]
        pnp_obj = self.pnp_objs[obj_id]

        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        reach_dist = np.linalg.norm(gripper_site_pos - pnp_obj_pos)
        reached = reach_dist < 0.06

        grasping_cube = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=pnp_obj)
        if grasping_cube:
            grasped = True
        else:
            grasped = False

        bin_pos = np.array(self.sim.data.body_xpos[self.bin_body_id])
        hovering = (abs(pnp_obj_pos[0] - bin_pos[0]) < 0.10 and abs(pnp_obj_pos[1] - bin_pos[1]) < 0.15)

        in_bin = self.in_bin(pnp_obj_pos)

        return reached, grasped, hovering, in_bin

    def _get_info_push(self, obj_id=0):
        push_obj_pos = self.sim.data.body_xpos[self.push_obj_body_ids[obj_id]]

        target_pos_xy = self.table_offset[:2] + np.array([-0.15, 0.15])
        d_push = np.linalg.norm(push_obj_pos[:2] - target_pos_xy)

        pushed = (d_push <= 0.10)

        return pushed