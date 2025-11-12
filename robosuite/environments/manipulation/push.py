from collections import OrderedDict
import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils import RandomizationError
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial


DEFAULT_PUSH_CONFIG = {
    'use_push_rew': True,
    'shaped_push_rew': True,
    'push_scale_fac': 5.0,
    'num_push_objs': 1,
}


class Push(SingleArmEnv):
    """
    Push-only environment: push a single cube to a fixed gray circular target on the table.
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
        reward_shaping=True,
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
        skill_config=None,
    ):
        # Store environment-specific parameters locally
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self.task_config = DEFAULT_PUSH_CONFIG.copy()

        # Call parent constructor with standard, expected arguments only
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

    # -------------------------
    # Reward functions
    # -------------------------
    def reward(self, action):
        _, reward = self.reward_infos()
        return reward

    def reward_infos(self):
        rew_push = 0
        for i in range(self.task_config['num_push_objs']):
            r_reach, r_push, _ = self.push_staged_rewards(obj_id=i)
            if self.reward_shaping:
                rew_push += r_reach + r_push
            else:
                rew_push += r_push

        if self.reward_scale is not None:
            rew_push *= self.reward_scale
        return {}, rew_push

    def push_staged_rewards(self, obj_id=0):
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        obj_pos = self.sim.data.body_xpos[self.push_obj_body_ids[obj_id]]
        # Fixed target position (x_table+0.15, y_table-0.15)
        target_xy = self.table_offset[:2] + np.array([0.15, -0.15]) 

        d_push = np.linalg.norm(obj_pos[:2] - target_xy)
        r_push = 1 - np.tanh(self.task_config['push_scale_fac'] * d_push)

        # reach reward
        th = [0.08, 0.08, 0.04]
        d_reach = np.sum(np.clip(np.abs(gripper_pos - obj_pos) - th, 0, None))
        r_reach = (1 - np.tanh(10.0 * d_reach)) * 0.25

        return r_reach, r_push, d_push

    def _check_success(self):
        for i in range(self.task_config['num_push_objs']):
            _, _, d_push = self.push_staged_rewards(obj_id=i)
            # 0.10m threshold for success
            if d_push > 0.10: 
                return False
        return True
    
    # -------------------------
    # Model / arena
    # -------------------------
    def _load_model(self):
        super()._load_model()

        # Set robot position
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Create table arena
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(1., 5e-3, 1e-4)
        )
        arena.set_origin([0, 0, 0])

        # 1. Define the custom push target object (as a static, flat box)
        target_size = [0.07, 0.07, 0.001]
        target_pos = self.table_offset + np.array([0.15, -0.15, 0.0005])
        
        self.target_obj = BoxObject(
            name="push_target_obj",
            size_min=target_size,
            size_max=target_size,
            rgba=[0.5, 0.5, 0.5, 1], # Gray color
            density=0,               # Make it massless
            joints=None,             # No joints, fixed to the world
        )
        
        # --- THE CORRECTED LINES: FINAL FIX FOR INITIAL POSITIONING ---
        # Set its fixed position directly on the object's initial attribute properties.
        self.target_obj.initial_pos = target_pos
        self.target_obj.initial_quat = np.array([1, 0, 0, 0])
        # ----------------------------------------------------------------------

        # 2. Define the movable push cube
        mat = CustomMaterial(
            texture="WoodLight",
            tex_name="push_cube_tex",
            mat_name="push_cube_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        self.push_objs = []
        for i in range(self.task_config['num_push_objs']):
            size = np.array([0.035, 0.0425, 0.0125])
            obj = BoxObject(
                name=f"obj_push_{i}",
                size_min=size,
                size_max=size,
                rgba=[0, 1, 0, 1],
                material=mat,
            )
            self.push_objs.append(obj)

        # 3. Combine all MujocoObject instances (movable + static target)
        all_objects = self.push_objs + [self.target_obj] 

        # 4. Placement sampler for the movable cube(s) ONLY
        if self.placement_initializer is None:
            self.placement_initializer = UniformRandomSampler(
                name="PushSampler",
                mujoco_objects=self.push_objs, # ONLY the movable cube(s)
                x_range=[0.0, 0.16],
                y_range=[-0.16, 0.16],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        else:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.push_objs)

        # 5. Create ManipulationTask with ALL objects
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_objects,
        )

    # -------------------------
    # References and reset
    # -------------------------
    def _get_reference(self):
        super()._get_reference()
        self.table_body_id = self.sim.model.body_name2id("table")
        self.push_obj_body_ids = []
        for obj in self.push_objs:
            bid = self.sim.model.body_name2id(obj.root_body)
            self.push_obj_body_ids.append(bid)
        
        self.target_body_id = self.sim.model.body_name2id(self.target_obj.root_body)


    def _reset_internal(self):
        super()._reset_internal()
        # Only reset position for the movable cubes using the initializer
        while True:
            try:
                placements = self.placement_initializer.sample()
                break
            except RandomizationError:
                continue
        for obj_pos, obj_quat, obj in placements.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    # -------------------------
    # Observations
    # -------------------------
    def _get_observation(self):
        di = super()._get_observation()
        
        pos = np.array(self.obj_positions).flatten()
        quat = np.array(self.obj_quats).flatten()
        di["obj_pos"] = pos
        di["obj_quat"] = quat
        di["object-state"] = np.concatenate([pos, quat])
        
        return di

    @property
    def obj_positions(self):
        return [self.sim.data.body_xpos[bid].copy() for bid in self.push_obj_body_ids]

    @property
    def obj_quats(self):
        return [convert_quat(self.sim.data.body_xquat[bid], to="xyzw") for bid in self.push_obj_body_ids]