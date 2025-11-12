from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler


class StackSequential(SingleArmEnv):
    """
    Multi-cube stacking environment with sequential deterministic order.
    Example: for 3 cubes, build C <- B <- A (bottom to top).

    - Modular reward for each cube (reach, grasp, lift, stack)
    - Sequential gating: next cube reward only after previous stacked
    - Additive shaping avoids plateaus
    - Optional contact-based stack validation
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
        full_stacking_bonus=0.0,
        skill_config=None,
        num_stack_objs=3,
    ):
        # Table and reward config
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.placement_initializer = placement_initializer
        self.full_stacking_bonus = full_stacking_bonus
        self.num_stack_objs = num_stack_objs

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

        self.task_config = dict(num_stack_objs=num_stack_objs)

    # ------------------------------------------------------------------
    # Reward logic
    # ------------------------------------------------------------------

    def reward(self, action):
        """Top-level reward call."""
        _, _, reward = self.reward_infos()
        return reward

    def reward_infos(self):
        """
        Handles reward computation across all cubes sequentially.
        Each cube contributes reward only after the previous one is successfully stacked.
        """
        total_reward = 0.0
        num_cubes = self.task_config.get('num_stack_objs', 3)
        stacked_success = [False] * num_cubes

        for i in range(num_cubes):
            r_reach, r_grasp, r_lift, r_stack, done = self.staged_stack_reward(obj_id=i)
            reward_i = r_reach + r_grasp + r_lift + r_stack
            total_reward += reward_i
            stacked_success[i] = done
            if not done:
                break

        info = dict(
            r_total=total_reward,
            stacked_success_count=stacked_success.count(True),
            success_all=all(stacked_success),
        )

        # Optional full-stack bonus
        if info["success_all"]:
            total_reward += self.full_stacking_bonus

        return info['stacked_success_count'], total_reward, total_reward

    def staged_stack_reward(self, obj_id=0):
        """
        Computes staged reward for one cube in stacking.
        Stage order: reach → grasp → lift → stack.
        """
        reach_mult = 0.15
        grasp_mult = 0.3
        lift_mult = 0.6
        stack_mult = 1.0

        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        obj_pos = self.sim.data.body_xpos[self.obj_body_ids[obj_id]]
        target_z = self.table_offset[2] + 0.05 + 0.05 * obj_id

        # --- Reach ---
        reach_dist = np.linalg.norm(gripper_site_pos - obj_pos)
        r_reach = (1 - np.tanh(8.0 * reach_dist)) * reach_mult

        # --- Grasp ---
        grasp_success = int(self._check_grasp(self.robots[0].gripper, self.objs[obj_id]))
        r_grasp = grasp_success * grasp_mult

        # --- Lift ---
        r_lift = 0.0
        if grasp_success:
            z_diff = max(target_z - obj_pos[2], 0.)
            r_lift = grasp_mult + (1 - np.tanh(10.0 * z_diff)) * (lift_mult - grasp_mult)

        # --- Stack alignment ---
        r_stack = 0.0
        done = False

        if obj_id > 0:
            below_obj_pos = self.sim.data.body_xpos[self.obj_body_ids[obj_id - 1]]
            xy_dist = np.linalg.norm(obj_pos[:2] - below_obj_pos[:2])
            z_expected = below_obj_pos[2] + 0.05  # cube height
            z_diff = abs(obj_pos[2] - z_expected)
            in_contact = self.check_contact(self.objs[obj_id], self.objs[obj_id - 1])

            if xy_dist < 0.03 and z_diff < 0.02 and in_contact:
                r_stack = stack_mult
                done = True
            else:
                align_xy = (1 - np.tanh(10.0 * xy_dist))
                align_z = np.exp(-30.0 * z_diff)
                r_stack = (align_xy * align_z) * (stack_mult - lift_mult)
        else:
            # first cube: target = table center
            target_xy = self.table_offset[:2]
            xy_dist = np.linalg.norm(obj_pos[:2] - target_xy)
            z_expected = self.table_offset[2] + 0.05
            z_diff = abs(obj_pos[2] - z_expected)

            if xy_dist < 0.03 and z_diff < 0.02:
                done = True
                r_stack = stack_mult
            else:
                align_xy = (1 - np.tanh(10.0 * xy_dist))
                align_z = np.exp(-30.0 * z_diff)
                r_stack = (align_xy * align_z) * (stack_mult - lift_mult)

        reward = r_reach + r_grasp + r_lift + r_stack
        return r_reach, r_grasp, r_lift, r_stack, done

    # ------------------------------------------------------------------
    # Model and setup
    # ------------------------------------------------------------------

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # cube materials
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}

        materials = [
            CustomMaterial(
                texture=f"Wood{i}",
                tex_name=f"wood{i}_tex",
                mat_name=f"wood{i}_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            for i in ["Red", "Green", "Blue", "Yellow"]
        ]

        # cubes (identical size)
        cube_size = [0.025, 0.025, 0.025]
        self.objs = []
        for i in range(self.num_stack_objs):
            color = np.random.choice(["Red", "Green", "Blue", "Yellow"])
            mat = next(m for m in materials if color in m.mat_name)
            cube = BoxObject(
                name=f"cube{i}",
                size_min=cube_size,
                size_max=cube_size,
                rgba=[0.8, 0.8, 0.8, 1],
                material=mat,
            )
            self.objs.append(cube)

        # placement
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.objs)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.objs,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objs,
        )

    def _get_reference(self):
        super()._get_reference()
        self.obj_body_ids = [self.sim.model.body_name2id(o.root_body) for o in self.objs]

    def _reset_internal(self):
        super()._reset_internal()
        object_placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in object_placements.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
            )

    def _get_observation(self):
        di = super()._get_observation()
        if self.use_object_obs:
            pr = self.robots[0].robot_model.naming_prefix

            obs = []
            for i, obj in enumerate(self.objs):
                pos = np.array(self.sim.data.body_xpos[self.obj_body_ids[i]])
                quat = convert_quat(np.array(self.sim.data.body_xquat[self.obj_body_ids[i]]), to="xyzw")
                obs.extend([*pos, *quat])
                di[f"{obj.name}_pos"] = pos
                di[f"{obj.name}_quat"] = quat

            di["object-state"] = np.array(obs)
        return di

    def _check_success(self):
        """Success if all cubes stacked sequentially."""
        for i in range(1, self.num_stack_objs):
            if not self.check_contact(self.objs[i], self.objs[i - 1]):
                return False
        return True
