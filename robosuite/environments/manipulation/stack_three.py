from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler


class StackThree(SingleArmEnv):
    """
    3-cube stacking environment (Phase 1 refactor).
    Fixes: deterministic target order, consistent held-cube logic,
    smoother reward shaping.  Goal: build tower C→B→A.
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
        full_stacking_bonus=0.0,
        skill_config=None,
    ):
        # table + reward settings
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self.full_stacking_bonus = full_stacking_bonus

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

    # ------------------------------------------------------------------
    # Reward logic (Phase 1 refactor)
    # ------------------------------------------------------------------
    def reward(self, action):
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 3.0 if (r_stack >= 3.0 - 1e-6) else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale / 3.0
        return reward

    def staged_rewards(self):
        """Staged rewards with deterministic order and smoother shaping."""
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]

        # cube positions
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id].copy()
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id].copy()
        cubeC_pos = self.sim.data.body_xpos[self.cubeC_body_id].copy()

        # stacking progress
        b_on_c = self.check_contact(self.cubeB, self.cubeC)
        a_on_b = self.check_contact(self.cubeA, self.cubeB)

        r_stack = 0.0
        if b_on_c:
            r_stack += 1.5
            if a_on_b:
                r_stack += 1.5

        # 1-2. deterministic order (always build C→B→A)
        if not b_on_c:
            target_pick = self.cubeB
            placement_base_pos = cubeC_pos
        else:
            target_pick = self.cubeA
            placement_base_pos = cubeB_pos

        target_pick_pos = self.sim.data.body_xpos[
            self.sim.model.body_name2id(target_pick.root_body)
        ]

        # 3. detect held cube correctly
        held_cube = None
        for cube in [self.cubeA, self.cubeB, self.cubeC]:
            if self._check_grasp(self.robots[0].gripper, cube):
                held_cube = cube
                break

        # reach + grasp
        dist = np.linalg.norm(gripper_pos - target_pick_pos)
        r_reach = (1 - np.tanh(5.0 * dist)) * 0.25  # smoother curve
        if held_cube is target_pick:
            r_reach += 0.25

        # 3-4. lift + align (smoother tanh)
        r_lift = 0.0
        if held_cube is not None:
            held_pos = self.sim.data.body_xpos[
                self.sim.model.body_name2id(held_cube.root_body)
            ]
            lifted = held_pos[2] > self.table_offset[2] + 0.04
            if lifted:
                r_lift += 1.0
                horiz_dist = np.linalg.norm(held_pos[:2] - placement_base_pos[:2])
                r_lift += 0.5 * (1 - np.tanh(3.0 * horiz_dist))

        return r_reach, r_lift, r_stack

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _get_env_info(self, action):
        info = super()._get_env_info(action)
        r_reach, r_lift, r_stack = self.staged_rewards()
        cubes_stacked = (r_stack >= 3.0 - 1e-6)
        info.update({
            "r_reach_grasp": r_reach / 0.50,
            "r_lift_align": r_lift / 1.50,
            "r_stack": r_stack / 3.0,
            "cubes_stacked": cubes_stacked,
            "success": self._check_success(),
        })
        return info

    # ------------------------------------------------------------------
    # Model + object setup
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

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(texture="WoodRed", tex_name="redwood", mat_name="redwood_mat",
                                 tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        greenwood = CustomMaterial(texture="WoodGreen", tex_name="greenwood", mat_name="greenwood_mat",
                                   tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        bluewood = CustomMaterial(texture="WoodBlue", tex_name="bluewood", mat_name="bluewood_mat",
                                  tex_attrib=tex_attrib, mat_attrib=mat_attrib)

        cube_size = [0.025, 0.025, 0.025]
        self.cubeA = BoxObject("cubeA", cube_size, cube_size, [1, 0, 0, 1], redwood)
        self.cubeB = BoxObject("cubeB", cube_size, cube_size, [0, 1, 0, 1], greenwood)
        self.cubeC = BoxObject("cubeC", cube_size, cube_size, [0, 0, 1, 1], bluewood)
        cubes = [self.cubeA, self.cubeB, self.cubeC]

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
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
            mujoco_objects=cubes,
        )

    def _get_reference(self):
        super()._get_reference()
        self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observation(self):
        di = super()._get_observation()
        if self.use_object_obs:
            pr = self.robots[0].robot_model.naming_prefix

            def get_data(bid):
                pos = np.array(self.sim.data.body_xpos[bid])
                quat = convert_quat(np.array(self.sim.data.body_xquat[bid]), to="xyzw")
                return pos, quat

            cubeA_pos, cubeA_quat = get_data(self.cubeA_body_id)
            cubeB_pos, cubeB_quat = get_data(self.cubeB_body_id)
            cubeC_pos, cubeC_quat = get_data(self.cubeC_body_id)

            di.update({
                "cubeA_pos": cubeA_pos, "cubeA_quat": cubeA_quat,
                "cubeB_pos": cubeB_pos, "cubeB_quat": cubeB_quat,
                "cubeC_pos": cubeC_pos, "cubeC_quat": cubeC_quat,
            })

            grip_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
            di[pr + "gripper_to_cubeA"] = grip_pos - cubeA_pos
            di[pr + "gripper_to_cubeB"] = grip_pos - cubeB_pos
            di[pr + "gripper_to_cubeC"] = grip_pos - cubeC_pos
            di["cubeA_to_cubeB"] = cubeA_pos - cubeB_pos
            di["cubeB_to_cubeC"] = cubeB_pos - cubeC_pos
            di["cubeA_to_cubeC"] = cubeA_pos - cubeC_pos

            di["object-state"] = np.concatenate([
                cubeA_pos, cubeA_quat,
                cubeB_pos, cubeB_quat,
                cubeC_pos, cubeC_quat,
                di[pr + "gripper_to_cubeA"],
                di[pr + "gripper_to_cubeB"],
                di[pr + "gripper_to_cubeC"],
                di["cubeA_to_cubeB"],
                di["cubeB_to_cubeC"],
                di["cubeA_to_cubeC"],
            ])
        return di

    # ------------------------------------------------------------------
    # Success + visualize
    # ------------------------------------------------------------------
    def _check_success(self):
        _, _, r_stack = self.staged_rewards()
        return r_stack >= 3.0 - 1e-6

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings.get("grippers", False):
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cubeA
            )
