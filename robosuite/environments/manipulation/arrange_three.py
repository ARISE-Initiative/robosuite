from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler


class ArrangeThree(SingleArmEnv):
    """
    A simple pre-stacking task:
    Arrange three cubes (A, B, C) side-by-side on a straight line along X (or Y).
    No balancing or precise stacking required. Pushing is enough.

    Shaping terms (if reward_shaping=True):
      - Line alignment (all cubes close to target line)
      - Spacing alignment (pairwise distances close to target spacing)
      - Height regularization (stay near tabletop height)

    Success (sparse):
      - All cubes within tolerances for line alignment + spacing.
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
        use_camera_obs=False,
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
        horizon=400,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        # Task params
        line_axis="x",            # "x" (default) or "y"
        target_spacing=0.06,      # distance between adjacent cube centers on the line
        tol_line=0.02,            # max distance to line (per cube)
        tol_spacing=0.02,         # max spacing error (per pair)
        tol_height=0.01,          # max height error from tabletop
        cube_size=0.025,          # half-size per axis
        skill_config=None,
    ):
        # Table & task settings
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)

        # Rewards
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # Obs / placement
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        # Arrangement params
        assert line_axis in ("x", "y")
        self.line_axis = line_axis
        self.target_spacing = float(target_spacing)
        self.tol_line = float(tol_line)
        self.tol_spacing = float(tol_spacing)
        self.tol_height = float(tol_height)
        self.cube_size = float(cube_size)

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

    # -------------------- Reward --------------------
    def reward(self, action):
        r_align, r_space, r_height = self.staged_rewards()  # each in [0, 1]
        if self.reward_shaping:
            # Weighted sum -> max 1.0 (before scaling)
            # Put most weight on line alignment, then spacing, tiny on height
            shaped = 0.5 * r_align + 0.45 * r_space + 0.05 * r_height
            reward = shaped
        else:
            reward = 1.0 if self._check_success() else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale  # scale to your taste
        return reward

    def staged_rewards(self):
        """
        Returns (r_align, r_space, r_height), each in [0, 1], where:
          - r_align: how close cubes are to the target line (perpendicular distance)
          - r_space: how close pairwise spacings are to target spacing
          - r_height: how close heights are to tabletop height (avoid accidental lifts)
        """
        # Positions
        A = self.sim.data.body_xpos[self.cubeA_body_id].copy()
        B = self.sim.data.body_xpos[self.cubeB_body_id].copy()
        C = self.sim.data.body_xpos[self.cubeC_body_id].copy()
        pos = np.stack([A, B, C], axis=0)

        # Table height reference
        table_z = self.table_offset[2]
        z = pos[:, 2]

        # Target line is through the table center along chosen axis.
        # We'll anchor at table_offset (x0,y0) and require all cubes be close to that line.
        if self.line_axis == "x":
            # x can vary freely; y should be near table_offset[1]
            dist_to_line = np.abs(pos[:, 1] - self.table_offset[1])
            # spacing along x
            coords = np.sort(pos[:, 0])
        else:
            # y-axis line: y can vary; x should be near table_offset[0]
            dist_to_line = np.abs(pos[:, 0] - self.table_offset[0])
            # spacing along y
            coords = np.sort(pos[:, 1])

        # r_align: map distance-to-line into [0,1] with tolerance
        # 1.0 when within tol_line; decays with tanh outside
        def smooth_clamp01(d, tol):
            # <= tol -> ~1; increases beyond tol -> decreases smoothly
            # Use scaled tanh for a soft boundary
            return np.clip(1.0 - np.tanh(5.0 * np.maximum(0.0, d - tol)), 0.0, 1.0)

        r_align_per = smooth_clamp01(dist_to_line, self.tol_line)
        r_align = float(np.mean(r_align_per))

        # r_space: compare pairwise distances on the line axis to [0, s, 2s]
        # After sorting coords: target deltas are [s, s] between neighbors
        deltas = np.diff(coords)  # shape (2,)
        spacing_err = np.abs(deltas - self.target_spacing)
        r_space_pair = smooth_clamp01(spacing_err, self.tol_spacing)
        r_space = float(np.mean(r_space_pair)) if r_space_pair.size > 0 else 0.0

        # r_height: keep near table height (+ half cube)
        # We want them resting on the table, not lifted: target z ~= table_z + cube_size
        target_z = table_z + self.cube_size
        r_height_per = smooth_clamp01(np.abs(z - target_z), self.tol_height)
        r_height = float(np.mean(r_height_per))

        return r_align, r_space, r_height

    # -------------------- Info / Success --------------------
    def _get_env_info(self, action):
        info = super()._get_env_info(action)
        r_align, r_space, r_height = self.staged_rewards()
        info.update({
            "r_align": r_align,
            "r_space": r_space,
            "r_height": r_height,
            "success": self._check_success(),
        })
        return info

    def _check_success(self):
        r_align, r_space, r_height = self.staged_rewards()
        # Require each term be high enough
        return (r_align > 0.95) and (r_space > 0.95) and (r_height > 0.95)

    # -------------------- Model & Objects --------------------
    def _load_model(self):
        super()._load_model()

        # Place robot base
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Arena
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        # Materials
        tex = {"type": "cube"}
        mat = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(texture="WoodRed",   tex_name="redwood",  mat_name="redwood_mat",  tex_attrib=tex, mat_attrib=mat)
        greenwd = CustomMaterial(texture="WoodGreen", tex_name="greenwd",  mat_name="greenwd_mat",  tex_attrib=tex, mat_attrib=mat)
        bluewd  = CustomMaterial(texture="WoodBlue",  tex_name="bluewd",   mat_name="bluewd_mat",   tex_attrib=tex, mat_attrib=mat)

        size = [self.cube_size, self.cube_size, self.cube_size]
        # Use keyword args to avoid size/rgba mix-ups across robosuite versions
        self.cubeA = BoxObject(name="cubeA", size_min=size, size_max=size, rgba=[1, 0, 0, 1], material=redwood)
        self.cubeB = BoxObject(name="cubeB", size_min=size, size_max=size, rgba=[0, 1, 0, 1], material=greenwd)
        self.cubeC = BoxObject(name="cubeC", size_min=size, size_max=size, rgba=[0, 0, 1, 1], material=bluewd)
        cubes = [self.cubeA, self.cubeB, self.cubeC]

        # Placement
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ArrangeSampler",
                mujoco_objects=cubes,
                x_range=[-0.10, 0.10],
                y_range=[-0.10, 0.10],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # Task
        self.model = ManipulationTask(
            mujoco_arena=arena,
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
            placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    # -------------------- Observations --------------------
    def _get_observation(self):
        di = super()._get_observation()

        if self.use_object_obs:
            pr = self.robots[0].robot_model.naming_prefix

            def get_data(bid):
                pos = np.array(self.sim.data.body_xpos[bid])
                quat = convert_quat(np.array(self.sim.data.body_xquat[bid]), to="xyzw")
                return pos, quat

            Apos, Aquat = get_data(self.cubeA_body_id)
            Bpos, Bquat = get_data(self.cubeB_body_id)
            Cpos, Cquat = get_data(self.cubeC_body_id)

            di["cubeA_pos"], di["cubeA_quat"] = Apos, Aquat
            di["cubeB_pos"], di["cubeB_quat"] = Bpos, Bquat
            di["cubeC_pos"], di["cubeC_quat"] = Cpos, Cquat

            grip = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
            di[pr + "gripper_to_cubeA"] = grip - Apos
            di[pr + "gripper_to_cubeB"] = grip - Bpos
            di[pr + "gripper_to_cubeC"] = grip - Cpos

            di["cubeA_to_cubeB"] = Apos - Bpos
            di["cubeB_to_cubeC"] = Bpos - Cpos
            di["cubeA_to_cubeC"] = Apos - Cpos

            di["object-state"] = np.concatenate([
                Apos, Aquat, Bpos, Bquat, Cpos, Cquat,
                di[pr + "gripper_to_cubeA"], di[pr + "gripper_to_cubeB"], di[pr + "gripper_to_cubeC"],
                di["cubeA_to_cubeB"], di["cubeB_to_cubeC"], di["cubeA_to_cubeC"],
            ])

        return di

    # -------------------- Viz --------------------
    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        # Optional: you could draw a thin line on the table to show the target alignment.
        # For a minimal version, we keep default visualization.
