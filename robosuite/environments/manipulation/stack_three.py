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
    Minimal change version of StackThree with corrected staged_rewards for a guaranteed C -> B -> A stack order.
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
        reward_shaping=False, # We recommend setting this to True for this task
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
        # table/top settings
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        # reward config
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        # whether to include object states in observations
        self.use_object_obs = use_object_obs
        # object placement initializer
        self.placement_initializer = placement_initializer
        # bonus for stacking (kept but not required for sparse success)
        self.full_stacking_bonus = full_stacking_bonus

        # Cube identifiers for staged rewards
        self.cubes = None
        self.cube_body_ids = None

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
            # sparse: only give reward when the full 3-block stack is present
            reward = 3.0 if (r_stack >= 3.0 - 1e-6) else 0.0 # Use 3.0 here for easier scaling

        if self.reward_scale is not None:
            # Final reward is scaled by reward_scale / 3.0 so that a full success equals reward_scale.
            reward *= self.reward_scale / 3.0
        return reward

    # -----------------------------------------------------------
    # CRITICAL FIX: Fixed-Stage Reward Logic for C -> B -> A Stacking
    # -----------------------------------------------------------
    def staged_rewards(self):
        """ 
        Returns 3 components (r_reach, r_lift, r_stack) with logic to enforce B on C, then A on B.
        """
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        
        # Cube positions
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id].copy()
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id].copy()
        cubeC_pos = self.sim.data.body_xpos[self.cubeC_body_id].copy()
        
        # Determine stack progress (B on C, then A on B)
        b_on_c = self.check_contact(self.cubeB, self.cubeC)
        a_on_b = self.check_contact(self.cubeA, self.cubeB)

        r_stack = 0.0
        target_pick = None         # The cube the robot should be aiming to GRASP
        placement_base_pos = None  # The position the HELD cube should be placed ON (e.g., C or B)

        # --- 1. Determine Stack Status and Next Target ---
        if not b_on_c:
            # Stage 1: Get Cube B and put it on Cube C.
            target_pick = self.cubeB
            placement_base_pos = cubeC_pos
        elif b_on_c and not a_on_b:
            # Stage 2: Get Cube A and put it on Cube B.
            target_pick = self.cubeA
            placement_base_pos = cubeB_pos
            r_stack += 1.5 # Reward for B on C
        elif b_on_c and a_on_b:
            # Stage 3: Full Success! Tower is A on B on C.
            target_pick = self.cubeA # Stay focused on the top block (A)
            placement_base_pos = cubeB_pos 
            r_stack += 3.0 # Full reward

        # --- 2. Determine Held Cube ---
        held_cube = None
        for cube in [self.cubeA, self.cubeB, self.cubeC]:
            if self._check_grasp(self.robots[0].gripper, cube):
                held_cube = cube
                break

        # --- 3. Reach and Grasp Reward (r_reach) ---
        target_pick_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(target_pick.root_body)]
        
        # Use a slightly softer tanh coefficient (5.0) than the original 10.0 for easier initial reaching
        dist_reach = np.linalg.norm(gripper_site_pos - target_pick_pos)
        r_reach = (1 - np.tanh(5.0 * dist_reach)) * 0.25 
        
        # Grasp reward: Only for successfully grasping the *correct* target cube. 
        if held_cube is target_pick:
            r_reach += 0.25 # Max r_reach is 0.5 (0.25 reach + 0.25 grasp)

        # --- 4. Lift and Align Reward (r_lift) ---
        r_lift = 0.0
        
        # Lift reward: Only given if the correct cube is held and lifted
        if held_cube is target_pick:
            held_cube_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(held_cube.root_body)]
            
            # Check if the held cube is lifted 4cm above the table plane
            lifted = held_cube_pos[2] > self.table_offset[2] + 0.04
            
            if lifted:
                r_lift += 1.0
                
                # Alignment reward: Distance between held cube and its intended placement base
                # Use a slightly softer tanh coefficient (5.0) than the original 10.0 for easier alignment
                horiz_dist = np.linalg.norm(held_cube_pos[:2] - placement_base_pos[:2])
                r_lift += 0.5 * (1 - np.tanh(5.0 * horiz_dist)) # Max r_lift is 1.5

        return r_reach, r_lift, r_stack

    # [Rest of the class methods remain the same as your second version]

    def _get_env_info(self, action):
        info = super()._get_env_info(action)
        r_reach, r_lift, r_stack = self.staged_rewards()
        cubes_stacked = (r_stack >= 3.0 - 1e-6)
        info.update({
            'r_reach_grasp': r_reach / 0.50, 
            'r_lift_align': r_lift / 1.50,
            'r_stack': r_stack / 3.0,
            'cubes_stacked': cubes_stacked,
            'success': self._check_success(),
        })
        return info

    def _get_skill_info(self):
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id].copy()
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id].copy()
        cubeC_pos = self.sim.data.body_xpos[self.cubeC_body_id].copy()
        pos_info = {}
        pos_info['grasp'] = [cubeA_pos] 
        pos_info['reach'] = [cubeB_pos]
        pos_info['base'] = [cubeC_pos] 
        info = {}
        for k in pos_info:
            info[k + '_pos'] = pos_info[k]
        return info

    def _load_model(self):
        """ Loads the model and adds three identical boxes (cubeA, cubeB, cubeC). """
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # materials
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(texture="WoodRed", tex_name="redwood", mat_name="redwood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        greenwood = CustomMaterial(texture="WoodGreen", tex_name="greenwood", mat_name="greenwood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        bluewood = CustomMaterial(texture="WoodBlue", tex_name="bluewood", mat_name="bluewood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)

        # Three IDENTICAL boxes (using the medium size: 0.025)
        cube_size = [0.025, 0.025, 0.025]
        self.cubeA = BoxObject(name="cubeA", size_min=cube_size, size_max=cube_size, rgba=[1, 0, 0, 1], material=redwood)
        self.cubeB = BoxObject(name="cubeB", size_min=cube_size, size_max=cube_size, rgba=[0, 1, 0, 1], material=greenwood)
        self.cubeC = BoxObject(name="cubeC", size_min=cube_size, size_max=cube_size, rgba=[0, 0, 1, 1], material=bluewood)
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
        self.cubes = cubes

    def _get_reference(self):
        super()._get_reference()
        self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)
        self.cube_body_ids = [self.cubeA_body_id, self.cubeB_body_id, self.cubeC_body_id]

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _get_observation(self):
        di = super()._get_observation()
        if self.use_object_obs:
            pr = self.robots[0].robot_model.naming_prefix
            
            def get_cube_data(cube_body_id):
                pos = np.array(self.sim.data.body_xpos[cube_body_id])
                quat = convert_quat(np.array(self.sim.data.body_xquat[cube_body_id]), to="xyzw")
                return pos, quat

            cubeA_pos, cubeA_quat = get_cube_data(self.cubeA_body_id)
            cubeB_pos, cubeB_quat = get_cube_data(self.cubeB_body_id)
            cubeC_pos, cubeC_quat = get_cube_data(self.cubeC_body_id)

            di["cubeA_pos"], di["cubeA_quat"] = cubeA_pos, cubeA_quat
            di["cubeB_pos"], di["cubeB_quat"] = cubeB_pos, cubeB_quat
            di["cubeC_pos"], di["cubeC_quat"] = cubeC_pos, cubeC_quat

            gripper_site_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
            
            di[pr + "gripper_to_cubeA"] = gripper_site_pos - cubeA_pos
            di[pr + "gripper_to_cubeB"] = gripper_site_pos - cubeB_pos
            di[pr + "gripper_to_cubeC"] = gripper_site_pos - cubeC_pos
            di["cubeA_to_cubeB"] = cubeA_pos - cubeB_pos
            di["cubeB_to_cubeC"] = cubeB_pos - cubeC_pos
            di["cubeA_to_cubeC"] = cubeA_pos - cubeC_pos

            # object-state contains positions, quaternions, and relative vectors for all three cubes
            di["object-state"] = np.concatenate(
                [
                    cubeA_pos, cubeA_quat,
                    cubeB_pos, cubeB_quat,
                    cubeC_pos, cubeC_quat,
                    di[pr + "gripper_to_cubeA"],
                    di[pr + "gripper_to_cubeB"],
                    di[pr + "gripper_to_cubeC"],
                    di["cubeA_to_cubeB"],
                    di["cubeB_to_cubeC"],
                    di["cubeA_to_cubeC"],
                ]
            )
        return di

    def _check_success(self):
        _, _, r_stack = self.staged_rewards()
        return r_stack >= 3.0 - 1e-6

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings.get("grippers", False):
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)

