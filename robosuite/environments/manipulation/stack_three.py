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
    Minimal change version of StackThree. Stacks three identical cubes (A, B, C).
    Reward shaping is updated to incentivize targeting the next unstacked cube.
    """
    # [Rest of __init__ is unchanged]
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
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            skill_config=skill_config,
        )

    # [Reward function is unchanged]
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
    # CRITICAL CHANGE: Phased Reward Logic to Target Unstacked Cubes
    # -----------------------------------------------------------
    def staged_rewards(self):
        """ 
        Returns 3 components (r_reach, r_lift, r_stack) adapted to three-block stacking.
        The reach/lift target is dynamically set to the next cube that needs to be picked.
        """
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        
        # Cube positions (must use copies as they are needed for comparisons)
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id].copy()
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id].copy()
        cubeC_pos = self.sim.data.body_xpos[self.cubeC_body_id].copy()
        
        # 1. Determine Stacking Status and Target Cube
        # Check contact (robust way to track stack progress)
        a_on_b = self.check_contact(self.cubeA, self.cubeB)
        a_on_c = self.check_contact(self.cubeA, self.cubeC)
        b_on_c = self.check_contact(self.cubeB, self.cubeC)

        r_stack = 0.0
        r_stack_progress = 0 # 0, 1, or 2 for progress

        # Assume C is the fixed base for now (simplification)
        # Check B on C (1-block stack achieved)
        if b_on_c:
            r_stack += 1.5
            r_stack_progress = 1
            
            # Check A on B (2-block stack achieved, full tower)
            if a_on_b:
                r_stack += 1.5
                r_stack_progress = 2
        
        # Identify the next cube to pick (target_cube) and its placement target (target_base_pos)
        target_cube = None
        target_pos = None

        if r_stack_progress == 0:
            # Stage 1: Pick the first cube (e.g., B) and place it on the table (C).
            # This is hard to define without a fixed sequence. A simpler approach is:
            # Target the cube that is not yet part of a 2-block stack.
            # We'll stick to a fixed sequence A->B->C (or B on C, then A on B) for simplicity,
            # and let the policy figure out the order, but prioritize picking an unstacked cube.

            # We'll use the cube furthest from the gripper as a default if no clear stack target.
            # Find the closest ungrasped, unstacked cube to the gripper.
            
            # Let's use a simpler heuristic for the target: the cube not currently being held.
            # If nothing is held, target the cube closest to the gripper.
            
            cubes_list = [self.cubeA, self.cubeB, self.cubeC]
            positions = [cubeA_pos, cubeB_pos, cubeC_pos]
            
            # Find the cube currently being held
            grasping_cubes = [self._check_grasp(self.robots[0].gripper, cube) for cube in cubes_list]
            held_cube_idx = np.where(grasping_cubes)[0]

            if len(held_cube_idx) == 0:
                # Nothing held: Target the closest cube to pick up.
                dists = [np.linalg.norm(gripper_site_pos - pos) for pos in positions]
                target_cube = cubes_list[np.argmin(dists)]
                target_cube_pos = positions[np.argmin(dists)]
            else:
                # Holding a cube: The target is the placement location.
                held_cube = cubes_list[held_cube_idx[0]]
                
                # If holding the first cube (e.g., B), target C for placement.
                if held_cube is self.cubeB and not b_on_c:
                    target_cube_pos = cubeC_pos
                # If holding the second cube (e.g., A), target B for placement (if B is on C).
                elif held_cube is self.cubeA and b_on_c and not a_on_b:
                    target_cube_pos = cubeB_pos
                else:
                    # Default: keep aligning it over a potential base (e.g., the table or the lowest base)
                    target_cube_pos = np.array([0, 0, self.table_offset[2]]) # Center of table
                    
                target_cube = held_cube # Redefine target_cube for the reach reward in the lift/align phase
                
        # 2. Reach and Grasp Reward (r_reach)
        # We always incentivize reaching and grasping the current TARGET CUBE.
        if target_cube is None: # Should only happen if all cubes are somehow placed or in a weird state
            target_cube = self.cubeA
            target_cube_pos = cubeA_pos

        dist = np.linalg.norm(gripper_site_pos - target_cube_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25
        
        # Grasping reward for the target cube
        grasping_target = self._check_grasp(self.robots[0].gripper, object_geoms=target_cube)
        if grasping_target:
            r_reach += 0.25 # Reach + Grasp max 0.5
            
        # 3. Lift and Align Reward (r_lift)
        
        # Cube must be lifted. We use the currently grasped cube's height.
        cube_to_check = target_cube if grasping_target else self.cubeA # Fallback to A if nothing is grasped
        cube_to_check_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(cube_to_check.root_body)]
        
        cube_lifted = cube_to_check_pos[2] > self.table_offset[2] + 0.04
        r_lift = 1.0 if cube_lifted else 0.0
        
        # Aligning: When the *grasped cube* is above its *placement target*.
        if target_cube_pos is not None and target_cube_pos[2] > self.table_offset[2] + 0.01:
             # The target_cube_pos here is the placement location (B or C), not the cube being held.
             # We need the pos of the CUBE BEING HELD for alignment
             held_cube_pos = cube_to_check_pos
             
             # Placement target is the base cube's center
             placement_base_pos = None
             if grasping_target and target_cube is self.cubeB and not b_on_c:
                 # Holding B, target is C
                 placement_base_pos = cubeC_pos
             elif grasping_target and target_cube is self.cubeA and b_on_c and not a_on_b:
                 # Holding A, target is B
                 placement_base_pos = cubeB_pos
             
             if placement_base_pos is not None:
                horiz_dist = np.linalg.norm(held_cube_pos[:2] - placement_base_pos[:2])
                r_lift += 0.5 * (1 - np.tanh(10.0 * horiz_dist)) # Max r_lift is 1.5

        # 4. Stacking (r_stack) is calculated at the beginning (0.0, 1.5, or 3.0)
        # Note: If r_stack is 1.5, the policy should shift its focus from 'picking first cube' to 'picking second cube' due to the dynamic target logic in step 1.
        
        return r_reach, r_lift, r_stack

    # [Rest of the class is mostly unchanged]
    def _get_env_info(self, action):
        info = super()._get_env_info(action)
        r_reach, r_lift, r_stack = self.staged_rewards()
        cubes_stacked = (r_stack >= 3.0 - 1e-6)
        info.update({
            # Max possible r_reach is 0.5, r_lift is 1.5, r_stack is 3.0
            'r_reach_grasp': r_reach / 0.50, 
            'r_lift_align': r_lift / 1.50,
            'r_stack': r_stack / 3.0,
            'cubes_stacked': cubes_stacked,
            'success': self._check_success(),
        })
        return info

    def _get_skill_info(self):
        # This is now less relevant since the target is dynamic, but we'll stick to A as a default target
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id].copy()
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id].copy()
        cubeC_pos = self.sim.data.body_xpos[self.cubeC_body_id].copy()
        pos_info = {}
        # Default targets: Grasp A, place on B, which is placed on C
        pos_info['grasp'] = [cubeA_pos] 
        pos_info['reach'] = [cubeB_pos]
        pos_info['base'] = [cubeC_pos] 
        info = {}
        for k in pos_info:
            info[k + '_pos'] = pos_info[k]
        return info

    # -----------------------------------------------------------
    # CRITICAL CHANGE: All Cubes Same Size (Medium)
    # -----------------------------------------------------------
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
            
            # Helper to get cube data
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
            # Target visualization is less meaningful with dynamic targets, keep the original target (A) for simplicity
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)