from collections import OrderedDict
import random
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array, range_to_uniform_grid
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas import PegsArena
from robosuite.models.objects import SquareNutObject, RoundNutObject
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import NutAssemblyTask, UniformRandomPegsSampler, RoundRobinPegsSampler
from robosuite.controllers import load_controller_config
import os

class SawyerNutAssembly(SawyerEnv):
    """
    This class corresponds to the nut assembly task for the Sawyer robot arm.
    """

    def __init__(
        self,
        controller_config=None,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.45, 0.69, 0.82),
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        nut_type=None,
        gripper_visualization=False,
        use_indicator_object=False,
        indicator_num=1,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        camera_real_depth=False,
        camera_segmentation=False,
        eval_mode=False,
        num_evals=50,
        perturb_evals=False,
    ):
        """
        Args:
            controller_config (dict): If set, contains relevant controller parameters for creating a custom controller.
                Else, uses the default controller for this specific task

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomPegsSampler
                is used by default.

            single_object_mode (int): specifies which version of the task to do. Note that
                the observations change accordingly.

                0: corresponds to the full task with both types of nuts.

                1: corresponds to an easier task with only one type of nut initialized
                   on the table with every reset. The type is randomized on every reset.

                2: corresponds to an easier task with only one type of nut initialized
                   on the table with every reset. The type is kept constant and will not
                   change between resets.

            nut_type (string): if provided, should be either "round" or "square". Determines
                which type of nut (round or square) will be spawned on every environment
                reset. Only used if @single_object_mode is 2.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            indicator_num (int): number of indicator objects to add to the environment.
                Only used if @use_indicator_object is True.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.

            camera_real_depth (bool): True if convert depth to real depth in meters

            camera_segmentation (bool): True if also return semantic segmentation of the camera view
        """

        # Load the default controller if none is specified
        if not controller_config:
            controller_path = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/default_sawyer.json')
            controller_config = load_controller_config(custom_fpath=controller_path)

        # Assert that the controller config is a dict file
        assert type(controller_config) == dict, \
            "Inputted controller config must be a dict! Instead, got type: {}".format(type(controller_config))

        # task settings
        self.single_object_mode = single_object_mode
        self.nut_to_id = {"square": 0, "round": 1}
        if nut_type is not None:
            assert (
                nut_type in self.nut_to_id.keys()
            ), "invalid @nut_type argument - choose one of {}".format(
                list(self.nut_to_id.keys())
            )
            self.nut_id = self.nut_to_id[nut_type]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        # placement initilizer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomPegsSampler(
                x_range={ "SquareNut" : [-0.115, -0.11], "RoundNut" : [-0.115, -0.11], },
                y_range={ "SquareNut" : [0.11, 0.225], "RoundNut" : [-0.225, -0.11], },
                z_range={ "SquareNut" : [0.02, 0.10], "RoundNut" : [0.02, 0.10], },
                ensure_object_boundary_in_range=False,
                z_rotation=None,
            )

        super().__init__(
            controller_config=controller_config,
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            indicator_num=indicator_num,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
            camera_real_depth=camera_real_depth,
            camera_segmentation=camera_segmentation,
            eval_mode=eval_mode,
            num_evals=num_evals,
            perturb_evals=perturb_evals,
        )

    def _get_placement_initializer_for_eval_mode(self):
        """
        Sets a placement initializer that is used to initialize the
        environment into a fixed set of known task instances.
        This is for reproducibility in policy evaluation.
        """

        assert(self.eval_mode)

        # set up placement grid by getting bounds per dimension and then
        # using meshgrid to get all combinations
        x_bounds, y_bounds, z_bounds, z_rot_bounds = self._grid_bounds_for_eval_mode()
        
        # iterate over nut types to get a grid per nut type
        final_x_grid = {}
        final_y_grid = {}
        final_z_grid = {}
        final_z_rot_grid = {}
        for k in x_bounds:
            x_grid = range_to_uniform_grid(a=x_bounds[k][0], b=x_bounds[k][1], n=x_bounds[k][2])
            y_grid = range_to_uniform_grid(a=y_bounds[k][0], b=y_bounds[k][1], n=y_bounds[k][2])
            z_grid = range_to_uniform_grid(a=z_bounds[k][0], b=z_bounds[k][1], n=z_bounds[k][2])
            z_rotation = range_to_uniform_grid(a=z_rot_bounds[k][0], b=z_rot_bounds[k][1], n=z_rot_bounds[k][2])
            grid = np.meshgrid(x_grid, y_grid, z_grid, z_rotation)
            x_grid = grid[0].ravel()
            y_grid = grid[1].ravel()
            z_grid = grid[2].ravel()
            z_rotation = grid[3].ravel()
            grid_length = x_grid.shape[0]

            round_robin_period = self.num_evals
            if self.perturb_evals:
                # sample 100 rounds of perturbations and then sampler will repeat
                round_robin_period *= 100

                # perturbation size should be half the grid spacing
                x_pos_perturb_size = ((x_bounds[k][1] - x_bounds[k][0]) / x_bounds[k][2]) / 2.
                y_pos_perturb_size = ((y_bounds[k][1] - y_bounds[k][0]) / y_bounds[k][2]) / 2.
                z_pos_perturb_size = ((z_bounds[k][1] - z_bounds[k][0]) / z_bounds[k][2]) / 2.
                z_rot_perturb_size = ((z_rot_bounds[k][1] - z_rot_bounds[k][0]) / z_rot_bounds[k][2]) / 2.

            # assign grid locations for the full round robin schedule
            final_x_grid[k] = np.zeros(round_robin_period)
            final_y_grid[k] = np.zeros(round_robin_period)
            final_z_grid[k] = np.zeros(round_robin_period)
            final_z_rot_grid[k] = np.zeros(round_robin_period)
            for t in range(round_robin_period):
                g_ind = t % grid_length
                x, y, z, z_rot = x_grid[g_ind], y_grid[g_ind], z_grid[g_ind], z_rotation[g_ind]
                if self.perturb_evals:
                    x += np.random.uniform(low=-x_pos_perturb_size, high=x_pos_perturb_size)
                    y += np.random.uniform(low=-y_pos_perturb_size, high=y_pos_perturb_size)
                    z += np.random.uniform(low=-z_pos_perturb_size, high=z_pos_perturb_size)
                    z_rot += np.random.uniform(low=-z_rot_perturb_size, high=z_rot_perturb_size)
                final_x_grid[k][t], final_y_grid[k][t], final_z_grid[k][t], final_z_rot_grid[k][t] = x, y, z, z_rot

        self.placement_initializer = RoundRobinPegsSampler(
            x_range=final_x_grid,
            y_range=final_y_grid,
            z_range=final_z_grid,
            ensure_object_boundary_in_range=False,
            z_rotation=final_z_rot_grid
        )

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, z positions,
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        print("")
        print("*" * 50)
        print("WARNING! TODO: figure out nice way to have all combinations of objects...")
        print("*" * 50)
        print("")

        # (low, high, number of grid points for this dimension)
        x_bounds = { "SquareNut" : (-0.115, -0.11, 3), "RoundNut" : (-0.115, -0.11, 3) }
        y_bounds = { "SquareNut" : (0.11, 0.225, 3), "RoundNut" : (-0.225, -0.11, 3) }
        z_bounds = { "SquareNut" : (0.02, 0.02, 1), "RoundNut" : (0.02, 0.02, 1) }
        z_rot_bounds = { "SquareNut" : (0., 2. * np.pi, 3), "RoundNut" : (0., 2. * np.pi, 3) }
        return x_bounds, y_bounds, z_bounds, z_rot_bounds

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = PegsArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([.5, -0.15, 0])

        # define mujoco objects
        self.ob_inits = [SquareNutObject, RoundNutObject]
        self.item_names = ["SquareNut", "RoundNut"]
        self.item_names_org = list(self.item_names)
        self.obj_to_use = (self.item_names[1] + "{}").format(0)
        self.ngeoms = [5, 9]

        # randomimze initial qpos of the joints
        self.init_qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        self.init_qpos += np.random.randn(self.init_qpos.shape[0]) * 0.02

        lst = []
        for i in range(len(self.ob_inits)):
            ob = self.ob_inits[i]()
            lst.append((str(self.item_names[i]) + "0", ob))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = NutAssemblyTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.placement_initializer,
        )
        self.model.place_objects()
        self.table_pos = string_to_array(self.model.table_body.get("pos"))
        self.peg1_pos = string_to_array(self.model.peg1_body.get("pos"))  # square
        self.peg2_pos = string_to_array(self.model.peg2_body.get("pos"))  # round

    def clear_objects(self, obj):
        """
        Clears objects with name @obj out of the task space. This is useful
        for supporting task modes with single types of objects, as in
        @self.single_object_mode without changing the model definition.
        """
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            if obj_name == obj:
                continue
            else:
                sim_state = self.sim.get_state()
                # print(self.sim.model.get_joint_qpos_addr(obj_name))
                sim_state.qpos[self.sim.model.get_joint_qpos_addr(obj_name)[0]] = 10
                self.sim.set_state(sim_state)
                self.sim.forward()

    def _get_reference(self):
        super()._get_reference()
        self.obj_body_id = {}
        self.obj_geom_id = {}

        for i in range(len(self.ob_inits)):
            obj_str = str(self.item_names[i]) + "0"
            self.obj_body_id[obj_str] = self.sim.model.body_name2id(obj_str)
            geom_ids = []
            for j in range(self.ngeoms[i]):
                geom_ids.append(self.sim.model.geom_name2id(obj_str + "-{}".format(j)))
            self.obj_geom_id[obj_str] = geom_ids

        # information of objects
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

        # keep track of which objects are on their corresponding pegs
        self.objects_on_pegs = np.zeros(len(self.ob_inits))

    def _reset_internal(self):
        super()._reset_internal()

        # reset positions of objects, and move objects out of the scene depending on the mode
        self.model.place_objects()
        if self.single_object_mode == 1:
            self.obj_to_use = (random.choice(self.item_names) + "{}").format(0)
            self.clear_objects(self.obj_to_use)
        elif self.single_object_mode == 2:
            self.obj_to_use = (self.item_names[self.nut_id] + "{}").format(0)
            self.clear_objects(self.obj_to_use)

    def reward(self, action=None):
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_on_pegs)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already on the correct pegs
        names_to_reach = []
        objs_to_reach = []
        geoms_to_grasp = []
        geoms_by_array = []

        for i in range(len(self.ob_inits)):
            if self.objects_on_pegs[i]:
                continue
            obj_str = str(self.item_names[i]) + "0"
            names_to_reach.append(obj_str)
            objs_to_reach.append(self.obj_body_id[obj_str])
            geoms_to_grasp.extend(self.obj_geom_id[obj_str])
            geoms_by_array.append(self.obj_geom_id[obj_str])

        ### reaching reward governed by distance to closest object ###
        r_reach = 0.
        if len(objs_to_reach):
            # reaching reward via minimum distance to the handles of the objects (the last geom of each nut)
            geom_ids = [elem[-1] for elem in geoms_by_array]
            target_geom_pos = self.sim.data.geom_xpos[geom_ids]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dists = np.linalg.norm(
                target_geom_pos - gripper_site_pos.reshape(1, -1), axis=1
            )
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        ### grasping reward for touching any objects of interest ###
        touch_left_finger = False
        touch_right_finger = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in geoms_to_grasp:
                if c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            elif c.geom2 in geoms_to_grasp:
                if c.geom1 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids:
                    touch_right_finger = True
        has_grasp = touch_left_finger and touch_right_finger
        r_grasp = int(has_grasp) * grasp_mult

        ### lifting reward for picking up an object ###
        r_lift = 0.
        if len(objs_to_reach) and r_grasp > 0.:
            z_target = self.table_pos[2] + 0.2
            object_z_locs = self.sim.data.body_xpos[objs_to_reach][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
                lift_mult - grasp_mult
            )

        ### hover reward for getting object above peg ###
        r_hover = 0.
        if len(objs_to_reach):
            r_hovers = np.zeros(len(objs_to_reach))
            for i in range(len(objs_to_reach)):
                if names_to_reach[i].startswith(self.item_names[0]):
                    peg_pos = self.peg1_pos[:2]
                elif names_to_reach[i].startswith(self.item_names[1]):
                    peg_pos = self.peg2_pos[:2]
                else:
                    raise Exception(
                        "Got invalid object to reach: {}".format(names_to_reach[i])
                    )
                ob_xy = self.sim.data.body_xpos[objs_to_reach[i]][:2]
                dist = np.linalg.norm(peg_pos - ob_xy)
                r_hovers[i] = r_lift + (1 - np.tanh(10.0 * dist)) * (
                    hover_mult - lift_mult
                )
            r_hover = np.max(r_hovers)

        return r_reach, r_grasp, r_lift, r_hover

    def on_peg(self, obj_pos, peg_id):

        if peg_id == 0:
            peg_pos = self.peg1_pos
        else:
            peg_pos = self.peg2_pos
        res = False
        if (
            abs(obj_pos[0] - peg_pos[0]) < 0.03
            and abs(obj_pos[1] - peg_pos[1]) < 0.03
            and obj_pos[2] < self.model.table_offset[2] + 0.05
        ):
            res = True
        return res

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        
        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()

        # low-level object information
        if self.use_object_obs:

            # remember the keys to collect into object info
            object_state_keys = []

            # for conversion to relative gripper frame
            gripper_pose = T.pose2mat((di["eef_pos"], di["eef_quat"]))
            world_pose_in_gripper = T.pose_inv(gripper_pose)

            for i in range(len(self.item_names_org)):

                if self.single_object_mode == 2 and self.nut_id != i:
                    # skip observations
                    continue

                obj_str = str(self.item_names_org[i]) + "0"
                obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_str]])
                obj_quat = T.convert_quat(
                    self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
                )
                di["{}_pos".format(obj_str)] = obj_pos
                di["{}_quat".format(obj_str)] = obj_quat

                object_pose = T.pose2mat((obj_pos, obj_quat))
                rel_pose = T.pose_in_A_to_pose_in_B(object_pose, world_pose_in_gripper)
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                di["{}_to_eef_pos".format(obj_str)] = rel_pos
                di["{}_to_eef_quat".format(obj_str)] = rel_quat

                object_state_keys.append("{}_pos".format(obj_str))
                object_state_keys.append("{}_quat".format(obj_str))
                object_state_keys.append("{}_to_eef_pos".format(obj_str))
                object_state_keys.append("{}_to_eef_quat".format(obj_str))

            if self.single_object_mode == 1:
                # zero out other objs
                for obj_str, obj_mjcf in self.mujoco_objects.items():
                    if obj_str == self.obj_to_use:
                        continue
                    else:
                        di["{}_pos".format(obj_str)] *= 0.0
                        di["{}_quat".format(obj_str)] *= 0.0
                        di["{}_to_eef_pos".format(obj_str)] *= 0.0
                        di["{}_to_eef_quat".format(obj_str)] *= 0.0

            di["object-state"] = np.concatenate([di[k] for k in object_state_keys])

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1) in self.finger_names
                or self.sim.model.geom_id2name(contact.geom2) in self.finger_names
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """

        # remember objects that are on the correct pegs
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        for i in range(len(self.ob_inits)):
            obj_str = str(self.item_names[i]) + "0"
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_on_pegs[i] = int(self.on_peg(obj_pos, i) and r_reach < 0.6)

        if self.single_object_mode > 0:
            return np.sum(self.objects_on_pegs) > 0  # need one object on peg

        # returns True if all objects are on correct pegs
        return np.sum(self.objects_on_pegs) == len(self.ob_inits)

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        # color the gripper site appropriately based on distance to nearest object
        if self.gripper_visualization:
            # find closest object
            square_dist = lambda x: np.sum(
                np.square(x - self.sim.data.get_site_xpos("grip_site"))
            )
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.eef_site_id] = np.inf  # make sure we don't pick the same site
            dists[self.eef_cylinder_id] = np.inf
            ob_dists = dists[
                self.object_site_ids
            ]  # filter out object sites we care about
            min_dist = np.min(ob_dists)
            ob_id = np.argmin(ob_dists)
            ob_name = self.object_names[ob_id]

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba


class SawyerNutAssemblySingle(SawyerNutAssembly):
    """
    Easier version of task - place either one round nut or one square nut into its peg.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=1, **kwargs)


class SawyerNutAssemblySquare(SawyerNutAssembly):
    """
    Easier version of task - place one square nut into its peg.
    """

    def __init__(self, **kwargs):
        assert (
            "single_object_mode" not in kwargs and "nut_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, nut_type="square", **kwargs)


class SawyerNutAssemblyRound(SawyerNutAssembly):
    """
    Easier version of task - place one round nut into its peg.
    """

    def __init__(self, **kwargs):
        assert (
            "single_object_mode" not in kwargs and "nut_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, nut_type="round", **kwargs)

### Some new environments... ###

class SawyerNutAssemblySquareConstantRotation(SawyerNutAssemblySquare):
    """
    Square nut is initialized in the same orientation each time, and with
    a fixed z-offset corresponding to lying flat on the table.
    The original environment had a random z-offset to initialize the nut 
    in the air for some reason...
    """

    def __init__(self, **kwargs):
        assert("placement_initializer" not in kwargs)
        kwargs["placement_initializer"] = UniformRandomPegsSampler(
            x_range={ "SquareNut" : [-0.115, -0.11], "RoundNut" : [-0.115, -0.11], },
            y_range={ "SquareNut" : [0.11, 0.225], "RoundNut" : [-0.225, -0.11], },
            z_range={ "SquareNut" : [0.02, 0.02], "RoundNut" : [0.02, 0.02], },
            ensure_object_boundary_in_range=False,
            z_rotation={ "SquareNut" : np.pi, "RoundNut" : np.pi, },
        )
        super().__init__(**kwargs)

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, z positions,
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = { "SquareNut" : (-0.115, -0.11, 3), "RoundNut" : (-0.115, -0.11, 3) }
        y_bounds = { "SquareNut" : (0.11, 0.225, 3), "RoundNut" : (-0.225, -0.11, 3) }
        z_bounds = { "SquareNut" : (0.02, 0.02, 1), "RoundNut" : (0.02, 0.02, 1) }
        z_rot_bounds = { "SquareNut" : (np.pi, np.pi, 1), "RoundNut" : (np.pi, np.pi, 1) }
        return x_bounds, y_bounds, z_bounds, z_rot_bounds

class SawyerNutAssemblySquareConstantRotationPosition(SawyerNutAssemblySquare):
    """
    Same as SawyerNutAssemblySquareConstantRotation but if using OSC, use
    position-only control.
    """
    def __init__(
        self,
        **kwargs
    ):
        assert("placement_initializer" not in kwargs)
        kwargs["placement_initializer"] = UniformRandomPegsSampler(
            x_range={ "SquareNut" : [-0.115, -0.11], "RoundNut" : [-0.115, -0.11], },
            y_range={ "SquareNut" : [0.11, 0.225], "RoundNut" : [-0.225, -0.11], },
            z_range={ "SquareNut" : [0.02, 0.02], "RoundNut" : [0.02, 0.02], },
            ensure_object_boundary_in_range=False,
            z_rotation={ "SquareNut" : np.pi, "RoundNut" : np.pi, },
        )
        if kwargs["controller"] == "position_orientation":
            kwargs["controller"] = "position"
        super().__init__(**kwargs)

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, z positions,
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = { "SquareNut" : (-0.115, -0.11, 3), "RoundNut" : (-0.115, -0.11, 3) }
        y_bounds = { "SquareNut" : (0.11, 0.225, 3), "RoundNut" : (-0.225, -0.11, 3) }
        z_bounds = { "SquareNut" : (0.02, 0.02, 1), "RoundNut" : (0.02, 0.02, 1) }
        z_rot_bounds = { "SquareNut" : (np.pi, np.pi, 1), "RoundNut" : (np.pi, np.pi, 1) }
        return x_bounds, y_bounds, z_bounds, z_rot_bounds
