from collections import OrderedDict
import random
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array
from robosuite.environments.robot_env import RobotEnv
from robosuite.robots import SingleArm

from robosuite.models.arenas import PegsArena
from robosuite.models.objects import SquareNutObject, RoundNutObject
from robosuite.models.tasks import NutAssemblyTask, UniformRandomPegsSampler


class NutAssembly(RobotEnv):
    """
    This class corresponds to the nut assembly task for a single robot arm.
    """

    def __init__(
        self,
        robots,
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        initialization_noise=0.02,
        table_full_size=(0.45, 0.69, 0.82),
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        nut_type=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        """
        Args:
            robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
                (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
                Note: Must be a single single-arm robot!

            controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
                custom controller. Else, uses the default controller for this specific task. Should either be single
                dict if same controller is to be used for all robots or else it should be a list of the same length as
                "robots" param

            gripper_types (str or list of str): type of gripper, used to instantiate
                gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
                with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
                overrides the default gripper. Should either be single str if same gripper type is to be used for all
                robots or else it should be a list of the same length as "robots" param

            gripper_visualizations (bool or list of bool): True if using gripper visualization.
                Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
                robots or else it should be a list of the same length as "robots" param

            initialization_noise (float or list of floats): The scale factor of uni-variate Gaussian random noise
                applied to each of a robot's given initial joint positions. Setting this value to "None" or 0.0 results
                in no noise being applied. Should either be single float if same noise value is to be used for all
                robots or else it should be a list of the same length as "robots" param

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

             use_camera_obs (bool): if True, every observation includes rendered image(s)

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
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

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering

            render_camera (str): Name of camera to render if `has_renderer` is True.

            render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

            control_freq (float): how many control signals to receive in every second. This sets the amount of
                simulation time that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_names (str or list of str): name of camera to be rendered. Should either be single str if
                same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
                Note: At least one camera must be specified if @use_camera_obs is True.
                Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
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

        """
        # First, verify that only one robot is being inputted
        self._check_robot_configuration(robots)

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

        # reward configuration
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomPegsSampler(
                x_range=[-0.15, 0.],
                y_range=[-0.2, 0.2],
                z_range=[0.02, 0.10],
                ensure_object_boundary_in_range=False,
                z_rotation=True,
            )

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            gripper_visualizations=gripper_visualizations,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

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
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
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

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], SingleArm), \
            "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["pegs"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        self.mujoco_arena = PegsArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # define mujoco objects
        self.ob_inits = [SquareNutObject, RoundNutObject]
        self.item_names = ["SquareNut", "RoundNut"]
        self.item_names_org = list(self.item_names)
        self.obj_to_use = (self.item_names[1] + "{}").format(0)
        self.ngeoms = [5, 9]

        lst = []
        for i in range(len(self.ob_inits)):
            ob = self.ob_inits[i]()
            lst.append((str(self.item_names[i]) + "0", ob))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = NutAssemblyTask(
            self.mujoco_arena,
            [robot.robot_model for robot in self.robots],
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )

        # set positions of objects
        self.model.place_objects()

        # positions of table and pegs
        self.table_pos = string_to_array(self.model.table_body.get("pos"))
        self.peg1_pos = string_to_array(self.model.peg1_body.get("pos"))  # square
        self.peg2_pos = string_to_array(self.model.peg2_body.get("pos"))  # round

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Additional object references from this env
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
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.right_finger_geoms
        ]
        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

        # keep track of which objects are on their corresponding pegs
        self.objects_on_pegs = np.zeros(len(self.ob_inits))

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            obj_pos, obj_quat = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for i, (obj_name, _) in enumerate(self.mujoco_objects.items()):
                self.sim.data.set_joint_qpos(obj_name, np.concatenate([np.array(obj_pos[i]), np.array(obj_quat[i])]))

        # information of objects
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # Move objects out of the scene depending on the mode
        if self.single_object_mode == 1:
            self.obj_to_use = (random.choice(self.item_names) + "{}").format(0)
            self.clear_objects(self.obj_to_use)
        elif self.single_object_mode == 2:
            self.obj_to_use = (self.item_names[self.nut_id] + "{}").format(0)
            self.clear_objects(self.obj_to_use)

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
            # Get robot prefix
            pr = self.robots[0].robot_model.naming_prefix

            # remember the keys to collect into object info
            object_state_keys = []

            # for conversion to relative gripper frame
            gripper_pose = T.pose2mat((di[pr + "eef_pos"], di[pr + "eef_quat"]))
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
                di["{}_to_{}eef_pos".format(obj_str, pr)] = rel_pos
                di["{}_to_{}eef_quat".format(obj_str, pr)] = rel_quat

                object_state_keys.append("{}_pos".format(obj_str))
                object_state_keys.append("{}_quat".format(obj_str))
                object_state_keys.append("{}_to_{}eef_pos".format(obj_str, pr))
                object_state_keys.append("{}_to_{}eef_quat".format(obj_str, pr))

            if self.single_object_mode == 1:
                # zero out other objs
                for obj_str, obj_mjcf in self.mujoco_objects.items():
                    if obj_str == self.obj_to_use:
                        continue
                    else:
                        di["{}_pos".format(obj_str)] *= 0.0
                        di["{}_quat".format(obj_str)] *= 0.0
                        di["{}_to_{}eef_pos".format(obj_str, pr)] *= 0.0
                        di["{}_to_{}eef_quat".format(obj_str, pr)] *= 0.0

            di["object-state"] = np.concatenate([di[k] for k in object_state_keys])

        return di

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        # remember objects that are on the correct pegs
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
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

    def _visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.robots[0].gripper_visualization:
            # find closest object
            square_dist = lambda x: np.sum(
                np.square(x - self.sim.data.get_site_xpos(self.robots[0].gripper.visualization_sites["grip_site"]))
            )
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.robots[0].eef_site_id] = np.inf  # make sure we don't pick the same site
            dists[self.robots[0].eef_cylinder_id] = np.inf
            ob_dists = dists[
                self.object_site_ids
            ]  # filter out object sites we care about
            min_dist = np.min(ob_dists)
            ob_id = np.argmin(ob_dists)

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.robots[0].eef_site_id] = rgba

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable
        """
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"


class NutAssemblySingle(NutAssembly):
    """
    Easier version of task - place either one round nut or one square nut into its peg.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=1, **kwargs)


class NutAssemblySquare(NutAssembly):
    """
    Easier version of task - place one square nut into its peg.
    """

    def __init__(self, **kwargs):
        assert (
                "single_object_mode" not in kwargs and "nut_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, nut_type="square", **kwargs)


class NutAssemblyRound(NutAssembly):
    """
    Easier version of task - place one round nut into its peg.
    """

    def __init__(self, **kwargs):
        assert (
                "single_object_mode" not in kwargs and "nut_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, nut_type="round", **kwargs)
