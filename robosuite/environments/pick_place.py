from collections import OrderedDict
import random
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array
from robosuite.environments.robot_env import RobotEnv
from robosuite.robots import SingleArm

from robosuite.models.arenas import BinsArena
from robosuite.models.objects import (
    MilkObject,
    BreadObject,
    CerealObject,
    CanObject,
)
from robosuite.models.objects import (
    MilkVisualObject,
    BreadVisualObject,
    CerealVisualObject,
    CanVisualObject,
)
from robosuite.models.tasks import TableTopTask, SequentialCompositeSampler


class PickPlace(RobotEnv):
    """
    This class corresponds to the pick place task for a single robot arm.
    """

    def __init__(
        self,
        robots,
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        initialization_noise="default",
        table_full_size=(0.39, 0.49, 0.82),
        table_friction=(1, 0.005, 0.0001),
        bin1_pos=(0.1, -0.25, 0.8),
        bin2_pos=(0.1, 0.28, 0.8),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        single_object_mode=0,
        object_type=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
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

            initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
                The expected keys and corresponding value types are specified below:
                "magnitude": The scale factor of uni-variate random noise applied to each of a robot's given initial
                    joint positions. Setting this value to "None" or 0.0 results in no noise being applied.
                    If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                    If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
                "type": Type of noise to apply. Can either specify "gaussian" or "uniform"
                Should either be single dict if same noise value is to be used for all robots or else it should be a
                list of the same length as "robots" param
                Note: Specifying "default" will automatically use the default noise settings
                    Specifying None will automatically create the required dict with "magnitude" set to 0.0

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            bin1_pos (3-tuple): Absolute cartesian coordinates of the bin initially holding the objects

            bin2_pos (3-tuple): Absolute cartesian coordinates of the goal bin

            use_camera_obs (bool): if True, every observation includes rendered image(s)

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_scale (None or float): Scales the normalized reward function by the amount specified.
                If None, environment reward remains unnormalized

            reward_shaping (bool): if True, use dense rewards.

            single_object_mode (int): specifies which version of the task to do. Note that
                the observations change accordingly.

                0: corresponds to the full task with both types of nuts.

                1: corresponds to an easier task with only one type of nut initialized
                   on the table with every reset. The type is randomized on every reset.

                2: corresponds to an easier task with only one type of nut initialized
                   on the table with every reset. The type is kept constant and will not
                   change between resets.

            object_type (string): if provided, should be one of "milk", "bread", "cereal",
                or "can". Determines which type of object will be spawned on every
                environment reset. Only used if @single_object_mode is 2.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering

            render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
                will result in the default angle being applied, which is useful as it can be dragged / panned by
                the user using the mouse

            render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

            control_freq (float): how many control signals to receive in every second. This sets the amount of
                simulation time that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
                only calls sim.reset and resets all robosuite-internal variables

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
        self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        if object_type is not None:
            assert (
                    object_type in self.object_to_id.keys()
            ), "invalid @object_type argument - choose one of {}".format(
                list(self.object_to_id.keys())
            )
            self.object_id = self.object_to_id[
                object_type
            ]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # settings for bin position
        self.bin1_pos = np.array(bin1_pos)
        self.bin2_pos = np.array(bin2_pos)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

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
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_in_bins)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
        if self.single_object_mode == 0:
            reward /= 4.0
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

        # filter out objects that are already in the correct bins
        objs_to_reach = []
        geoms_to_grasp = []
        target_bin_placements = []
        for i in range(len(self.ob_inits)):
            if self.objects_in_bins[i]:
                continue
            obj_str = str(self.item_names[i]) + "0"
            objs_to_reach.append(self.obj_body_id[obj_str])
            geoms_to_grasp.append(self.obj_geom_id[obj_str])
            target_bin_placements.append(self.target_bin_placements[i])
        target_bin_placements = np.array(target_bin_placements)

        ### reaching reward governed by distance to closest object ###
        r_reach = 0.
        if len(objs_to_reach):
            # get reaching reward via minimum distance to a target object
            target_object_pos = self.sim.data.body_xpos[objs_to_reach]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dists = np.linalg.norm(
                target_object_pos - gripper_site_pos.reshape(1, -1), axis=1
            )
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        ### grasping reward for touching any objects of interest ###
        touch_left_finger = False
        touch_right_finger = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in geoms_to_grasp:
                bin_id = geoms_to_grasp.index(c.geom1)
                if c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            elif c.geom2 in geoms_to_grasp:
                bin_id = geoms_to_grasp.index(c.geom2)
                if c.geom1 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids:
                    touch_right_finger = True
        has_grasp = touch_left_finger and touch_right_finger
        r_grasp = int(has_grasp) * grasp_mult

        ### lifting reward for picking up an object ###
        r_lift = 0.
        if len(objs_to_reach) and r_grasp > 0.:
            z_target = self.bin2_pos[2] + 0.25
            object_z_locs = self.sim.data.body_xpos[objs_to_reach][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
                    lift_mult - grasp_mult
            )

        ### hover reward for getting object above bin ###
        r_hover = 0.
        if len(objs_to_reach):
            # segment objects into left of the bins and above the bins
            object_xy_locs = self.sim.data.body_xpos[objs_to_reach][:, :2]
            y_check = (
                    np.abs(object_xy_locs[:, 1] - target_bin_placements[:, 1])
                    < self.bin_size[1] / 4.
            )
            x_check = (
                    np.abs(object_xy_locs[:, 0] - target_bin_placements[:, 0])
                    < self.bin_size[0] / 4.
            )
            objects_above_bins = np.logical_and(x_check, y_check)
            objects_not_above_bins = np.logical_not(objects_above_bins)
            dists = np.linalg.norm(
                target_bin_placements[:, :2] - object_xy_locs, axis=1
            )
            # objects to the left get r_lift added to hover reward, those on the right get max(r_lift) added (to encourage dropping)
            r_hover_all = np.zeros(len(objs_to_reach))
            r_hover_all[objects_above_bins] = lift_mult + (
                    1 - np.tanh(10.0 * dists[objects_above_bins])
            ) * (hover_mult - lift_mult)
            r_hover_all[objects_not_above_bins] = r_lift + (
                    1 - np.tanh(10.0 * dists[objects_not_above_bins])
            ) * (hover_mult - lift_mult)
            r_hover = np.max(r_hover_all)

        return r_reach, r_grasp, r_lift, r_hover

    def not_in_bin(self, obj_pos, bin_id):

        bin_x_low = self.bin2_pos[0]
        bin_y_low = self.bin2_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= self.bin_size[0] / 2
        if bin_id < 2:
            bin_y_low -= self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0] / 2
        bin_y_high = bin_y_low + self.bin_size[1] / 2

        res = True
        if (
            bin_x_low < obj_pos[0] < bin_x_high
            and bin_y_low < obj_pos[1] < bin_y_high
            and self.bin2_pos[2] < obj_pos[2] < self.bin2_pos[2] + 0.1
        ):
            res = False
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
                sim_state.qpos[self.sim.model.get_joint_qpos_addr(obj_name + "_jnt0")[0]] = 10
                self.sim.set_state(sim_state)
                self.sim.forward()

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler()

        # can sample anywhere in bin
        bin_x_half = self.mujoco_arena.table_full_size[0] / 2 - 0.05
        bin_y_half = self.mujoco_arena.table_full_size[1] / 2 - 0.05

        # each object should just be sampled in the bounds of the bin (with some tolerance)
        for obj_name in self.mujoco_objects:

            self.placement_initializer.sample_on_top(
                obj_name,
                surface_name="table",
                x_range=[-bin_x_half, bin_x_half],
                y_range=[-bin_y_half, bin_y_half],
                rotation=None,
                rotation_axis='z',
                z_offset=0.,
                ensure_object_boundary_in_range=True,
            )

        # each visual object should just be at the center of each target bin
        index = 0
        for obj_name in self.visual_objects:

            # get center of target bin
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if index == 0 or index == 2:
                bin_x_low -= self.bin_size[0] / 2
            if index < 2:
                bin_y_low -= self.bin_size[1] / 2
            bin_x_high = bin_x_low + self.bin_size[0] / 2
            bin_y_high = bin_y_low + self.bin_size[1] / 2
            bin_center = np.array([
                (bin_x_low + bin_x_high) / 2., 
                (bin_y_low + bin_y_high) / 2., 
            ])

            # placement is relative to object bin, so compute difference and send to placement initializer
            rel_center = bin_center - self.bin1_pos[:2]

            self.placement_initializer.sample_on_top(
                obj_name,
                surface_name="table",
                x_range=[rel_center[0], rel_center[0]],
                y_range=[rel_center[1], rel_center[1]],
                rotation=0.,
                rotation_axis='z',
                z_offset=self.bin2_pos[2] - self.bin1_pos[2],
                ensure_object_boundary_in_range=False,
            )

            index += 1

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], SingleArm), \
            "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["bins"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        self.mujoco_arena = BinsArena(
            bin1_pos=self.bin1_pos,
            table_full_size=self.table_full_size,
            table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # store some arena attributes
        self.bin_size = self.mujoco_arena.table_full_size

        # define mujoco objects
        self.ob_inits = [MilkObject, BreadObject, CerealObject, CanObject]
        self.vis_inits = [
            MilkVisualObject,
            BreadVisualObject,
            CerealVisualObject,
            CanVisualObject,
        ]
        self.item_names = ["Milk", "Bread", "Cereal", "Can"]
        self.item_names_org = list(self.item_names)
        self.obj_to_use = (self.item_names[0] + "{}").format(0)

        lst = []
        for j in range(len(self.vis_inits)):
            visual_ob_name = ("Visual" + self.item_names[j] + "0")
            visual_ob = self.vis_inits[j](
                name=visual_ob_name,
                joints=[], # no free joint for visual objects
            )
            lst.append((visual_ob_name, visual_ob))
        self.visual_objects = OrderedDict(lst)

        lst = []
        for i in range(len(self.ob_inits)):
            ob_name = (self.item_names[i] + "0")
            ob = self.ob_inits[i](
                name=ob_name,
                joints=[dict(type="free", damping="0.0005")], # damp the free joint for each object
            )
            lst.append((ob_name, ob))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self._get_placement_initializer()
        self.model = TableTopTask(
            mujoco_arena=self.mujoco_arena, 
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.mujoco_objects, 
            visual_objects=self.visual_objects, 
            initializer=self.placement_initializer,
        )

        # set positions of objects
        self.model.place_objects()

        # self.model.place_visual()

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

        # id of grippers for contact checking
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.important_geoms["left_finger"]
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.important_geoms["right_finger"]
        ]

        # object-specific ids
        for i in range(len(self.ob_inits)):
            obj_str = str(self.item_names[i]) + "0"
            self.obj_body_id[obj_str] = self.sim.model.body_name2id(obj_str)
            self.obj_geom_id[obj_str] = self.sim.model.geom_name2id(obj_str)

        # for checking distance to / contact with objects we want to pick up
        self.target_object_body_ids = list(map(int, self.obj_body_id.values()))
        self.contact_with_object_geom_ids = list(map(int, self.obj_geom_id.values()))

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.ob_inits))

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.ob_inits), 3))
        for j in range(len(self.ob_inits)):
            bin_id = j
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2.
            if bin_id < 2:
                bin_y_low -= self.bin_size[1] / 2.
            bin_x_low += self.bin_size[0] / 4.
            bin_y_low += self.bin_size[1] / 4.
            self.target_bin_placements[j, :] = [bin_x_low, bin_y_low, self.bin2_pos[2]]

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
                self.sim.data.set_joint_qpos(obj_name + "_jnt0", np.concatenate([np.array(obj_pos[i]), np.array(obj_quat[i])]))

        # information of objects
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # Set the bins to the desired position
        self.sim.model.body_pos[self.sim.model.body_name2id("bin1")] = self.bin1_pos
        self.sim.model.body_pos[self.sim.model.body_name2id("bin2")] = self.bin2_pos

        # Move objects out of the scene depending on the mode
        if self.single_object_mode == 1:
            self.obj_to_use = (random.choice(self.item_names) + "{}").format(0)
            self.clear_objects(self.obj_to_use)
        elif self.single_object_mode == 2:
            self.obj_to_use = (self.item_names[self.object_id] + "{}").format(0)
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

                if self.single_object_mode == 2 and self.object_id != i:
                    # Skip adding to observations
                    continue

                obj_str = str(self.item_names_org[i]) + "0"
                obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_str]])
                obj_quat = T.convert_quat(
                    self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
                )
                di["{}_pos".format(obj_str)] = obj_pos
                di["{}_quat".format(obj_str)] = obj_quat

                # get relative pose of object in gripper frame
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
                # Zero out other objects observations
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
        # remember objects that are in the correct bins
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        for i in range(len(self.ob_inits)):
            obj_str = str(self.item_names[i]) + "0"
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_in_bins[i] = int(
                (not self.not_in_bin(obj_pos, i)) and r_reach < 0.6
            )

        # returns True if a single object is in the correct bin
        if self.single_object_mode == 1 or self.single_object_mode == 2:
            return np.sum(self.objects_in_bins) > 0

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.ob_inits)

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


class PickPlaceSingle(PickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=1, **kwargs)


class PickPlaceMilk(PickPlace):
    """
    Easier version of task - place one milk into its bin.
    """

    def __init__(self, **kwargs):
        assert (
                "single_object_mode" not in kwargs and "object_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="milk", **kwargs)


class PickPlaceBread(PickPlace):
    """
    Easier version of task - place one bread into its bin.
    """

    def __init__(self, **kwargs):
        assert (
                "single_object_mode" not in kwargs and "object_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="bread", **kwargs)


class PickPlaceCereal(PickPlace):
    """
    Easier version of task - place one cereal into its bin.
    """

    def __init__(self, **kwargs):
        assert (
                "single_object_mode" not in kwargs and "object_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="cereal", **kwargs)


class PickPlaceCan(PickPlace):
    """
    Easier version of task - place one can into its bin.
    """

    def __init__(self, **kwargs):
        assert (
                "single_object_mode" not in kwargs and "object_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="can", **kwargs)
