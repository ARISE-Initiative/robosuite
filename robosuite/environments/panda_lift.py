from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.panda import PandaEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.robots import Panda
from robosuite.models.tasks import TableTopTask, UniformRandomSampler
from robosuite.controllers import load_controller_config

import os

class PandaLift(PandaEnv):
    """
    This class corresponds to the lifting task for the Panda robot arm.
    """

    def __init__(
        self,
        controller_config=None,
        gripper_type="PandaGripper",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
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
        eval_mode=False,
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
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

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
        """

        # Load the default controller if none is specified
        if not controller_config:
            controller_path = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/default_panda.json')
            controller_config = load_controller_config(custom_fpath=controller_path)

        # Assert that the controller config is a dict file
        assert type(controller_config) == dict, \
            "Inputted controller config must be a dict! Instead, got type: {}".format(type(controller_config))

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
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
            eval_mode=eval_mode,
            perturb_evals=perturb_evals,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The panda robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # initialize objects of interest
        cube = BoxObject(
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
        )
        self.mujoco_objects = OrderedDict([("cube", cube)])

        # reset initial joint positions (gets reset in sim during super() call in _reset_internal)
        self.init_qpos = np.array(self.mujoco_robot.init_qpos)
        self.init_qpos += np.random.randn(self.init_qpos.shape[0]) * 0.02

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.cube_body_id = self.sim.model.body_name2id("cube")
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        self.cube_geom_id = self.sim.model.geom_name2id("cube")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def reward(self, action=None):
        """
        Reward function for the task.

        The dense reward has three components.

            Reaching: in [0, 1], to encourage the arm to reach the cube
            Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:

            # reaching reward
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            touch_left_finger = False
            touch_right_finger = False
            for i in range(self.sim.data.ncon):
                c = self.sim.data.contact[i]
                if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.cube_geom_id:
                    touch_left_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.cube_geom_id:
                    touch_right_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            if touch_left_finger and touch_right_finger:
                reward += 0.25

        return reward

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
        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        # low-level object information
        if self.use_object_obs:
            # position and rotation of object
            cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
            cube_quat = convert_quat(
                np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw"
            )
            di["cube_pos"] = cube_pos
            di["cube_quat"] = cube_quat

            gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["gripper_to_cube"] = gripper_site_pos - cube_pos

            di["object-state"] = np.concatenate(
                [cube_pos, cube_quat, di["gripper_to_cube"]]
            )

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1)
                in self.gripper.contact_geoms()
                or self.sim.model.geom_id2name(contact.geom2)
                in self.gripper.contact_geoms()
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_full_size[2]

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.04

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.gripper_visualization:
            # get distance to cube
            cube_site_id = self.sim.model.site_name2id("cube")
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[cube_site_id]
                    - self.sim.data.get_site_xpos("grip_site")
                )
            )

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba
