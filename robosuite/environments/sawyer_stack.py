from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import TableTopTask, UniformRandomSampler


class SawyerStack(SawyerEnv):
    """
    This class corresponds to the stacking task for the Sawyer robot arm.
    """

    def __init__(
        self,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        gripper_visualization=False,
        use_indicator_object=False,
        indicator_args=None,
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
    ):
        """
        Args:

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

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                ensure_object_boundary_in_range=False,
                z_rotation=None,
            )

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            indicator_args=indicator_args,
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
        )

        # reward configuration
        self.reward_shaping = reward_shaping

        # information of objects
        # self.object_names = [o['object_name'] for o in self.object_metadata]
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

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
            self.mujoco_arena.add_pos_indicator(**self.indicator_args)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # initialize objects of interest
        cubeA = BoxObject(
            size_min=[0.02, 0.02, 0.02], size_max=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1]
        )
        cubeB = BoxObject(
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
        )
        self.mujoco_objects = OrderedDict([("cubeA", cubeA), ("cubeB", cubeB)])
        self.n_objects = len(self.mujoco_objects)

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
        self.cubeA_body_id = self.sim.model.body_name2id("cubeA")
        self.cubeB_body_id = self.sim.model.body_name2id("cubeB")
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        self.cubeA_geom_id = self.sim.model.geom_name2id("cubeA")
        self.cubeB_geom_id = self.sim.model.geom_name2id("cubeB")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # reset positions of objects
        self.model.place_objects()

        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

    def reward(self, action):
        """
        Reward function for the task.

        The dense reward has five components.

            Reaching: in [0, 1], to encourage the arm to reach the cube
            Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            Lifting: in {0, 1}, non-zero if arm has lifted the cube
            Aligning: in [0, 0.5], encourages aligning one cube over the other
            Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The sparse reward only consists of the stacking component.
        However, the sparse reward is either 0 or 1.

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 1.0 if r_stack > 0 else 0.0

        return reward

    def staged_rewards(self):
        """
        Helper function to return staged rewards based on current physical states.

        Returns:
            r_reach (float): reward for reaching and grasping
            r_lift (float): reward for lifting and aligning
            r_stack (float): reward for stacking
        """
        # reaching is successful when the gripper site is close to
        # the center of the cube
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # collision checking
        touch_left_finger = False
        touch_right_finger = False
        touch_cubeA_cubeB = False

        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.cubeA_geom_id:
                touch_left_finger = True
            if c.geom1 == self.cubeA_geom_id and c.geom2 in self.l_finger_geom_ids:
                touch_left_finger = True
            if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.cubeA_geom_id:
                touch_right_finger = True
            if c.geom1 == self.cubeA_geom_id and c.geom2 in self.r_finger_geom_ids:
                touch_right_finger = True
            if c.geom1 == self.cubeA_geom_id and c.geom2 == self.cubeB_geom_id:
                touch_cubeA_cubeB = True
            if c.geom1 == self.cubeB_geom_id and c.geom2 == self.cubeA_geom_id:
                touch_cubeA_cubeB = True

        # additional grasping reward
        if touch_left_finger and touch_right_finger:
            r_reach += 0.25

        # lifting is successful when the cube is above the table top
        # by a margin
        cubeA_height = cubeA_pos[2]
        table_height = self.table_full_size[2]
        cubeA_lifted = cubeA_height > table_height + 0.04
        r_lift = 1.0 if cubeA_lifted else 0.0

        # Aligning is successful when cubeA is right above cubeB
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(
                np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2])
            )
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # stacking is successful when the block is lifted and
        # the gripper is not holding the object
        r_stack = 0
        not_touching = not touch_left_finger and not touch_right_finger
        if not_touching and r_lift > 0 and touch_cubeA_cubeB:
            r_stack = 2.0

        return (r_reach, r_lift, r_stack)

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
            # position and rotation of the first cube
            cubeA_pos = np.array(self.sim.data.body_xpos[self.cubeA_body_id])
            cubeA_quat = convert_quat(
                np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw"
            )
            di["cubeA_pos"] = cubeA_pos
            di["cubeA_quat"] = cubeA_quat

            # position and rotation of the second cube
            cubeB_pos = np.array(self.sim.data.body_xpos[self.cubeB_body_id])
            cubeB_quat = convert_quat(
                np.array(self.sim.data.body_xquat[self.cubeB_body_id]), to="xyzw"
            )
            di["cubeB_pos"] = cubeB_pos
            di["cubeB_quat"] = cubeB_quat

            # relative positions between gripper and cubes
            gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["gripper_to_cubeA"] = gripper_site_pos - cubeA_pos
            di["gripper_to_cubeB"] = gripper_site_pos - cubeB_pos
            di["cubeA_to_cubeB"] = cubeA_pos - cubeB_pos

            di["object-state"] = np.concatenate(
                [
                    cubeA_pos,
                    cubeA_quat,
                    cubeB_pos,
                    cubeB_quat,
                    di["gripper_to_cubeA"],
                    di["gripper_to_cubeB"],
                    di["cubeA_to_cubeB"],
                ]
            )

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
        _, _, r_stack = self.staged_rewards()
        return r_stack > 0

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

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba
