from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.baxter import BaxterEnv

from robosuite.models.objects import PotWithHandlesObject
from robosuite.models.arenas import TableArena
from robosuite.models.robots import Baxter
from robosuite.models.tasks import TableTopTask, UniformRandomSampler


class BaxterLift(BaxterEnv):
    """
    This class corresponds to the bimanual lifting task for the Baxter robot.
    """

    def __init__(
        self,
        gripper_type_right="TwoFingerGripper",
        gripper_type_left="LeftTwoFingerGripper",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_object_obs=True,
        reward_shaping=True,
        **kwargs
    ):
        """
        Args:

            gripper_type_right (str): type of gripper used on the right hand.

            gripper_type_lefft (str): type of gripper used on the right hand.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_object_obs (bool): if True, include object (pot) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

        Inherits the Baxter environment; refer to other parameters described there.
        """

        # initialize the pot
        self.pot = PotWithHandlesObject()
        self.mujoco_objects = OrderedDict([("pot", self.pot)])

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        self.object_initializer = UniformRandomSampler(
            x_range=(-0.15, -0.04),
            y_range=(-0.015, 0.015),
            z_rotation=(-0.15 * np.pi, 0.15 * np.pi),
            ensure_object_boundary_in_range=False,
        )

        super().__init__(
            gripper_left=gripper_type_left, gripper_right=gripper_type_right, **kwargs
        )

    def _load_model(self):
        """
        Loads the arena and pot object.
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.45 + self.table_full_size[0] / 2, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.object_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flattened array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.cube_body_id = self.sim.model.body_name2id("pot")
        self.handle_1_site_id = self.sim.model.site_name2id("pot_handle_1")
        self.handle_2_site_id = self.sim.model.site_name2id("pot_handle_2")
        self.table_top_id = self.sim.model.site_name2id("table_top")
        self.pot_center_id = self.sim.model.site_name2id("pot_center")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        self.model.place_objects()

    def reward(self, action):
        """
        Reward function for the task.

          1. the agent only gets the lifting reward when flipping no more than 30 degrees.
          2. the lifting reward is smoothed and ranged from 0 to 2, capped at 2.0.
             the initial lifting reward is 0 when the pot is on the table;
             the agent gets the maximum 2.0 reward when the pot’s height is above a threshold.
          3. the reaching reward is 0.5 when the left gripper touches the left handle,
             or when the right gripper touches the right handle before the gripper geom
             touches the handle geom, and once it touches we use 0.5
        """
        reward = 0

        cube_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.get_top_offset()[2]
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # check if the pot is tilted more than 30 degrees
        mat = T.quat2mat(self._pot_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_30 = np.cos(np.pi / 6)
        direction_coef = 1 if cos_z >= cos_30 else 0

        # cube is higher than the table top above a margin
        if cube_height > table_height + 0.15:
            reward = 1.0 * direction_coef

        # use a shaping reward
        if self.reward_shaping:
            reward = 0

            # lifting reward
            elevation = cube_height - table_height
            r_lift = min(max(elevation - 0.05, 0), 0.2)
            reward += 10. * direction_coef * r_lift

            l_gripper_to_handle = self._l_gripper_to_handle
            r_gripper_to_handle = self._r_gripper_to_handle

            # gh stands for gripper-handle
            # When grippers are far away, tell them to be closer
            l_contacts = list(
                self.find_contacts(
                    self.gripper_left.contact_geoms(), self.pot.handle_1_geoms()
                )
            )
            r_contacts = list(
                self.find_contacts(
                    self.gripper_right.contact_geoms(), self.pot.handle_2_geoms()
                )
            )
            l_gh_dist = np.linalg.norm(l_gripper_to_handle)
            r_gh_dist = np.linalg.norm(r_gripper_to_handle)

            if len(l_contacts) > 0:
                reward += 0.5
            else:
                reward += 0.5 * (1 - np.tanh(l_gh_dist))

            if len(r_contacts) > 0:
                reward += 0.5
            else:
                reward += 0.5 * (1 - np.tanh(r_gh_dist))

        return reward

    @property
    def _handle_1_xpos(self):
        """Returns the position of the first handle."""
        return self.sim.data.site_xpos[self.handle_1_site_id]

    @property
    def _handle_2_xpos(self):
        """Returns the position of the second handle."""
        return self.sim.data.site_xpos[self.handle_2_site_id]

    @property
    def _pot_quat(self):
        """Returns the orientation of the pot."""
        return T.convert_quat(self.sim.data.body_xquat[self.cube_body_id], to="xyzw")

    @property
    def _world_quat(self):
        """World quaternion."""
        return T.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    @property
    def _l_gripper_to_handle(self):
        """Returns vector from the left gripper to the handle."""
        return self._handle_1_xpos - self._l_eef_xpos

    @property
    def _r_gripper_to_handle(self):
        """Returns vector from the right gripper to the handle."""
        return self._handle_2_xpos - self._r_eef_xpos

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
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            cube_quat = T.convert_quat(
                self.sim.data.body_xquat[self.cube_body_id], to="xyzw"
            )
            di["cube_pos"] = cube_pos
            di["cube_quat"] = cube_quat

            di["l_eef_xpos"] = self._l_eef_xpos
            di["r_eef_xpos"] = self._r_eef_xpos
            di["handle_1_xpos"] = self._handle_1_xpos
            di["handle_2_xpos"] = self._handle_2_xpos
            di["l_gripper_to_handle"] = self._l_gripper_to_handle
            di["r_gripper_to_handle"] = self._r_gripper_to_handle

            di["object-state"] = np.concatenate(
                [
                    di["cube_pos"],
                    di["cube_quat"],
                    di["l_eef_xpos"],
                    di["r_eef_xpos"],
                    di["handle_1_xpos"],
                    di["handle_2_xpos"],
                    di["l_gripper_to_handle"],
                    di["r_gripper_to_handle"],
                ]
            )

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        contact_geoms = (
            self.gripper_right.contact_geoms() + self.gripper_left.contact_geoms()
        )
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1) in contact_geoms
                or self.sim.model.geom_id2name(contact.geom2) in contact_geoms
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task is successfully completed
        """
        # cube is higher than the table top above a margin
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_full_size[2]
        return cube_height > table_height + 0.10
