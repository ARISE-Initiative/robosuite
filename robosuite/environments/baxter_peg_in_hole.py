from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.baxter import BaxterEnv

from robosuite.models.objects import CylinderObject, PlateWithHoleObject
from robosuite.models.arenas import EmptyArena
from robosuite.models.robots import Baxter
from robosuite.models import MujocoWorldBase

import json
import os

class BaxterPegInHole(BaxterEnv):
    """
    This class corresponds to the peg in hole task for the Baxter robot. There's
    a cylinder attached to one gripper and a hole attached to the other one.
    """

    def __init__(
        self,
        controller_config=None,
        cylinder_radius=(0.015, 0.03),
        cylinder_length=0.13,
        use_object_obs=True,
        reward_shaping=True,
        **kwargs
    ):
        """
        Args:
            controller_config (dict): If set, contains relevant controller parameters for creating a custom controller.
                Else, uses the default controller for this specific task

            cylinder_radius (2-tuple): low and high limits of the (uniformly sampled)
                radius of the cylinder

            cylinder_length (float): length of the cylinder

            use_object_obs (bool): if True, include object information in the observation.

            reward_shaping (bool): if True, use dense rewards

        Inherits the Baxter environment; refer to other parameters described there.
        """

        # Load the default controller if none is specified
        if not controller_config:
            # TODO: Change to baxter default
            controller_path = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/default_sawyer.json')
            try:
                with open(controller_path) as f:
                    controller_config = json.load(f)
            except FileNotFoundError:
                print("Error opening default controller filepath at: {}. "
                      "Please check filepath and try again.".format(controller_path))

        # initialize objects of interest
        self.hole = PlateWithHoleObject()

        cylinder_radius = np.random.uniform(0.015, 0.03)
        self.cylinder = CylinderObject(
            size_min=(cylinder_radius, cylinder_length),
            size_max=(cylinder_radius, cylinder_length),
        )

        self.mujoco_objects = OrderedDict()

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        super().__init__(
            controller_config=controller_config,
            gripper_left=None,
            gripper_right=None,
            **kwargs
        )

    def _load_model(self):
        """
        Loads the peg and the hole models.
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # Add arena and robot
        self.model = MujocoWorldBase()
        self.arena = EmptyArena()
        if self.use_indicator_object:
            self.arena.add_pos_indicator()
        self.model.merge(self.arena)
        self.model.merge(self.mujoco_robot)

        # Load hole object
        self.hole_obj = self.hole.get_collision(name="hole", site=True)
        self.hole_obj.set("quat", "0 0 0.707 0.707")
        self.hole_obj.set("pos", "0.11 0 0.18")
        self.model.merge_asset(self.hole)
        self.model.worldbody.find(".//body[@name='left_hand']").append(self.hole_obj)

        # Load cylinder object
        self.cyl_obj = self.cylinder.get_collision(name="cylinder", site=True)
        self.cyl_obj.set("pos", "0 0 0.15")
        self.model.merge_asset(self.cylinder)
        self.model.worldbody.find(".//body[@name='right_hand']").append(self.cyl_obj)
        self.model.worldbody.find(".//geom[@name='cylinder']").set("rgba", "0 1 0 1")

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flattened array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.hole_body_id = self.sim.model.body_name2id("hole")
        self.cyl_body_id = self.sim.model.body_name2id("cylinder")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _compute_orientation(self):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.
        """
        cyl_mat = self.sim.data.body_xmat[self.cyl_body_id]
        cyl_mat.shape = (3, 3)
        cyl_pos = self.sim.data.body_xpos[self.cyl_body_id]

        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)

        v = cyl_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center = hole_pos + hole_mat @ np.array([0.1, 0, 0])

        t = (center - cyl_pos) @ v / (np.linalg.norm(v) ** 2)
        d = np.linalg.norm(np.cross(v, cyl_pos - center)) / np.linalg.norm(v)

        hole_normal = hole_mat @ np.array([0, 0, 1])
        return (
            t,
            d,
            abs(
                np.dot(hole_normal, v) / np.linalg.norm(hole_normal) / np.linalg.norm(v)
            ),
        )

    def reward(self, action):
        """
        Reward function for the task.

        The sparse reward is 0 if the peg is outside the hole, and 1 if it's inside.
        We enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        The dense reward has four components.

            Reaching: in [0, 1], to encourage the arms to get together.
            Perpendicular and parallel distance: in [0,1], for the same purpose.
            Cosine of the angle: in [0, 1], to encourage having the right orientation.
        """
        reward = 0

        t, d, cos = self._compute_orientation()

        # Right location and angle
        if d < 0.06 and t >= -0.12 and t <= 0.14 and cos > 0.95:
            reward = 1

        # use a shaping reward
        if self.reward_shaping:
            # reaching reward
            hole_pos = self.sim.data.body_xpos[self.hole_body_id]
            gripper_site_pos = self.sim.data.body_xpos[self.cyl_body_id]
            dist = np.linalg.norm(gripper_site_pos - hole_pos)
            reaching_reward = 1 - np.tanh(1.0 * dist)
            reward += reaching_reward

            # Orientation reward
            reward += 1 - np.tanh(d)
            reward += 1 - np.tanh(np.abs(t))
            reward += cos

        return reward

    def _peg_pose_in_hole_frame(self):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.
        """
        # World frame
        peg_pos_in_world = self.sim.data.get_body_xpos("cylinder")
        peg_rot_in_world = self.sim.data.get_body_xmat("cylinder").reshape((3, 3))
        peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)

        # World frame
        hole_pos_in_world = self.sim.data.get_body_xpos("hole")
        hole_rot_in_world = self.sim.data.get_body_xmat("hole").reshape((3, 3))
        hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)

        world_pose_in_hole = T.pose_inv(hole_pose_in_world)

        peg_pose_in_hole = T.pose_in_A_to_pose_in_B(
            peg_pose_in_world, world_pose_in_hole
        )
        return peg_pose_in_hole

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
            # position and rotation of cylinder and hole
            hole_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
            hole_quat = T.convert_quat(
                self.sim.data.body_xquat[self.hole_body_id], to="xyzw"
            )
            di["hole_pos"] = hole_pos
            di["hole_quat"] = hole_quat

            cyl_pos = np.array(self.sim.data.body_xpos[self.cyl_body_id])
            cyl_quat = T.convert_quat(
                self.sim.data.body_xquat[self.cyl_body_id], to="xyzw"
            )
            di["cyl_to_hole"] = cyl_pos - hole_pos
            di["cyl_quat"] = cyl_quat

            # Relative orientation parameters
            t, d, cos = self._compute_orientation()
            di["angle"] = cos
            di["t"] = t
            di["d"] = d

            di["object-state"] = np.concatenate(
                [
                    di["hole_pos"],
                    di["hole_quat"],
                    di["cyl_to_hole"],
                    di["cyl_quat"],
                    [di["angle"]],
                    [di["t"]],
                    [di["d"]],
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
        Returns True if task is successfully completed.
        """
        t, d, cos = self._compute_orientation()

        return d < 0.06 and t >= -0.12 and t <= 0.14 and cos > 0.95
