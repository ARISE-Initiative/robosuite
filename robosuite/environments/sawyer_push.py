from collections import OrderedDict
import numpy as np
from copy import deepcopy

from robosuite.utils.mjcf_utils import range_to_uniform_grid
from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.sawyer import SawyerEnv
from robosuite.environments.sawyer_lift import SawyerLift
import robosuite.utils.env_utils as EU

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import TableTopTask, UniformRandomSampler, RoundRobinSampler, TableTopVisualTask
import hjson
import os


class SawyerPush(SawyerLift):
    """
    This class corresponds to the pushing task for the Sawyer robot arm.
    """

    def __init__(
        self,
        placement_initializer=None,
        target_color=(0, 1, 0, 0.3),
        hide_target=False,
        goal_tolerance=0.05,
        goal_radius_low=0.1,
        goal_radius_high=0.2,
        **kwargs
    ):
        """
        See SawyerLift for more args.
        """
        self._target_name = 'cube_target'
        self._object_name = 'cube'
        self._target_rgba = target_color

        # NOTE: these are ignored for now
        self._goal_tolerance = goal_tolerance
        self._goal_radius_low = goal_radius_low
        self._goal_radius_high = goal_radius_high
        self._goal_grid = None

        self._hide_target = hide_target
        if self._hide_target:
            self._target_rgba = (0., 0., 0., 0.,)

        # can override placement initializer here
        if placement_initializer is None:
            placement_initializer = UniformRandomSampler(
                x_range=[-0.16, -0.1],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                z_rotation=0.,
                # x_range=[-0.03, 0.03],
                # y_range=[-0.03, 0.03],
                # ensure_object_boundary_in_range=False,
                # z_rotation=True,
            )
        kwargs['placement_initializer'] = placement_initializer

        super().__init__(
            **kwargs
        )

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = (-0.16, -0.1, 3)
        y_bounds = (-0.03, 0.03, 3)
        z_rot_bounds = (1., 1., 1)
        return x_bounds, y_bounds, z_rot_bounds

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # initialize objects of interest
        cube = BoxObject(
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1, 0, 0, 1],
        )
        self.mujoco_objects = OrderedDict([(self._object_name, cube)])

        # target visual object
        target_size = np.array(self.mujoco_objects[self._object_name].size)
        target = BoxObject(
            size_min=target_size,
            size_max=target_size,
            rgba=self._target_rgba,
        )
        self.visual_objects = OrderedDict([(self._target_name, target)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopVisualTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.visual_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()
        self.model.place_visual()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        self.object_body_id = self.sim.model.body_name2id(self._object_name)
        self.target_body_id = self.sim.model.body_name2id(self._target_name)
        target_qpos = self.sim.model.get_joint_qpos_addr(self._target_name)
        target_qvel = self.sim.model.get_joint_qvel_addr(self._target_name)
        self._ref_target_pos_low, self._ref_target_pos_high = target_qpos
        self._ref_target_vel_low, self._ref_target_vel_high = target_qvel

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # reset positions of objects
        self.model.place_objects()
        self.model.place_visual()

        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

        # for now, place target 0.5 in front of cube location
        cube_pos = np.array(self.sim.data.body_xpos[self.object_body_id])
        cube_pos[0] += 0.3
        self._set_target(pos=cube_pos)

    def _pre_action(self, action, policy_step=None):
        """
        Last gripper dimensions of action are ignored.
        """

        # close gripper
        action[-self.gripper.dof:] = 1.

        super()._pre_action(action, policy_step=policy_step)

        # gravity compensation for target object
        self.sim.data.qfrc_applied[
                self._ref_target_vel_low : self._ref_target_vel_high
            ] = self.sim.data.qfrc_bias[
                self._ref_target_vel_low : self._ref_target_vel_high
            ]

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

            # reaching reward for object close to target
            object_pos = self.sim.data.body_xpos[self.object_body_id]
            dist = np.linalg.norm(object_pos[:2] - self.target_pos[:2])
            reaching_reward = 1. - np.tanh(10.0 * dist)
            reward += reaching_reward

        return reward

    def _set_target(self, pos, quat=None):
        """
        Set the target position and quaternion.
        Quaternion should be (x, y, z, w).
        """
        EU.set_body_pose(self.sim, self._target_name, pos=pos, quat=quat)
        self.sim.forward()

    @property
    def target_pos(self):
        return np.array(self.sim.data.body_xpos[self.target_body_id])

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

        # add in target position
        if self.use_object_obs:
            di["object-state"] = np.concatenate(
                [di["object-state"], self.target_pos]
            )
        return di

    def _check_success(self):
        """
        Returns True if task has been completed.
        """

        # successful if object within close range of target
        object_pos = self.sim.data.body_xpos[self.object_body_id]
        return (np.linalg.norm(object_pos[:2] - self.target_pos[:2]) <= 0.05)

### Some new environments... ###

class SawyerPushPosition(SawyerPush):
    """
    Cube is initialized with a constant z-rotation of 0.
    If using OSC control, force control to be position-only.
    """
    def __init__(
        self,
        **kwargs
    ):
        assert("placement_initializer" not in kwargs)
        kwargs["placement_initializer"] = UniformRandomSampler(
            x_range=[-0.03, 0.03],
            y_range=[-0.03, 0.03],
            ensure_object_boundary_in_range=False,
            z_rotation=0.
        )
        if kwargs["controller"] == "position_orientation":
            kwargs["controller"] = "position"
        super(SawyerPushPosition, self).__init__(**kwargs)

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """

        # (low, high, number of grid points for this dimension)
        x_bounds = (-0.03, 0.03, 3)
        y_bounds = (-0.03, 0.03, 3)
        z_rot_bounds = (0., 0., 1)
        return x_bounds, y_bounds, z_rot_bounds


