from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments import MujocoEnv

from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import Panda
from robosuite.controllers.arm_controller import *
import pdb
import mujoco_py

from mujoco_py import load_model_from_xml, MjSim, functions
from enum import Enum

from collections import deque

#import cv2
#import multiprocessing

#from baselines.baselines import logger
import logging
logger = logging.getLogger(__name__)
import time
import hjson


class PandaRobotArmEnv(MujocoEnv):
    """Initializes a robot environment."""

    def __init__(
            self,
            controller_config_file,
            logging_filename,
            only_cartesian_obs,
            data_logging,
            reward_scale,
            controller,
            gripper_type=None,
            gripper_visualization=False,
            use_indicator_object=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=100,
            horizon=1000,
            ignore_done=False,
            use_camera_obs=False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
            impedance_ctrl=True,    # TODO
            initial_policy=None,    # TODO - currently not included in the config file (should be a function)
            **kwargs
    ):

        """
        Args:
            controller_config_file (str): filepath to the corresponding controller config file that contains the
                associated controller parameters

            logging_filename (str): name of h5py to log to (if data_logging is true)

            only_cartesian_obs (bool): True if no joint space observations are included.

            data_logging (bool): True if the h5py file given by logging_filename is to be used TODO - maybe combine with logging_filename

            reward_scale (float): Multiplier for final reward

            controller (str): Can be 'position', 'position_orientation', 'joint_velocity', 'joint_impedance', or
                'joint_torque'. Specifies the type of controller to be used for dynamic trajectories

            inertia_decoupling (bool): True if the robot's motion is decoupled.

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

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

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.

            impedance_ctrl (bool) : True if we want to control impedance of the end effector

        #########
            **kwargs includes additional params that may be specified and will override values found in
            the controller configuration file if the names match
        """

        self.reward_range = (float('-inf'), float('inf'))  # TODO
        from argparse import Namespace
        self.spec = Namespace(max_episode_steps=horizon)  # TODO

        def seed(seed):
            pass

        self.seed = seed
        self.has_gripper = gripper_type is not None
        self.gripper_type = gripper_type
        self.gripper_visualization = gripper_visualization
        self.use_indicator_object = use_indicator_object

        self.impedance_ctrl = impedance_ctrl
        if self.impedance_ctrl:
            # Load the appropriate controller
            self._load_controller(controller, controller_config_file)

        self.goal = np.zeros(3)
        self.goal_orientation = np.zeros(3)
        self.desired_force = np.zeros(3)
        self.desired_torque = np.zeros(3)
        if 'residual_policy_multiplier' in kwargs:
            self.residual_policy_multiplier = kwargs['residual_policy_multiplier']
        else:
            self.residual_policy_multiplier = None

        self.initial_policy = initial_policy

        self.only_cartesian_obs = only_cartesian_obs

        self.control_freq = control_freq
        self.timestep = 0
        self.data_logging = data_logging
        self.reward_scale = reward_scale

        # self.position_limits = [[0,0,0],[0,0,0]]
        # self.orientation_limits = [[0,0,0],[0,0,0]]

        self.ee_force = np.zeros(3)
        self.ee_force_bias = np.zeros(3)
        self.contact_threshold = 1  # Maximum contact variation allowed without contact [N]

        self.ee_torque = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        #self.controller = controller
        # TODO - check that these are updated properly
        self.total_kp = np.zeros(6)
        self.total_damping = np.zeros(6)

        self.n_avg_ee_acc = 10

        super(PandaRobotArmEnv, self).__init__(
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
            camera_width=camera_height,
            camera_depth=camera_depth,
        )

        # Current and previous policy step q values, joint torques, ft ee applied and actions
        self.prev_pstep_ft = np.zeros(6)
        self.curr_pstep_ft = np.zeros(6)
        self.prev_pstep_a = np.zeros(self.dof)
        self.curr_pstep_a = np.zeros(self.dof)
        self.prev_pstep_q = np.zeros(len(self._ref_joint_vel_indexes))
        self.curr_pstep_q = np.zeros(len(self._ref_joint_vel_indexes))
        self.prev_pstep_t = np.zeros(len(self._ref_joint_vel_indexes))
        self.curr_pstep_t = np.zeros(len(self._ref_joint_vel_indexes))
        self.prev_pstep_ee_v = np.zeros(6)
        self.curr_pstep_ee_v = np.zeros(6)
        self.buffer_pstep_ee_v = deque(np.zeros(6) for _ in range(self.n_avg_ee_acc))
        self.ee_acc = np.zeros(6)

        self.total_ee_acc = np.zeros(6)  # used to compute average
        self.total_js_energy = np.zeros(len(self._ref_joint_vel_indexes))

        self.torque_total = 0
        self.joint_torques = 0

        self.prev_ee_pos = np.zeros(7)
        self.ee_pos = np.zeros(7)

        ## counting joint limits
        self.joint_limit_count = 0

        if self.data_logging:
            ########################## DATA LOGGING #################
            self.counter = 0
            import h5py
            import os
            import datetime
            self.data_count = kwargs['horizon']

            self.logging_filename = logging_filename
            logger.info("filename for h5py: %s", self.logging_filename)
            try:
                self.file_logging = h5py.File(self.logging_filename, 'w')
            except Exception as e:
                logger.exception(e)
                self.file_logging = h5py.File(logging_filename, 'r+')

            self.file_logging.create_dataset("action", (self.data_count, self.dof), maxshape=(None, self.dof))
            self.file_logging.create_dataset("transformed_action", (self.data_count, self.dof),
                                             maxshape=(None, self.dof))
            self.file_logging.create_dataset("joint_pos", (self.data_count, 7),
                                             maxshape=(None, len(self._ref_joint_vel_indexes)))
            self.file_logging.create_dataset("joint_vel", (self.data_count, 7),
                                             maxshape=(None, len(self._ref_joint_vel_indexes)))
            self.file_logging.create_dataset("ee_pos", (self.data_count, 3), maxshape=(None, 3))
            self.file_logging.create_dataset("ee_ori", (self.data_count, 4), maxshape=(None, 4))
            self.file_logging.create_dataset("ee_vel", (self.data_count, 3), maxshape=(None, 3))
            self.file_logging.create_dataset("ee_vel_ori", (self.data_count, 3), maxshape=(None, 3))
            self.file_logging.create_dataset("ee_acc", (self.data_count, 6), maxshape=(None, 6))
            self.file_logging.create_dataset("js_energy", (self.data_count, 7), maxshape=(None, 7))
            self.file_logging.create_dataset("reward", (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset("ee_force", (self.data_count, 3), maxshape=(None, 3))
            self.file_logging.create_dataset("joint_limit_count", (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset("ee_torque", (self.data_count, 3), maxshape=(None, 3))
            self.file_logging.create_dataset("joint_torques", (self.data_count, 7), maxshape=(None, 7))

            try:
                impedance_dim = self.controller.impedance_kp.shape[0]
                self.file_logging.create_dataset("impedance_kp", (self.data_count, impedance_dim),
                                                 maxshape=(None, impedance_dim))
                self.file_logging.create_dataset("impedance_damping", (self.data_count, impedance_dim),
                                                 maxshape=(None, impedance_dim))
                self.file_logging.create_dataset("impedance_kv", (self.data_count, impedance_dim),
                                                 maxshape=(None, impedance_dim))
            except:
                pass  # if a controller doesn't have impedance, that's fine

    ########################## DATA LOGGING #################

    def _load_controller(self, controller_type, controller_file, kwargs):
        """
        Loads controller to be used for dynamic trajectories

        Controller_type is a specified controller, and controller_params is a config file containing the appropriate
        parameters for that controller

        Kwargs is kwargs passed from init call and represents individual params to override in controller config file
        """

        # Load the controller config file
        try:
            with open(controller_file) as f:
                params = hjson.load(f)
        except FileNotFoundError:
            print("Controller config file '{}' not found. Please check filepath and try again.".format(
                controller_file))

        controller_params = params[controller_type]

        # Load additional arguments from kwargs and override the prior config-file loaded ones
        for key, value in kwargs.items():
            if key in controller_params:
                controller_params[key] = value

        if controller_type == ControllerType.POS:
            self.controller = PositionController(**controller_params)
        elif controller_type == ControllerType.POS_ORI:
            self.controller = PositionOrientationController(**controller_params)
        elif controller_type == ControllerType.JOINT_IMP:
            self.controller = JointImpedanceController(**controller_params)
        elif controller_type == ControllerType.JOINT_TORQUE:
            self.controller = JointTorqueController(**controller_params)
        else:
            self.controller = JointVelocityController(**controller_params)

    def _load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        super()._load_model()
        self.mujoco_robot = Panda(xml_path="robots/panda/robot_torque.xml")

        if self.has_gripper:
            self.gripper = gripper_factory(self.gripper_type)
            if not self.gripper_visualization:
                self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)

        self.controller.initial_joint = self.mujoco_robot.init_qpos

    def _reset_internal(self):
        """
        Sets initial pose of arm and grippers.
        """
        super()._reset_internal()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos

        if self.has_gripper:
            self.sim.data.qpos[
                self._ref_joint_gripper_actuator_indexes
            ] = self.gripper.init_qpos

        self.goal = np.zeros(3)
        self.goal_orientation = np.zeros(3)
        self.desired_force = np.zeros(3)
        self.desired_torque = np.zeros(3)
        self.prev_pstep_q = np.array(self.mujoco_robot.init_qpos)
        self.curr_pstep_q = np.array(self.mujoco_robot.init_qpos)
        self.prev_pstep_a = np.zeros(self.dof)
        self.curr_pstep_a = np.zeros(self.dof)
        self.prev_pstep_ee_v = np.zeros(6)
        self.curr_pstep_ee_v = np.zeros(6)
        self.buffer_pstep_ee_v = deque(np.zeros(6) for _ in range(self.n_avg_ee_acc))
        self.ee_acc = np.zeros(6)
        self.total_ee_acc = np.zeros(6)  # used to compute average
        self.total_kp = np.zeros(6)
        self.total_damping = np.zeros(6)
        self.total_js_energy = np.zeros(len(self._ref_joint_vel_indexes))
        self.prev_ee_pos = np.zeros(7)
        self.ee_pos = np.zeros(7)
        self.total_joint_torque = 0
        self.joint_torques = 0

    def _get_reference(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        super()._get_reference()

        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]

        if self.use_indicator_object:
            ind_qpos = self.sim.model.get_joint_qpos_addr("pos_indicator")
            self._ref_indicator_pos_low, self._ref_indicator_pos_high = ind_qpos

            ind_qvel = self.sim.model.get_joint_qvel_addr("pos_indicator")
            self._ref_indicator_vel_low, self._ref_indicator_vel_high = ind_qvel

            self.indicator_id = self.sim.model.body_name2id("pos_indicator")

        # indices for grippers in qpos, qvel
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self._ref_gripper_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
            ]
            self._ref_gripper_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
            ]

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("pos")
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("vel")
        ]

        if self.has_gripper:
            self._ref_joint_gripper_actuator_indexes = [
                self.sim.model.actuator_name2id(actuator)
                for actuator in self.sim.model.actuator_names
                if actuator.startswith("gripper")
            ]

        # IDs of sites for gripper visualization
        self.visual_grip_site_id = self.sim.model.site_name2id("grip_site")
        self.visual_grip_cylinder_id = self.sim.model.site_name2id("grip_site_cylinder")

    def move_indicator(self, pos):
        """
        Sets 3d position of indicator object to @pos.
        """
        if self.use_indicator_object:
            index = self._ref_indicator_pos_low
            self.sim.data.qpos[index: index + 3] = pos

    def _pre_action(self, action, policy_step): # Removed policy_step from arg list in _pre_action
        """
        Overrides the superclass method to actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.mujoco_robot.dof dimensions should be the desired
                normalized joint velocities and if the robot has
                a gripper, the next @self.gripper.dof dimensions should be
                actuation controls for the gripper.
        """

        self.policy_step = policy_step

        # Make sure action length is correct
        assert len(action) == self.dof, "environment got invalid action dimension"

        # i.e.: not using new controller
        if not self.impedance_ctrl:

            # clip actions into valid range
            low, high = self.action_spec
            action = np.clip(action, low, high)

            if self.has_gripper:
                arm_action = action[: self.mujoco_robot.dof]
                gripper_action_in = action[
                                    self.mujoco_robot.dof: self.mujoco_robot.dof + self.gripper.dof
                                    ]
                gripper_action_actual = self.gripper.format_action(gripper_action_in)
                action = np.concatenate([arm_action, gripper_action_actual])

            # rescale normalized action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_action = bias + weight * action
            self.sim.data.ctrl[self._ref_joint_vel_indexes] = applied_action

            # gravity compensation
            self.sim.data.qfrc_applied[
                self._ref_joint_vel_indexes
            ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

            if self.use_indicator_object:
                self.sim.data.qfrc_applied[
                self._ref_indicator_vel_low: self._ref_indicator_vel_high
                ] = self.sim.data.qfrc_bias[
                    self._ref_indicator_vel_low: self._ref_indicator_vel_high
                    ]

        # using new controller
        else:
            # Split action into joint control and peripheral (i.e.: gripper) control (as specified by individual gripper)
            gripper_action = []
            if self.has_gripper:
                gripper_action = action[self.controller.control_dim:]  # all indexes past controller dimension indexes
                action = action[:self.controller.control_dim]


            # TODO
            # First, get joint space action
            #action = action.copy()  # ensure that we don't change the action outside of this scope
            self.controller.update_model(self.sim, id_name='right_hand', joint_index=self._ref_joint_pos_indexes)
            torques = self.controller.action_to_torques(action,
                                                        self.policy_step)  # this scales and clips the actions correctly

            if self.initial_policy:
                initial_policy_torques = self.initial_policy.action_to_torques(self.sim, 'right_hand',
                                                                               self._ref_joint_pos_indexes,
                                                                               self.initial_policy_action,
                                                                               self.policy_step)
                self.residual_torques = torques
                self.initial_torques = initial_policy_torques
                if self.residual_policy_multiplier is not None:
                    torques = self.residual_policy_multiplier*torques + initial_policy_torques
                else:
                    torques = torques + initial_policy_torques  # TODO

            self.total_joint_torque += np.sum(abs(torques))
            self.joint_torques = torques

            # Get gripper action, if applicable
            if self.has_gripper:
                gripper_action_actual = self.gripper.format_action(gripper_action)
                # rescale normalized gripper action to control ranges
                ctrl_range = self.sim.model.actuator_ctrlrange[self._ref_gripper_joint_vel_indexes]
                bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
                weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
                applied_gripper_action = bias + weight * gripper_action_actual
                self.sim.data.ctrl[self._ref_gripper_joint_vel_indexes] = applied_gripper_action

            # Now, control both gripper and joints
            self.sim.data.ctrl[self._ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes] + torques

            if self.policy_step:
                self.prev_pstep_q = np.array(self.curr_pstep_q)
                self.curr_pstep_q = np.array(self.sim.data.qpos[self._ref_joint_vel_indexes])
                self.prev_pstep_a = np.array(self.curr_pstep_a)
                if self.has_gripper:
                    self.curr_pstep_a = np.concatenate((action, gripper_action))#.copy()) # TODO
                else:
                    self.curr_pstep_a = np.array(action)
                self.prev_pstep_t = np.array(self.curr_pstep_t)
                self.curr_pstep_t = np.array(self.sim.data.ctrl[self._ref_joint_vel_indexes])
                self.prev_pstep_ft = np.array(self.curr_pstep_ft)


                # Assumes a ft sensor on the wrist
                force_sensor_id = self.sim.model.sensor_name2id("force_ee")
                force_ee = self.sim.data.sensordata[force_sensor_id * 3: force_sensor_id * 3 + 3]
                torque_sensor_id = self.sim.model.sensor_name2id("torque_ee")
                torque_ee = self.sim.data.sensordata[torque_sensor_id * 3: torque_sensor_id * 3 + 3]
                self.curr_pstep_ft = np.concatenate([force_ee, torque_ee])

                self.prev_pstep_ee_v = self.curr_pstep_ee_v
                self.curr_pstep_ee_v = np.concatenate(
                    [self.sim.data.body_xvelp[self.sim.model.body_name2id("right_hand")],
                     self.sim.data.body_xvelr[self.sim.model.body_name2id("right_hand")]])

                self.buffer_pstep_ee_v.popleft()
                self.buffer_pstep_ee_v.append(self.curr_pstep_ee_v)

                # convert to matrix
                buffer_mat = []
                for v in self.buffer_pstep_ee_v:
                    buffer_mat += [v]
                buffer_mat = np.vstack(buffer_mat)

                diffs = np.diff(buffer_mat, axis=0)
                diffs *= self.control_freq
                diffs = np.vstack([self.ee_acc, diffs])
                diffs.reshape((self.n_avg_ee_acc, 6))

                self.ee_acc = np.array(
                    [np.convolve(col, np.ones((self.n_avg_ee_acc,)) / self.n_avg_ee_acc, mode='valid')[0] for col in
                     diffs.transpose()])


    def _post_action(self, action):
        """
        (Optional) does gripper visualization after actions.
        """
        self.prev_ee_pos = self.ee_pos
        self.ee_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])


        force_sensor_id = self.sim.model.sensor_name2id("force_ee")
        self.ee_force = np.array(self.sim.data.sensordata[force_sensor_id * 3: force_sensor_id * 3 + 3])

        if np.linalg.norm(self.ee_force_bias) == 0:
            self.ee_force_bias = self.ee_force

        torque_sensor_id = self.sim.model.sensor_name2id("torque_ee")
        self.ee_torque = np.array(self.sim.data.sensordata[torque_sensor_id * 3: torque_sensor_id * 3 + 3])

        if np.linalg.norm(self.ee_torque_bias) == 0:
            self.ee_torque_bias = self.ee_torque


        reward, done, info = super()._post_action(action)
        reward *= self.reward_scale

        info['add_vals'] = []
        # Note: custom values (i.e. not eprewmean) are only logged at the last timestep of the episode
        try:
            # log kp
            kp_indices = self.controller.kp_index
            self.total_kp += self.controller.impedance_kp
            for idx, val in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz']):
                key = 'kp_{}_'.format(val)
                info['add_vals'].append(key)
                info[key] = self.total_kp[idx] / self.timestep

            # log kv
            damping_indices = self.controller.damping_index
            self.total_damping += self.controller.damping
            for idx, val in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz']):
                key = 'damping_{}_'.format(val)
                info['add_vals'].append(key)
                info[key] = self.total_damping[idx] / self.timestep
        except:
            pass

        # log energy per joint
        self.total_js_energy += self._compute_js_energy()
        for idx, energy in enumerate(self.total_js_energy):
            key = 'energy_j{}_'.format(idx)
            info['add_vals'].append(key)
            info[key] = energy / self.timestep

        info['add_vals'].append('total_js_energy')
        info['total_js_energy'] = np.sum(self.total_js_energy) / self.timestep

        # maintain mean of acceleration of end effector
        self.total_ee_acc += abs(self.ee_acc)
        for idx, val in enumerate(['vx', 'vy', 'vz', 'rx', 'ry', 'rz']):
            key = 'eef_acc_' + val + '_'
            info['add_vals'].append(key)
            info[key] = self.total_ee_acc[idx] / self.timestep

        info['add_vals'].append('total_ee_acc')
        info['total_ee_acc'] = np.sum(self.total_ee_acc) / self.timestep

        self._gripper_visualization()

        num_steps = self.control_timestep / self.model_timestep
        info['add_vals'].append('avg_joint_torque')
        info['avg_joint_torque'] = np.array(self.total_joint_torque)/num_steps

        if self.initial_policy:
            info['add_vals'].append('residual_similarity_')
            info['residual_similarity_'] = np.dot(self.initial_torques, self.residual_torques) / (
                        np.linalg.norm(self.initial_torques) * (np.linalg.norm(self.residual_torques)))

        ###### DATA LOGGING #######
        if self.data_logging:
            # Note that done here doesn't reflect task-specific doneness
            if self.counter < self.data_count:
                self.file_logging['action'][self.counter] = action
                self.file_logging['transformed_action'][self.counter] = self.controller.transform_action(action)
                self.file_logging['joint_pos'][self.counter] = np.array(
                    [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes])
                self.file_logging['joint_vel'][self.counter] = np.array(
                    [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes])
                self.file_logging['ee_pos'][self.counter] = np.array(
                    self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
                self.file_logging['ee_ori'][self.counter] = T.convert_quat(
                    self.sim.data.get_body_xquat("right_hand"), to="xyzw"
                )
                self.file_logging['ee_vel'][self.counter] = np.array(
                    self.sim.data.body_xvelp[self.sim.model.body_name2id('right_hand')])
                self.file_logging['ee_vel_ori'][self.counter] = np.array(
                    self.sim.data.body_xvelr[self.sim.model.body_name2id('right_hand')])
                self.file_logging['ee_acc'][self.counter] = abs(self.ee_acc)
                self.file_logging['js_energy'][self.counter] = np.array(self._compute_js_energy())

                try:
                    self.file_logging['impedance_kp'][self.counter] = np.array(self.controller.impedance_kp)
                    self.file_logging['impedance_damping'][self.counter] = np.array(self.controller.damping)
                    self.file_logging['impedance_kv'][self.counter] = np.array(self.controller.impedance_kv)
                except Exception as e:
                    # if impedance isn't enabled, don't worry too much
                    pass
                self.file_logging['ee_force'][self.counter] = np.array(self.ee_force)
                self.file_logging['ee_torque'][self.counter] = np.array(self.ee_torque)
                self.file_logging['joint_limit_count'][self.counter] = np.array([self.joint_limit_count])
                self.file_logging['joint_torques'][self.counter] = self.joint_torques
                self.counter += 1
            else:
                logger.info("Logging file: %s", self.logging_filename)
                self.file_logging.flush()
                self.file_logging.close()
                exit(0)
        ###### DATA LOGGING #######
        return reward, done, info

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
        """

        di = super()._get_observation()

        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(camera_name=self.camera_name,
                                         width=self.camera_width,
                                         height=self.camera_height,
                                         depth=self.camera_depth)
            if self.camera_depth:
                di['image'], di['depth'] = camera_obs
            else:
                di['image'] = camera_obs

                # Skip for now, not worth importing cv2 just for this
                # if self.visualize_offscreen and not self.real_robot:
                    # cv2.imshow('Robot observation', np.flip(camera_obs[..., ::-1], 0))
                    # cv2.waitKey(10)

        # proprioceptive features
        di["joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        )
        di["joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        )

        if self.only_cartesian_obs:
            robot_states = []
        else:
            robot_states = [
                np.sin(di["joint_pos"]),
                np.cos(di["joint_pos"]),
                di["joint_vel"],
            ]

        if self.has_gripper:
            di["gripper_qpos"] = np.array(
                [self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes]
            )
            di["gripper_qvel"] = np.array(
                [self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes]
            )

            di["eef_pos"] = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
            di["eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )
            di["eef_vlin"] = np.array(self.sim.data.get_body_xvelp('right_hand'))
            di["eef_vang"] = np.array(self.sim.data.get_body_xvelr('right_hand'))

            # add in gripper information
            robot_states.extend([di["gripper_qpos"], di["eef_pos"], di["eef_quat"], di["eef_vlin"], di["eef_vang"]])

        di["robot-state"] = np.concatenate(robot_states)

        di["prev-act"] = self.prev_pstep_a

        # Adding binary contact observation
        in_contact = np.linalg.norm(self.ee_force - self.ee_force_bias) > self.contact_threshold
        di["contact-obs"] = in_contact

        return di

    @property
    def joint_damping(self):
        return self.sim.model.dof_damping

    @property
    def joint_frictionloss(self):
        return self.sim.model.dof_frictionloss

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """

        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.

        return low, high

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with grippers).
        """
        if self.impedance_ctrl:
            dof = self.controller.action_dim
        else:
            dof = self.mujoco_robot.dof

        if self.has_gripper:
            dof += self.gripper.dof
            return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def _right_hand_joint_cartesian_pose(self):
        """
        Returns the cartesian pose of the last robot joint in base frame of robot.
        """
        return self.pose_in_base_from_name("right_l6")

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_quat(self):
        """
        Returns eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame
        as a numpy array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp("right_hand").reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes]

        Jr = self.sim.data.get_body_jacr("right_hand").reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[:3]

    @property
    def _right_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[3:]

    @property
    def _joint_positions(self):
        """
        Returns a numpy array of joint positions.
        Panda robots have 7 joints and positions are in rotation angles.
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        Panda robots have 7 joints and velocities are angular velocities.
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    def _gripper_visualization(self):
        """
        Do any needed visualization here.
        """

        # By default, don't do any coloring.
        self.sim.model.site_rgba[self.visual_grip_site_id] = [0., 0., 0., 0.]

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False

    def _check_arm_contact(self):
        """
        Returns True if the arm is in contact with another object.
        """
        collision = False
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if self.sim.model.geom_id2name(contact.geom1) in self.mujoco_robot.contact_geoms or \
                    self.sim.model.geom_id2name(contact.geom2) in self.mujoco_robot.contact_geoms:
                collision = True
                break
        return collision

    def _check_q_limits(self):
        """
        Returns True if the arm is in joint limits or very close to.
        """
        joint_limits = False
        tolerance = 0.1
        for (idx, (q, q_limits)) in enumerate(
                zip(self.sim.data.qpos[self._ref_joint_pos_indexes], self.sim.model.jnt_range)):
            if not (q > q_limits[0] + tolerance and q < q_limits[1] - tolerance):
                print("Joint limit reached in joint " + str(idx))
                joint_limits = True
                self.joint_limit_count += 1
        return joint_limits

    def _compute_q_delta(self):
        """
        Returns the change in joint space configuration between previous and current steps
        """
        q_delta = self.prev_pstep_q - self.curr_pstep_q

        return q_delta

    def _compute_t_delta(self):
        """
        Returns the change in joint space configuration between previous and current steps
        """
        t_delta = self.prev_pstep_t - self.curr_pstep_t

        return t_delta

    def _compute_a_delta(self):
        """
        Returns the change in policy action between previous and current steps
        """
        a_delta = self.prev_pstep_a - self.curr_pstep_a

        return a_delta

    def _compute_ft_delta(self):
        """
        Returns the change in policy action between previous and current steps
        """

        ft_delta = self.prev_pstep_ft - self.curr_pstep_ft

        return ft_delta

    def _compute_js_energy(self):
        """
        Returns the energy consumed by each joint between previous and current steps
        """
        # Mean torque applied
        mean_t = self.prev_pstep_t - self.curr_pstep_t

        # We assume in the motors torque is proportional to current (and voltage is constant)
        # In that case the amount of power scales proportional to the torque and the energy is the
        # time integral of that
        js_energy = np.abs((1.0 / self.control_freq) * mean_t)

        return js_energy

    def _compute_ee_ft_integral(self):
        """
        Returns the integral over time of the applied ee force-torque
        """

        mean_ft = self.prev_pstep_ft - self.curr_pstep_ft
        integral_ft = np.abs((1.0 / self.control_freq) * mean_ft)

        return integral_ft

    def render_additional_image(self, camera_name, camera_width, camera_height, camera_depth):
        img = self.sim.render(camera_name=camera_name,
                              width=camera_width,
                              height=camera_height,
                              depth=camera_depth)
        return img

