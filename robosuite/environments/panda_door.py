import numpy as np
from collections import OrderedDict
from robosuite.utils import RandomizationError
from robosuite.environments.panda_robot_arm import PandaRobotArmEnv
from robosuite.models import *
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.tasks import Task, UniformRandomSampler, HeightTableTask, DoorTask
from robosuite.models.arenas import HeightTableArena, EmptyArena, TableArena
from robosuite.models.objects import CylinderObject, PlateWithHoleObject, BoxObject, DoorObject
import multiprocessing
from robosuite.controllers.arm_controller import *

import robosuite.utils.transform_utils as T

import logging
logger = logging.getLogger(__name__)
import hjson
import os


class PandaDoor(PandaRobotArmEnv):

    def __init__(
            self,
            gripper_type="PandaGripper",
            use_camera_obs=True,
            use_object_obs=True,
            reward_shaping=False,  # TODO: no shaping option currently
            placement_initializer=None,
            gripper_visualization=False,
            use_indicator_object=False,
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
            use_default_task_config=True,
            task_config_file=None,
            use_default_controller_config=True,
            controller_config_file=None,
            controller='joint_velocity',
            **kwargs
    ):
        """
        Args:

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

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

            use_default_task_config (bool): True if using default configuration file
                for remaining environment parameters. Default is true

            task_config_file (str): filepath to configuration file to be
                used for remaining environment parameters (taken relative to head of robosuite repo).

            use_default_controller_config (bool): True if using default configuration file
                for remaining environment parameters. Default is true

            controller_config_file (str): filepath to configuration file to be
                used for remaining environment parameters (taken relative to head of robosuite repo).

            controller (str): Can be 'position', 'position_orientation', 'joint_velocity', 'joint_impedance', or
                'joint_torque'. Specifies the type of controller to be used for dynamic trajectories

            controller_config_file (str): filepath to the corresponding controller config file that contains the
                associated controller parameters

            #########
            **kwargs includes additional params that may be specified and will override values found in
            the configuration file
        """

        # Load the parameter configuration files
        if use_default_controller_config == True:
            controller_filepath = os.path.join(os.path.dirname(__file__), '..',
                                               'scripts/config/controller_config.hjson')
        else:
            controller_filepath = os.path.join(os.path.dirname(__file__), '..',
                                               controller_config_file)

        if use_default_task_config == True:
            task_filepath = os.path.join(os.path.dirname(__file__), '..',
                                         'scripts/config/Door_task_config.hjson')
        else:
            task_filepath = os.path.join(os.path.dirname(__file__), '..',
                                         task_config_file)

        try:
            with open(task_filepath) as f:
                task = hjson.load(f)
        except FileNotFoundError:
            print("Env Config file '{}' not found. Please check filepath and try again.".format(task_filepath))

        # settings for table top
        self.dist_threshold = task['dist_threshold']
        self.timestep = 0
        self.excess_force_penalty_mul = task['excess_force_penalty_mul']
        self.excess_torque_penalty_mul = task['excess_force_penalty_mul'] * 10.0
        self.torque_threshold_max = task['pressure_threshold_max'] * 0.1
        self.pressure_threshold_max = task['pressure_threshold_max']

        # set reward shaping
        self.energy_penalty = task['energy_penalty']
        self.ee_accel_penalty = task['ee_accel_penalty']
        self.action_delta_penalty = task['action_delta_penalty']
        self.handle_reward = task['handle_reward']
        self.arm_collision_penalty = task['arm_collision_penalty']

        # TODO: Where do these magic nums come from? Can they be included in the config file?
        self.handle_final_reward = 1
        self.handle_shaped_reward = 0.5
        self.max_hinge_diff = 0.05
        self.max_hinge_vel = 0.1
        self.final_reward = 500
        self.door_shaped_reward = 30
        self.hinge_goal = 1.04
        self.velocity_penalty = 10

        # set what is included in the observation
        self.use_door_state = task['use_door_state']
        self.use_object_obs = use_object_obs  # ground-truth object states

        # door friction
        self.change_door_friction = task['change_door_friction']
        self.door_damping_max = task['door_damping_max']
        self.door_damping_min = task['door_damping_min']
        self.door_friction_max = task['door_friction_max']
        self.door_friction_min = task['door_friction_min']

        # self.table_full_size = table_full_size
        self.gripper_on_handle = task['gripper_on_handle']

        self.collisions = 0
        self.f_excess = 0
        self.t_excess = 0

        self.placement_initializer = placement_initializer
        self.table_full_size = task['table_full_size']
        self.table_origin = [0.50 + self.table_full_size[0] / 2, 0, 0]

        super(PandaDoor, self).__init__(
            logging_filename=task['logging_filename'],
            only_cartesian_obs=task['only_cartesian_obs'],
            data_logging=task['data_logging'],
            reward_scale=task['reward_scale'],
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
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
            controller_config_file=controller_filepath,
            controller=controller,
            **kwargs
        )

        if self.data_logging:
            self.file_logging.create_dataset('percent_viapoints_', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('hinge_angle', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('hinge_diff', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('hinge_goal', (self.data_count, 1), maxshape=(None, 1))

            self.file_logging.create_dataset('done', (self.data_count, 1), maxshape=(None, 1))

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        self.robot_contact_geoms = self.mujoco_robot.contact_geoms

        self.mujoco_arena = TableArena(table_full_size=(0.8, 0.8, 1.43 - 0.375))

        # The panda robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin(self.table_origin)

        self.mujoco_objects = OrderedDict()
        self.door = DoorObject()
        self.mujoco_objects = OrderedDict([("Door", self.door)])

        # For Panda initialization
        if self.gripper_on_handle:
            self.mujoco_robot._init_qpos = np.array(
                [-0.01068642,-0.05599809,0.22389938,-1.81999415,-1.54907898,2.82220116,2.28768505])

        # task includes arena, robot, and objects of interest
        self.model = DoorTask(self.mujoco_arena,
                              self.mujoco_robot,
                              self.mujoco_objects)

        if self.change_door_friction:
            damping = np.random.uniform(high=np.array([self.door_damping_max]), low=np.array([self.door_damping_min]))
            friction = np.random.uniform(high=np.array([self.door_friction_max]),
                                         low=np.array([self.door_friction_min]))
            self.model.set_door_damping(damping)
            self.model.set_door_friction(friction)

        self.model.place_objects(randomize=self.placement_initializer)

    def _get_reference(self):
        super()._get_reference()

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()
        # reset joint positions
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(self.mujoco_robot.init_qpos)
        self.timestep = 0
        self.wiped_sensors = []
        self.touched_handle = 0
        self.collisions = 0
        self.f_excess = 0
        self.t_excess = 0

    def reward(self, action):
        reward = 0
        grip_id = self.sim.model.site_name2id("grip_site")
        eef_position = self.sim.data.site_xpos[grip_id]

        force_sensor_id = self.sim.model.sensor_name2id("force_ee")
        self.force_ee = self.sim.data.sensordata[force_sensor_id * 3: force_sensor_id * 3 + 3]
        total_force_ee = np.linalg.norm(np.array(self.ee_force))

        torque_sensor_id = self.sim.model.sensor_name2id("torque_ee")
        self.torque_ee = self.sim.data.sensordata[torque_sensor_id * 3: torque_sensor_id * 3 + 3]
        total_torque_ee = np.linalg.norm(np.array(self.torque_ee))

        self.hinge_diff = np.abs(self.hinge_goal - self.hinge_qpos)

        # Neg Reward from collisions of the arm with the table
        if self._check_arm_contact():
            reward = self.arm_collision_penalty
        elif self._check_q_limits():
            reward = self.arm_collision_penalty
        else:

            # add reward for touching handle or being close to it
            if self.handle_reward:
                dist = np.linalg.norm(eef_position[0:2] - self.handle_position[0:2])

                if dist < self.dist_threshold and abs(eef_position[2] - self.handle_position[2]) < 0.02:
                    self.touched_handle = 1
                    reward += self.handle_reward
                else:
                    # if robot starts 0.3 away and dist_threshold is 0.05: [0.005, 0.55] without scaling
                    reward += (self.handle_shaped_reward * (1 - np.tanh(3 * dist))).squeeze()
                    self.touched_handle = 0

            # penalize excess force
            if total_force_ee > self.pressure_threshold_max:
                reward -= self.excess_force_penalty_mul * total_force_ee
                self.f_excess += 1

            # penalize excess torque
            if total_torque_ee > self.torque_threshold_max:
                reward -= self.excess_torque_penalty_mul * total_torque_ee
                self.t_excess += 1

            # award bonus either for opening door or for making process toward it
            if self.hinge_diff < self.max_hinge_diff and abs(self.hinge_qvel) < self.max_hinge_vel:
                reward += self.final_reward

            else:
                reward += (self.door_shaped_reward * (np.abs(self.hinge_goal) - self.hinge_diff)).squeeze()
                reward -= (self.hinge_qvel * self.velocity_penalty).squeeze()

        # penalize for jerkiness
        reward -= self.energy_penalty * np.sum(np.abs(self.joint_torques))
        reward -= self.ee_accel_penalty * np.mean(abs(self.ee_acc))
        reward -= self.action_delta_penalty * np.mean(abs(self._compute_a_delta()[:3]))

        string_to_print = 'Process {pid}, timestep {ts:>4}: reward: {rw:8.4f} hinge diff: {ha} excess-f: {ef}, excess-t: {et}'.format(
            pid=id(multiprocessing.current_process()),
            ts=self.timestep,
            rw=reward,
            con=self._check_contact(),
            ha=self.hinge_diff,
            ef=self.f_excess,
            et=self.t_excess)

        logger.debug(string_to_print)

        return reward

    def _get_observation(self):
        di = super()._get_observation()

        if self.use_camera_obs:
            camera_obs = self.sim.render(camera_name=self.camera_name,
                                         width=self.camera_width,
                                         height=self.camera_height,
                                         depth=self.camera_depth)
            if self.camera_depth:
                di['image'], di['depth'] = camera_obs
            else:
                di['image'] = camera_obs

        if self.use_object_obs:
            # checking if contact is made, add as state
            contact = self._check_contact()
            di['object-state'] = np.array([[0, 1][contact]])

            # door information for rewards
            handle_id = self.sim.model.site_name2id("S_handle")
            self.handle_position = self.sim.data.site_xpos[handle_id]
            handle_orientation_mat = self.sim.data.site_xmat[handle_id].reshape(3, 3)
            handle_orientation = T.mat2quat(handle_orientation_mat)
            hinge_id = self.sim.model.get_joint_qpos_addr("door_hinge")

            self.hinge_qpos = np.array((self.sim.data.qpos[hinge_id])).reshape(-1, )
            self.hinge_qvel = np.array((self.sim.data.qvel[hinge_id])).reshape(-1, )

            if self.use_door_state:
                di['object-state'] = np.concatenate([
                    di['object-state'],
                    self.hinge_qpos,
                    self.hinge_qvel])

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if self.sim.model.geom_id2name(contact.geom1) in self.gripper.contact_geoms() or \
                    self.sim.model.geom_id2name(contact.geom2) in self.gripper.contact_geoms():
                collision = True
                break
        return collision

    def _check_terminated(self):
        """
        Returns True if task is successfully completed
        """
        terminated = False
        if self._check_q_limits():
            print(40 * '-' + " JOINT LIMIT " + 40 * '-')
            terminated = True

        # Prematurely terminate if contacting the table with the arm
        if self._check_arm_contact():
            print(40 * '-' + " COLLIDED " + 40 * '-')
            terminated = True

        return terminated

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        return

    def _post_action(self, action):
        """
        If something to do
        """
        reward, done, info = super()._post_action(action)

        info['add_vals'] += ['hinge_angle', 'hinge_diff', 'percent_viapoints_', 'touched_handle']
        info['hinge_angle'] = self.hinge_qpos
        info['hinge_diff'] = self.hinge_diff
        info['touched_handle'] = self.touched_handle
        info['percent_viapoints_'] = (np.abs(self.hinge_goal) - self.hinge_diff) / (np.abs(self.hinge_goal))

        done = done or self._check_terminated()

        if self.data_logging:
            self.file_logging['percent_viapoints_'][self.counter - 1] = (np.abs(self.hinge_goal) - self.hinge_diff) / (
                np.abs(self.hinge_goal))
            self.file_logging['hinge_angle'][self.counter - 1] = self.hinge_qpos
            self.file_logging['hinge_diff'][self.counter - 1] = self.hinge_diff
            self.file_logging['hinge_goal'][self.counter - 1] = self.hinge_goal

            done = done or self._check_terminated()

            self.file_logging['done'][self.counter - 1] = done

        return reward, done, info