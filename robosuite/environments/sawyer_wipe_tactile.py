import numpy as np
from collections import OrderedDict
from robosuite.utils import RandomizationError
from robosuite.environments.sawyer_robot_arm import SawyerRobotArmEnv
from robosuite.models import *
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.tasks import Task, UniformRandomSampler, TactileTableTask
from robosuite.models.arenas import TactileTableArena
from robosuite.models.objects import CylinderObject, PlateWithHoleObject, BoxObject
import multiprocessing
from robosuite.controllers.arm_controller import *

import logging
logger = logging.getLogger(__name__)
import hjson


class SawyerWipeTactile(SawyerRobotArmEnv):

    def __init__(
        self,
        gripper_type="WipingGripper",
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
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
        controller='position_orientation',
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
            controller_filepath = 'robosuite/scripts/config/controller_config.hjson'
        else:
            controller_filepath = controller_config_file

        if use_default_task_config == True:
            task_filepath = 'robosuite/scripts/config/Wipe_base_task_config.hjson'
        else:
            task_filepath = task_config_file

        try:
            with open(task_filepath) as f:
                task = hjson.load(f)
        except FileNotFoundError:
            print("Env Config file '{}' not found. Please check filepath and try again.".format(task_filepath))

        # settings for the reward
        self.arm_collision_penalty = task['arm_collision_penalty']
        self.wipe_contact_reward = task['wipe_contact_reward']
        self.unit_wiped_reward = task['unit_wiped_reward']

        # settings for table top
        self.table_full_size = task['table_full_size']
        self.table_friction = task['table_friction']

        # num of squares to divide each dim of the table surface
        self.num_squares = task['num_squares']

        # force threshold (N) to overcome to change the color of the sensor
        self.touch_threshold = task['touch_threshold']

        # whether to include and use ground-truth object states
        self.use_object_obs = use_object_obs

        # whether to include and use ground-truth proprioception in the observation
        self.observe_robot_state = True

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # reward configuration
        self.reward_shaping = reward_shaping

        # Whether to print results or not
        self.print_results = task['print_results']

        self.use_proprio_obs = task['use_proprio_obs']

        self.wiped_sensors = []

        self.collisions = 0

        # object placement initializer\
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[0, 0.2], y_range=[0, 0.2],
                ensure_object_boundary_in_range=False,
                z_rotation=True)

        super(SawyerWipeTactile, self).__init__(
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

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        self.robot_contact_geoms = self.mujoco_robot.contact_geoms

        self.mujoco_arena = TactileTableArena(table_full_size=self.table_full_size,
                                              table_friction=self.table_friction,
                                              num_squares=self.num_squares
                                              )

        # The robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.26 + self.table_full_size[0] / 2, 0, 0])

        self.mujoco_objects = OrderedDict()

        # task includes arena, robot, and objects of interest
        self.model = TactileTableTask(self.mujoco_arena,
                                      self.mujoco_robot,
                                      self.mujoco_objects,
                                      initializer=self.placement_initializer)
        self.model.place_objects()

        # print(self.model.get_xml())
        # exit()

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
        self.collisions = 0

    def reward(self, action):
        reward = 0

        self.table_id = self.sim.model.geom_name2id('table_visual')
        table_height = self.table_full_size[2]
        table_location = self.mujoco_arena.table_top_abs
        table_x_border_plus = table_location[0] + self.table_full_size[0] * 0.5
        table_x_border_minus = table_location[0] - self.table_full_size[0] * 0.5
        table_y_border_plus = table_location[1] + self.table_full_size[1] * 0.5
        table_y_border_minus = table_location[1] - self.table_full_size[1] * 0.5

        # Neg Reward from collisions of the arm with the table
        if self._check_arm_contact():
            reward = self.arm_collision_penalty
            self.collisions += 1
        else:
            force_sensor_id = self.sim.model.sensor_name2id("force_ee")
            force_ee = self.sim.data.sensordata[force_sensor_id * 3: force_sensor_id * 3 + 3]

            # TODO: Use the sensed touch to shape reward
            # Only do computation if there are active sensors and they weren't active before
            sensors_active_ids = np.argwhere(self.sim.data.sensordata > self.touch_threshold).flatten()
            new_sensors_active_ids = sensors_active_ids[
                np.where(np.isin(sensors_active_ids, self.wiped_sensors, invert=True))]

            if np.any(new_sensors_active_ids):
                for new_sensor_active_id in new_sensors_active_ids:
                    # TODO: Careful! The sensor IDs do not correspond to the sites IDs
                    # Site id 0 -> Table top site
                    # We need to add +1 to all the sensor IDs to match the site IDs
                    new_sensor_active_site_id = new_sensor_active_id + 1
                    self.sim.model.site_rgba[new_sensor_active_site_id] = [1, 1, 1, 1]
                    self.wiped_sensors += [new_sensor_active_id]
                    reward += self.unit_wiped_reward

            # Mean distance to all sensors
            # mean_d = 0
            num_sensors_inactive = 0
            for sensor_id in range(len(self.sim.data.sensordata)):
                if sensor_id not in self.wiped_sensors:
                    # TODO: Careful! The sensor IDs do not correspond to the sites IDs
                    # Site id 0 -> Table top site
                    # We need to add +1 to all the sensor IDs to match the site IDs
                    sensor_site_id = sensor_id + 1
                    sensor_position = np.array(self.sim.data.site_xpos[sensor_site_id])
                    gripper_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
                    sensor_i_to_gripper_d = np.linalg.norm(gripper_position - sensor_position)
                    continuous_distance_reward = 0.01 * (1 - np.tanh(5 * sensor_i_to_gripper_d))
                    reward += continuous_distance_reward

                    # Reward for keeping contact
            # if self.sim.data.ncon != 0 :
            if np.linalg.norm(np.array(force_ee)) > 1:
                reward += self.wipe_contact_reward

        if(self.print_results):
            print('Process %i, timestep %i: reward: %5.4f wiped sensors: %i collisions: %i' % (
                id(multiprocessing.current_process()), self.timestep, reward, len(self.wiped_sensors),
                self.collisions))  # 'reward ', reward)
        return reward

    def _get_observation(self):
        di = super()._get_observation()

        # object information in the observation
        if self.use_object_obs:
            # position of objects to wipe
            acc = np.array([])
            for i in range(len(self.sim.data.sensordata)):
                sensor_site_id = i + 1
                sensor_position = np.array(self.sim.data.site_xpos[sensor_site_id])
                di['sensor' + str(i) + '_pos'] = sensor_position
                acc = np.concatenate([acc, di['sensor' + str(i) + '_pos']])
                acc = np.concatenate([acc, [[0, 1][i in self.wiped_sensors]]])
                # proprioception
                if self.use_proprio_obs:
                    gripper_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
                    di['gripper_to_sensor' + str(i)] = gripper_position - sensor_position
                    acc = np.concatenate([acc, di['gripper_to_sensor' + str(i)]])
            di['object-state'] = acc

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

        # If all the pegs are wiped off
        # cube is higher than the table top above a margin
        terminated = True

        # If any other part of the robot (not the wiping) touches the table
        # if len([c for c in self.find_contacts(['table_collision'],self.robot_contact_geoms)]) > 0:
        #    terminated = True

        if terminated:
            print(60 * "*")
            print("TERMINATED")
            print(60 * "*")

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

        info['add_vals'] = ['nwipedsensors', 'colls']
        info['nwipedsensors'] = len(self.wiped_sensors)
        info['colls'] = self.collisions

        return reward, done, info