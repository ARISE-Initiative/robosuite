import numpy as np
from collections import OrderedDict
from robosuite.utils import RandomizationError
from robosuite.environments.sawyer_robot_arm import SawyerRobotArmEnv
from robosuite.models import *
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.tasks import Task, UniformRandomSampler, WipeForceTableTask
from robosuite.models.arenas import WipeForceTableArena
from robosuite.models.objects import CylinderObject, PlateWithHoleObject, BoxObject
import multiprocessing
from robosuite.controllers.arm_controller import *

import robosuite.utils.transform_utils as T
import logging
logger = logging.getLogger(__name__)
import hjson
import os


class SawyerWipeForce(SawyerRobotArmEnv):

    def __init__(
        self,
        gripper_type="WipingGripper",
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
            the configuration files
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
                                         'scripts/config/Wipe_force_task_config.hjson')
        else:
            task_filepath = os.path.join(os.path.dirname(__file__), '..',
                                         task_config_file)

        try:
            with open(task_filepath) as f:
                task = hjson.load(f)
                # Load additional arguments from kwargs and override the prior config-file loaded ones
                for key, value in kwargs.items():
                    if key in task:
                        task[key] = value
        except FileNotFoundError:
            print("Env Config file '{}' not found. Please check filepath and try again.".format(task_filepath))

        self.randomize_initialization = task['randomize_initialization']

        # settings for the reward
        self.arm_collision_penalty = task['arm_collision_penalty']
        self.wipe_contact_reward = task['wipe_contact_reward']
        self.unit_wiped_reward = task['unit_wiped_reward']
        self.ee_accel_penalty = task['ee_accel_penalty']
        self.excess_force_penalty_mul = task['excess_force_penalty_mul']

        # settings for table top
        self.table_full_size = task['table_full_size']
        self.table_friction = task['table_friction']
        self.table_height_std = task['table_height_std']
        self.table_friction_std = task['table_friction_std']
        self.line_width = task['line_width']
        self.two_clusters = task['two_clusters']

        self.distance_multiplier = task['distance_multiplier']
        self.distance_th_multiplier = task['distance_th_multiplier']

        # num of squares to divide each dim of the table surface
        self.num_squares = task['num_squares']

        # force threshold (N) to overcome to change the color of the sensor
        self.touch_threshold = task['touch_threshold']

        # Whether to print results or not
        self.print_results = task['print_results']

        # whether to include and use ground-truth object states
        self.use_object_obs = use_object_obs

        # whether we're using the actual robot or a sim
        self.real_robot = task['real_robot']

        self.use_robot_obs = task['use_robot_obs']

        self.wiped_sensors = []

        self.collisions = 0
        self.f_excess = 0

        self.pressure_threshold = task['touch_threshold']
        self.pressure_threshold_max = task['pressure_threshold_max']
        self.shear_threshold = task['shear_threshold']

        self.prob_sensor = task['prob_sensor']

        self.table_rot_x = task['table_rot_x']
        self.table_rot_y = task['table_rot_y']

        self.with_pos_limits = task['with_pos_limits']

        self.reward_range = [-1e6, 1e6]
        self.metadata = []
        self.spec = "spec"

        self.q_inits = [[-0.271568, -0.229342, -0.25828, -2.69931, -0.103803, 2.39945, 0.314063],
                        [0.757221, -0.175985, -0.173126, -2.63342, 0.0725866, 2.44531, 1.30004],
                        [0.706691, 0.272314, -0.290495, -2.13187, 0.200039, 2.38476, 1.09143],
                        [0.237219, 0.316146, -0.649803, -2.1357, 0.295333, 2.30397, 0.14446],
                        [0.637202, -0.556079, -0.630931, -2.63205, -0.334799, 2.07146, 1.04501]]
        self.with_qinits = task['with_qinits']

        self.draw_line = task['draw_line']  # whether you want a line for sensors
        self.num_sensors = task['num_sensors']

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[0, 0.2], y_range=[0, 0.2],
                ensure_object_boundary_in_range=False,
                z_rotation=True)

        self.table_origin = [0.36 + self.table_full_size[0] / 2, 0, 0]

        super(SawyerWipeForce, self).__init__(
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
            self.file_logging.create_dataset('f_excess', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('colls', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('nwipedsensors', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('done', (self.data_count, 1), maxshape=(None, 1))

        if self.with_pos_limits:
            self.controller.position_limits = [
                [self.table_origin[0] - self.table_full_size[0] / 2, self.table_origin[1] - self.table_full_size[1] / 2,
                 -2],
                [self.table_origin[0] + self.table_full_size[0] / 2, self.table_origin[1] + self.table_full_size[1] / 2,
                 2]]
        # self.controller.orientation_limits = [[np.pi - np.pi/4, -np.pi/4, -np.pi],
        #     [-np.pi+ np.pi/4, np.pi/4, np.pi]]

    def _load_model(self):
        super()._load_model()
        # Sawyer qpos_init
        self.mujoco_robot._init_qpos = np.array(
            [-0.8731347152710154, -1.321322055491266, 0.7948190211959069, 2.060864441092745, -0.27352056874045805,
             0.7947925648666623, 0.3014004480926677])

        self.mujoco_robot.set_base_xpos([0, 0, 0])

        self.robot_contact_geoms = self.mujoco_robot.contact_geoms

        # Delta goes down
        delta_height = min(0, np.random.normal(0.0, self.table_height_std))

        table_full_size_sampled = (
        self.table_full_size[0], self.table_full_size[1], self.table_full_size[2] + delta_height)
        self.mujoco_arena = WipeForceTableArena(table_full_size=table_full_size_sampled,
                                                table_friction=self.table_friction,
                                                table_friction_std=self.table_friction_std,
                                                num_squares=self.num_squares if not self.real_robot else 0,
                                                prob_sensor=self.prob_sensor,
                                                rotation_x=np.random.normal(0, self.table_rot_x),
                                                rotation_y=np.random.normal(0, self.table_rot_y),
                                                draw_line=self.draw_line,
                                                num_sensors=self.num_sensors if not self.real_robot else 0,
                                                line_width=self.line_width,
                                                two_clusters=self.two_clusters

                                                )

        # The sawyer robot has a pedestal, we want to align it with the table

        self.mujoco_arena.set_origin(self.table_origin)

        self.mujoco_objects = OrderedDict()

        # task includes arena, robot, and objects of interest
        self.model = WipeForceTableTask(self.mujoco_arena,
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
        # Small randomization of the initial configuration
        if self.randomize_initialization:
            if self.with_qinits:
                qinit_now = self.q_inits[np.random.choice(len(self.q_inits))]
                self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(qinit_now + np.random.randn(7) * 0.02)
            else:
                self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(
                    self.mujoco_robot.init_qpos + np.random.randn(7) * 0.02)
        else:
            self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(self.mujoco_robot.init_qpos)
        self.timestep = 0
        self.wiped_sensors = []
        self.collisions = 0
        self.f_excess = 0

    def reward(self, action):

        reward = 0

        total_force_ee = np.linalg.norm(np.array(self.ee_force))

        # Neg Reward from collisions of the arm with the table
        if self._check_arm_contact():
            reward = self.arm_collision_penalty
            self.collisions += 1
        elif self._check_q_limits():
            reward = self.arm_collision_penalty
            self.collisions += 1
        else:
            # TODO: Use the sensed touch to shape reward
            # Only do computation if there are active sensors and they weren't active before
            sensors_active_ids = []

            # Current 3D location of the corners of the wiping tool in world frame
            corner1_id = self.sim.model.geom_name2id("wiping_corner1")
            corner1_pos = np.array(self.sim.data.geom_xpos[corner1_id])
            corner2_id = self.sim.model.geom_name2id("wiping_corner2")
            corner2_pos = np.array(self.sim.data.geom_xpos[corner2_id])
            corner3_id = self.sim.model.geom_name2id("wiping_corner3")
            corner3_pos = np.array(self.sim.data.geom_xpos[corner3_id])
            corner4_id = self.sim.model.geom_name2id("wiping_corner4")
            corner4_pos = np.array(self.sim.data.geom_xpos[corner4_id])

            touching_points = [np.array(corner1_pos), np.array(corner2_pos), np.array(corner3_pos),
                               np.array(corner4_pos)]

            # Unit vectors on my plane
            v1 = corner1_pos - corner2_pos
            v1 /= np.linalg.norm(v1)
            v2 = corner4_pos - corner2_pos
            v2 /= np.linalg.norm(v2)

            # Corners of the tool in the coordinate frame of the plane
            t1 = np.array([np.dot(corner1_pos - corner2_pos, v1), np.dot(corner1_pos - corner2_pos, v2)])
            t2 = np.array([np.dot(corner2_pos - corner2_pos, v1), np.dot(corner2_pos - corner2_pos, v2)])
            t3 = np.array([np.dot(corner3_pos - corner2_pos, v1), np.dot(corner3_pos - corner2_pos, v2)])
            t4 = np.array([np.dot(corner4_pos - corner2_pos, v1), np.dot(corner4_pos - corner2_pos, v2)])

            pp = [t1, t2, t4, t3]

            # Normal of the plane defined by v1 and v2
            n = np.cross(v1, v2)
            n /= np.linalg.norm(n)

            def area(p1, p2, p3):
                return abs(0.5 * ((p1[0] - p3[0]) * (p2[1] - p1[1]) - (p1[0] - p2[0]) * (p3[1] - p1[1])))

            def isPinRectangle(r, P, printing=False):
                """
                    r: A list of four points, each has a x- and a y- coordinate
                    P: A point
                """
                areaRectangle = area(r[0], r[1], r[2]) + area(r[1], r[2], r[3])

                ABP = area(r[0], r[1], P)
                BCP = area(r[1], r[2], P)
                CDP = area(r[2], r[3], P)
                DAP = area(r[3], r[0], P)

                inside = abs(areaRectangle - (ABP + BCP + CDP + DAP)) < 1e-6

                if printing:
                    print(areaRectangle)
                    print((ABP + BCP + CDP + DAP))

                return inside

            def isLeft(P0, P1, P2):
                return ((P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1]))

            def PointInRectangle(X, Y, Z, W, P):

                return (isLeft(X, Y, P) < 0 and isLeft(Y, Z, P) < 0 and isLeft(Z, W, P) < 0 and isLeft(W, X, P) < 0)

            if self.sim.data.ncon != 0:

                for sensor_name in self.model.arena.sensor_names:

                    # Current sensor 3D location in world frame
                    # sensor_pos = np.array(
                    #     self.sim.data.body_xpos[self.sim.model.site_bodyid[self.sim.model.site_name2id(self.model.arena.sensor_site_names[sensor_name])]])
                    sensor_pos = np.array(
                        self.sim.data.site_xpos[
                            self.sim.model.site_name2id(self.model.arena.sensor_site_names[sensor_name])])

                    # We use the second tool corner as point on the plane and define the vector connecting
                    # the sensor position to that point
                    v = sensor_pos - corner2_pos

                    # Shortest distance between the center of the sensor and the plane
                    dist = np.dot(v, n)

                    # Projection of the center of the sensor onto the plane
                    projected_point = np.array(sensor_pos) - dist * n

                    # Positive distances means the center of the sensor is over the plane
                    # The plane is aligned with the bottom of the wiper and pointing up, so the sensor would be over it
                    if dist > 0.0:
                        # Distance smaller than this threshold means we are close to the plane on the upper part
                        if dist < 0.02:
                            # Write touching points and projected point in coordinates of the plane
                            pp_2 = np.array(
                                [np.dot(projected_point - corner2_pos, v1), np.dot(projected_point - corner2_pos, v2)])
                            # if isPinRectangle(pp, pp_2):
                            if PointInRectangle(pp[0], pp[1], pp[2], pp[3], pp_2):
                                parts = sensor_name.split('_')
                                sensors_active_ids += [int(parts[1])]

            lall = np.where(np.isin(sensors_active_ids, self.wiped_sensors, invert=True))
            new_sensors_active_ids = np.array(sensors_active_ids)[lall]

            for new_sensor_active_id in new_sensors_active_ids:
                new_sensor_active_site_id = self.sim.model.site_name2id(
                    self.model.arena.sensor_site_names['contact_' + str(new_sensor_active_id) + '_sensor'])
                self.sim.model.site_rgba[new_sensor_active_site_id] = [0, 0, 0, 0]

                self.wiped_sensors += [new_sensor_active_id]
                reward += self.unit_wiped_reward

            mean_distance_to_things_to_wipe = 0
            num_non_wiped_sensors = 0

            for sensor_name in self.model.arena.sensor_names:
                parts = sensor_name.split('_')
                sensor_id = int(parts[1])
                if sensor_id not in self.wiped_sensors:
                    sensor_pos = np.array(
                        self.sim.data.site_xpos[
                            self.sim.model.site_name2id(self.model.arena.sensor_site_names[sensor_name])])
                    gripper_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
                    mean_distance_to_things_to_wipe += np.linalg.norm(gripper_position - sensor_pos)
                    num_non_wiped_sensors += 1

            mean_distance_to_things_to_wipe /= max(1, num_non_wiped_sensors)

            reward += self.distance_multiplier * (
                        1 - np.tanh(self.distance_th_multiplier * mean_distance_to_things_to_wipe))

            # Reward for keeping contact
            if self.sim.data.ncon != 0:
                reward += 0.001

            if total_force_ee > self.pressure_threshold_max:
                reward -= self.excess_force_penalty_mul * total_force_ee
                self.f_excess += 1
            elif total_force_ee > self.pressure_threshold and self.sim.data.ncon > 1:
                reward += self.wipe_contact_reward + 0.01 * total_force_ee
                if self.sim.data.ncon > 50:
                    reward += 10 * self.wipe_contact_reward

            # Final reward if all wiped
            if len(self.wiped_sensors) == len(self.model.arena.sensor_names):
                reward += 50 * (
                            self.wipe_contact_reward + 0.5)  # So that is better to finish that to stay touching the table for 100 steps
                # The 0.5 comes from continuous_distance_reward at 0. If something changes, this may change as well

        # Penalize large accelerations
        reward -= self.ee_accel_penalty * np.mean(abs(self.ee_acc))

        # Printing results
        if(self.print_results):
            string_to_print = 'Process {pid}, timestep {ts:>4}: reward: {rw:8.4f} wiped sensors: {ws:>3} collisions: {sc:>3} f-excess: {fe:>3}'.format(
                pid=id(multiprocessing.current_process()),
                ts=self.timestep,
                rw=reward,
                ws=len(self.wiped_sensors),
                sc=self.collisions,
                fe=self.f_excess)
            print(string_to_print)

        return reward

    def _get_observation(self):
        di = super()._get_observation()

        # object information in the observation
        if self.use_object_obs:
            gripper_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
            # position of objects to wipe
            acc = np.array([])
            for sensor_name in self.model.arena.sensor_names:
                parts = sensor_name.split('_')
                sensor_id = int(parts[1])
                sensor_pos = np.array(
                    self.sim.data.site_xpos[
                        self.sim.model.site_name2id(self.model.arena.sensor_site_names[sensor_name])])
                di['sensor' + str(sensor_id) + '_pos'] = sensor_pos
                acc = np.concatenate([acc, di['sensor' + str(sensor_id) + '_pos']])
                acc = np.concatenate([acc, [[0, 1][sensor_id in self.wiped_sensors]]])
                # proprioception
                if self.use_robot_obs:
                    di['gripper_to_sensor' + str(sensor_id)] = gripper_position - sensor_pos
                    acc = np.concatenate([acc, di['gripper_to_sensor' + str(sensor_id)]])
            di['object-state'] = acc

            # for sensor_name in self.model.arena.sensor_names:
            #     sensor_id = self.sim.model.sensor_name2id(sensor_name)
            #     sensor_site_id = self.sim.model.site_name2id(self.model.arena.sensor_site_names[self.sim.model.sensor_id2name(sensor_id)])
            #     sensor_body_id = self.sim.model.site_bodyid[sensor_site_id]
            #     sensor_position = np.array(self.sim.data.body_xpos[sensor_body_id])
            #     di['sensor' + str(sensor_id) + '_pos'] = sensor_position
            #     acc = np.concatenate([acc, di['sensor' + str(sensor_id) + '_pos'] ])
            #     acc = np.concatenate([acc, [[0,1][sensor_id in self.wiped_sensors]] ])
            #     # proprioception
            #     if self.use_robot_obs:
            #         gripper_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
            #         di['gripper_to_sensor'+str(sensor_id)] = gripper_position - sensor_position
            #         acc = np.concatenate([acc, di['gripper_to_sensor' + str(sensor_id)] ])
            # di['object-state'] = acc

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

        # Prematurely terminate if contacting the table with the arm
        if self._check_arm_contact():
            print(40 * '-' + " COLLIDED " + 40 * '-')
            terminated = True

        # Prematurely terminate if finished
        if len(self.wiped_sensors) == len(self.model.arena.sensor_names):
            print(40 * '+' + " FINISHED WIPING " + 40 * '+')
            terminated = True

        # force_sensor_id = self.sim.model.sensor_name2id("force_ee")
        # force_ee = self.sensor_data[force_sensor_id*3: force_sensor_id*3+3]
        # if np.linalg.norm(np.array(force_ee)) > 3*self.pressure_threshold_max:
        #     print(35*'*' + " TOO MUCH FORCE " + str(np.linalg.norm(np.array(force_ee))) + 35*'*')
        #     terminated = True

        # Prematurely terminate if contacting the table with the arm
        if self._check_q_limits():
            print(40 * '-' + " JOINT LIMIT " + 40 * '-')
            terminated = True

        return terminated

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        # if self.gripper_visualization:
        #     # get distance to cube
        #     cube_site_id = self.sim.model.site_name2id('cube')
        #     dist = np.sum(np.square(self.sim.data.site_xpos[cube_site_id] - self.sim.data.get_site_xpos('grip_site')))

        #     # set RGBA for the EEF site here
        #     max_dist = 0.1
        #     scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
        #     rgba = np.zeros(4)
        #     rgba[0] = 1 - scaled
        #     rgba[1] = scaled
        #     rgba[3] = 0.5

        #     self.sim.model.site_rgba[self.eef_site_id] = rgba
        return

    def _post_action(self, action):
        """
        If something to do
        """
        reward, done, info = super()._post_action(action)

        info['add_vals'] += ['nwipedsensors', 'colls', 'percent_viapoints_', 'f_excess']
        info['nwipedsensors'] = len(self.wiped_sensors)
        info['colls'] = self.collisions
        info['percent_viapoints_'] = len(self.wiped_sensors) / len(self.model.arena.sensor_names)
        info['f_excess'] = self.f_excess

        # allow episode to finish early
        done = done or self._check_terminated()

        if self.data_logging:
            self.file_logging['f_excess'][self.counter - 1] = np.array([self.f_excess])
            self.file_logging['colls'][self.counter - 1] = np.array([self.collisions])
            self.file_logging['nwipedsensors'][self.counter - 1] = np.array([len(self.wiped_sensors)])
            self.file_logging['done'][self.counter - 1] = np.array([1 if done else 0])

            if done or self.counter >= self.data_count:
                logger.info("Logging file: ", self.logging_filename)
                self.file_logging.flush()
                self.file_logging.close()
                # exit(0)

        return reward, done, info