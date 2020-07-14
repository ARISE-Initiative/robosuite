from collections import OrderedDict
import numpy as np

from robosuite.environments.robot_env import RobotEnv
from robosuite.robots import SingleArm

from robosuite.models.arenas import WipeArena
from robosuite.models.tasks import TableTopTask, UniformRandomSampler
import multiprocessing


# Default Wipe environment configuration
DEFAULT_WIPE_CONFIG = {
    # settings for reward
    "arm_collision_penalty": -50,                   # penalty for each collision of the arm (except the wiping tool) with the table
    "wipe_contact_reward": 0.001,                    # reward for contacting something with the wiping tool
    "unit_wiped_reward": 20,                        # reward per peg wiped
    "ee_accel_penalty": 0,                          # penalty for large end-effector accelerations 
    "excess_force_penalty_mul": 0.01,               # penalty for each step that the force is over the safety threshold
    "distance_multiplier": 0.0,                     # multiplier for the dense reward inversely proportional to the mean location of the pegs to wipe
    "distance_th_multiplier": 0,                    # multiplier in the tanh function for the aforementioned reward

    # settings for table top
    "table_full_size": [0.5, 0.5, 0.05],            # Size of the cover of the table
    "table_friction": [0.00001, 0.005, 0.0001],     # Friction parameters for the table
    "table_friction_std": 0,                        # Standard deviation to sample different friction parameters for the table each episode
    "table_height": 0.0,                            # Additional height of the table over the default location
    "table_height_std": 0.0,                        # Standard deviation to sample different heigths of the table each episode
    "table_rot_x": 0.0,                             # Rotation of the table surface around the x axis
    "table_rot_y": 0.0,                             # Rotation of the table surface around the y axis
    "line_width": 0.02,                             # Width of the line to wipe (diameter of the pegs)
    "two_clusters": False,                          # if the dirt to wipe is one continuous line or two
    "num_squares": [4, 4],                          # num of squares to divide each dim of the table surface

    # settings for thresholds
    "touch_threshold": 5,                           # force threshold (N) to overcome to change the color of the sensor (wipe the peg)
    "pressure_threshold_max": 30,                   # maximum force allowed (N)
    "shear_threshold": 5,                           # shear force required to overcome the change of color of the sensor (wipe the peg) - NOT USED

    # misc settings
    "print_results": False,                         # Whether to print results or not'
    "use_robot_obs": True,                          # if we use robot observations (proprioception) as input to the policy
    "real_robot": False,                            # whether we're using the actual robot or a sim
    "prob_sensor": 1.0,
    "draw_line": True,                              # whether you want a line for sensors
    "num_sensors": 10,
}


class Wipe(RobotEnv):
    """
    This class corresponds to the Wiping task for a single robot arm
    """

    def __init__(
        self,
        robots,
        controller_configs=None,
        gripper_types="WipingGripper",
        gripper_visualizations=False,
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
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
        task_config=None,
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
                gripper models from gripper factory.
                For this environment, setting a value other than the default ("WipingGripper") will raise an
                    AssertionError, as this environment is not meant to be used with any other alternative gripper.

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

             use_camera_obs (bool): if True, every observation includes rendered image(s)

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_scale (None or float): Scales the normalized reward function by the amount specified.
                If None, environment reward remains unnormalized

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

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

            task_config (None or dict): Specifies the parameters relevant to this task. For a full list of expected
                parameters, see the default configuration dict at the top of this file.
                If None is specified, the default configuration will be used.

        """
        # First, verify that only one robot is being inputted
        self._check_robot_configuration(robots)

        # Assert that the gripper type is None
        assert gripper_types == "WipingGripper",\
            "Tried to specify gripper other than WipingGripper in Wipe environment!"

        # Get config
        self.task_config = task_config if task_config is not None else DEFAULT_WIPE_CONFIG

        # Set task-specific parameters

        # settings for the reward
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.arm_collision_penalty = self.task_config['arm_collision_penalty']
        self.wipe_contact_reward = self.task_config['wipe_contact_reward']
        self.unit_wiped_reward = self.task_config['unit_wiped_reward']
        self.ee_accel_penalty = self.task_config['ee_accel_penalty']
        self.excess_force_penalty_mul = self.task_config['excess_force_penalty_mul']
        self.distance_multiplier = self.task_config['distance_multiplier']
        self.distance_th_multiplier = self.task_config['distance_th_multiplier']

        # settings for table top
        self.table_full_size = self.task_config['table_full_size']
        self.table_friction = self.task_config['table_friction']
        self.table_friction_std = self.task_config['table_friction_std']
        self.table_height = self.task_config['table_height']
        self.table_height_std = self.task_config['table_height_std']
        self.table_rot_x = self.task_config['table_rot_x']
        self.table_rot_y = self.task_config['table_rot_y']
        self.line_width = self.task_config['line_width']
        self.two_clusters = self.task_config['two_clusters']
        self.num_squares = self.task_config['num_squares']

        # settings for thresholds
        self.touch_threshold = self.task_config['touch_threshold']
        self.pressure_threshold = self.task_config['touch_threshold']
        self.pressure_threshold_max = self.task_config['pressure_threshold_max']
        self.shear_threshold = self.task_config['shear_threshold']

        # misc settings
        self.print_results = self.task_config['print_results']
        self.use_robot_obs = self.task_config['use_robot_obs']
        self.real_robot = self.task_config['real_robot']
        self.prob_sensor = self.task_config['prob_sensor']
        self.draw_line = self.task_config['draw_line']
        self.num_sensors = self.task_config['num_sensors']

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # set other wipe-specific attributes
        self.wiped_sensors = []
        self.collisions = 0
        self.f_excess = 0
        self.metadata = []
        self.spec = "spec"

        # whether to include and use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[0, 0.2],
                y_range=[0, 0.2],
                ensure_object_boundary_in_range=False,
                rotation=None)

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

    def reward(self, action):
        # The sparse reward is given after wiping ALL the dirt elements
        # If the reward is dense, there are additional elements (some are deactivated by default)
        # - Penalty due to collisions of the arm with the table
        # - Penalty due to reaching arm's joint limits
        # - Reward per "unit" wiped
        # - Reward for getting closer to the centroid of the "units" to wipe
        # - Reward for keeping the wiping tool in contact
        # - Penalty for large forces with the end-effector
        # - Penalty for large end-effector accelerations
        
        reward = 0

        total_force_ee = np.linalg.norm(np.array(self.robots[0].recent_ee_forcetorques.current[:3]))

        # Neg Reward from collisions of the arm with the table
        if self._check_arm_contact()[0]:
            if self.reward_shaping: reward = self.arm_collision_penalty
            self.collisions += 1
        elif self._check_q_limits()[0]:
            if self.reward_shaping: reward = self.arm_collision_penalty
            self.collisions += 1
        else:
            # If the arm is not colliding or in joint limits, we check if we are wiping (we don't want to reward wiping if there are unsafe situations)
            sensors_active_ids = []

            # Current 3D location of the corners of the wiping tool in world frame
            c_geoms = self.robots[0].gripper.important_geoms["corners"]
            corner1_id = self.sim.model.geom_name2id(c_geoms[0])
            corner1_pos = np.array(self.sim.data.geom_xpos[corner1_id])
            corner2_id = self.sim.model.geom_name2id(c_geoms[1])
            corner2_pos = np.array(self.sim.data.geom_xpos[corner2_id])
            corner3_id = self.sim.model.geom_name2id(c_geoms[2])
            corner3_pos = np.array(self.sim.data.geom_xpos[corner3_id])
            corner4_id = self.sim.model.geom_name2id(c_geoms[3])
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

            # Only go into this computation if there are contact points
            if self.sim.data.ncon != 0:

                # Check each sensor that is still active
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

            # Obtain the list of currently active (wiped) sensors that where not wiped before
            # These are the sensors we are wiping at this step
            lall = np.where(np.isin(sensors_active_ids, self.wiped_sensors, invert=True))
            new_sensors_active_ids = np.array(sensors_active_ids)[lall]

            # For all the new sensors we are wiping at this step, make them transparent (alpha=0), add them to the list
            # of wiped sensors, and add reward (if dense reward is ON)
            for new_sensor_active_id in new_sensors_active_ids:
                new_sensor_active_site_id = self.sim.model.site_name2id(
                    self.model.arena.sensor_site_names['contact_' + str(new_sensor_active_id) + '_sensor'])
                self.sim.model.site_rgba[new_sensor_active_site_id] = [0, 0, 0, 0]

                self.wiped_sensors += [new_sensor_active_id]
                if self.reward_shaping: reward += self.unit_wiped_reward

            # Additional dense reward: provides dense reward for getting closer to the centroid of the dirt to wipe
            mean_distance_to_things_to_wipe = 0
            num_non_wiped_sensors = 0
            for sensor_name in self.model.arena.sensor_names:
                parts = sensor_name.split('_')
                sensor_id = int(parts[1])
                if sensor_id not in self.wiped_sensors:
                    sensor_pos = np.array(
                        self.sim.data.site_xpos[
                            self.sim.model.site_name2id(self.model.arena.sensor_site_names[sensor_name])])
                    gripper_position = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
                    mean_distance_to_things_to_wipe += np.linalg.norm(gripper_position - sensor_pos)
                    num_non_wiped_sensors += 1
            mean_distance_to_things_to_wipe /= max(1, num_non_wiped_sensors)
            if self.reward_shaping: reward += self.distance_multiplier * (
                        1 - np.tanh(self.distance_th_multiplier * mean_distance_to_things_to_wipe))

            # Reward for keeping contact
            if self.sim.data.ncon != 0:
                if self.reward_shaping: reward += self.wipe_contact_reward

            # Penalty for excessive force with the end-effector
            if total_force_ee > self.pressure_threshold_max:
                if self.reward_shaping: reward -= self.excess_force_penalty_mul * total_force_ee
                self.f_excess += 1

            # Final reward if all wiped
            if len(self.wiped_sensors) == len(self.model.arena.sensor_names):
                reward += 50 * (
                            self.wipe_contact_reward + 0.5)  # So that is better to finish that to stay touching the table for 100 steps
                # The 0.5 comes from continuous_distance_reward at 0. If something changes, this may change as well

        # Penalize large accelerations
        if self.reward_shaping: reward -= self.ee_accel_penalty * np.mean(abs(self.robots[0].recent_ee_acc.current))

        # Printing results
        if self.print_results:
            string_to_print = 'Process {pid}, timestep {ts:>4}: reward: {rw:8.4f} wiped sensors: {ws:>3} collisions: {sc:>3} f-excess: {fe:>3}'.format(
                pid=id(multiprocessing.current_process()),
                ts=self.timestep,
                rw=reward,
                ws=len(self.wiped_sensors),
                sc=self.collisions,
                fe=self.f_excess)
            print(string_to_print)

        # This assumes the maximum reward per step is to wipe ONE unit. Some steps the agent will wipe more than one (normalized reward > 1)
        if self.reward_scale:
            reward *= self.reward_scale / (self.unit_wiped_reward + self.wipe_contact_reward)
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], SingleArm), \
            "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Get robot's contact geoms
        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms

        # Delta goes down
        delta_height = min(0, np.random.normal(self.table_height, self.table_height_std))

        table_full_size_sampled = (
        self.table_full_size[0], self.table_full_size[1], self.table_full_size[2] + delta_height)
        self.mujoco_arena = WipeArena(table_full_size=table_full_size_sampled,
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
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        self.mujoco_objects = OrderedDict()

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(self.mujoco_arena,
                                   [robot.robot_model for robot in self.robots],
                                   self.mujoco_objects,
                                   initializer=self.placement_initializer)
        self.model.place_objects()

    def _get_reference(self):
        super()._get_reference()

    def _reset_internal(self):
        super()._reset_internal()

        # inherited class should reset positions of objects (only if we're not using a deterministic reset)
        if not self.deterministic_reset:
            self.model.place_objects()
            self.mujoco_arena.reset_arena(self.sim)

        # Reset all internal vars for this wipe task
        self.timestep = 0
        self.wiped_sensors = []
        self.collisions = 0
        self.f_excess = 0

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

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

        # object information in the observation
        if self.use_object_obs:
            gripper_site_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
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
                    di['gripper_to_sensor' + str(sensor_id)] = gripper_site_pos - sensor_pos
                    acc = np.concatenate([acc, di['gripper_to_sensor' + str(sensor_id)]])
            di['object-state'] = acc

        return di

    def _check_terminated(self):
        """
        Returns True if task is successfully completed
        """

        terminated = False

        # Prematurely terminate if contacting the table with the arm
        if self._check_arm_contact()[0]:
            if self.print_results: print(40 * '-' + " COLLIDED " + 40 * '-')
            terminated = True

        # Prematurely terminate if finished
        if len(self.wiped_sensors) == len(self.model.arena.sensor_names):
            if self.print_results: print(40 * '+' + " FINISHED WIPING " + 40 * '+')
            terminated = True

        # Prematurely terminate if contacting the table with the arm
        if self._check_q_limits()[0]:
            if self.print_results: print(40 * '-' + " JOINT LIMIT " + 40 * '-')
            terminated = True

        return terminated

    def _post_action(self, action):
        """
        If something to do
        """
        reward, done, info = super()._post_action(action)

        info['add_vals'] = ['nwipedsensors', 'colls', 'percent_viapoints_', 'f_excess']
        info['nwipedsensors'] = len(self.wiped_sensors)
        info['colls'] = self.collisions
        info['percent_viapoints_'] = len(self.wiped_sensors) / len(self.model.arena.sensor_names)
        info['f_excess'] = self.f_excess

        # allow episode to finish early
        done = done or self._check_terminated()

        return reward, done, info

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable
        """
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"
