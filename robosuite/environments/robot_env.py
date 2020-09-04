import numpy as np

from robosuite.environments.base import MujocoEnv

from robosuite.robots.single_arm import SingleArm
from robosuite.robots.bimanual import Bimanual
from robosuite.models.robots import check_bimanual

from robosuite.controllers import reset_controllers


class RobotEnv(MujocoEnv):
    """
    Initializes a robot environment in Mujoco.

    Args:
        robots: Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (None or str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        gripper_visualizations (bool or list of bool): True if using gripper visualization.
            Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

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

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
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

    Raises:
        ValueError: [Camera obs require offscreen renderer]
        ValueError: [Camera name must be specified to use camera obs]
    """

    def __init__(
        self,
        robots,
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        initialization_noise=None,
        use_camera_obs=True,
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
    ):
        # Robot
        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]
        self.num_robots = len(robots)
        self.robot_names = robots
        self.robots = self._input2list(None, self.num_robots)
        self._action_dim = None

        # Controller
        controller_configs = self._input2list(controller_configs, self.num_robots)

        # Initialization Noise
        initialization_noise = self._input2list(initialization_noise, self.num_robots)

        # Gripper
        gripper_types = self._input2list(gripper_types, self.num_robots)
        gripper_visualizations = self._input2list(gripper_visualizations, self.num_robots)

        # Observations -- Ground truth = object_obs, Image data = camera_obs
        self.use_camera_obs = use_camera_obs

        # Camera / Rendering Settings
        self.has_offscreen_renderer = has_offscreen_renderer
        self.camera_names = list(camera_names) if type(camera_names) is list or \
            type(camera_names) is tuple else [camera_names]
        self.num_cameras = len(self.camera_names)

        self.camera_heights = self._input2list(camera_heights, self.num_cameras)
        self.camera_widths = self._input2list(camera_widths, self.num_cameras)
        self.camera_depths = self._input2list(camera_depths, self.num_cameras)

        # sanity checks for camera rendering
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Error: Camera observations require an offscreen renderer!")
        if self.use_camera_obs and self.camera_names is None:
            raise ValueError("Must specify at least one camera name when using camera obs")

        # Robot configurations
        self.robot_configs = [
            {
                "controller_config": controller_configs[idx],
                "initialization_noise": initialization_noise[idx],
                "gripper_type": gripper_types[idx],
                "gripper_visualization": gripper_visualizations[idx],
                "control_freq": control_freq
            }
            for idx in range(self.num_robots)
        ]

        # whether to use indicator object or not
        self.use_indicator_object = use_indicator_object

        # Run superclass init
        super().__init__(
            has_renderer=has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
        )

    @property
    def action_spec(self):
        """
        Action space (low, high) for this environment

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        low, high = [], []
        for robot in self.robots:
            lo, hi = robot.action_limits
            low, high = np.concatenate([low, lo]), np.concatenate([high, hi])
        return low, high

    @property
    def action_dim(self):
        """
        Size of the action space

        Returns:
            int: Action space dimension
        """
        return self._action_dim

    def move_indicator(self, pos):
        """
        Sets 3d position of indicator object to @pos.

        Args:
            pos (3-tuple): (x,y,z) values to place the indicator within the env
        """
        if self.use_indicator_object:
            index = self._ref_indicator_pos_low
            self.sim.data.qpos[index : index + 3] = pos

    @staticmethod
    def _input2list(inp, length):
        """
        Helper function that converts an input that is either a single value or a list into a list

        Args:
            inp (None or str or list): Input value to be converted to list
            length (int): Length of list to broadcast input to

        Returns:
            list: input @inp converted into a list of length @length
        """
        # convert to list if necessary
        return list(inp) if type(inp) is list or type(inp) is tuple else [inp for _ in range(length)]

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Load robots
        self._load_robots()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Setup robot-specific references as well (note: requires resetting of sim for robot first)
        for robot in self.robots:
            robot.reset_sim(self.sim)
            robot.setup_references()

        # Indicator object references
        if self.use_indicator_object:
            ind_qpos = self.sim.model.get_joint_qpos_addr("pos_indicator")
            self._ref_indicator_pos_low, self._ref_indicator_pos_high = ind_qpos

            ind_qvel = self.sim.model.get_joint_qvel_addr("pos_indicator")
            self._ref_indicator_vel_low, self._ref_indicator_vel_high = ind_qvel

            self.indicator_id = self.sim.model.body_name2id("pos_indicator")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # Run superclass reset functionality
        super()._reset_internal()

        # Reset controllers
        reset_controllers()

        # Reset action dim
        self._action_dim = 0

        # Reset robot and update action space dimension along the way
        for robot in self.robots:
            robot.reset(deterministic=self.deterministic_reset)
            self._action_dim += robot.action_dim

        # Update cameras if appropriate
        if self.use_camera_obs:
            temp_names = []
            for cam_name in self.camera_names:
                if "all-" in cam_name:
                    # We need to add all robot-specific camera names that include the key after the tag "all-"
                    start_idx = len(temp_names) - 1
                    key = cam_name.replace("all-", "")
                    for robot in self.robots:
                        for robot_cam_name in robot.robot_model.cameras:
                            if key in robot_cam_name:
                                temp_names.append(robot_cam_name)
                    # We also need to broadcast the corresponding values from each camera dimensions as well
                    end_idx = len(temp_names) - 1
                    self.camera_widths = self.camera_widths[:start_idx] + \
                        [self.camera_widths[start_idx]] * (end_idx - start_idx) + \
                        self.camera_widths[(start_idx + 1):]
                    self.camera_heights = self.camera_heights[:start_idx] + \
                        [self.camera_heights[start_idx]] * (end_idx - start_idx) + \
                        self.camera_heights[(start_idx + 1):]
                    self.camera_depths = self.camera_depths[:start_idx] + \
                        [self.camera_depths[start_idx]] * (end_idx - start_idx) + \
                        self.camera_depths[(start_idx + 1):]
                else:
                    # We simply add this camera to the temp_names
                    temp_names.append(cam_name)
            # Lastly, replace camera names with the updated ones
            self.camera_names = temp_names

    def _pre_action(self, action, policy_step=False):
        """
        Overrides the superclass method to control the robot(s) within this enviornment using their respective
        controllers using the passed actions and gripper control.

        Args:
            action (np.array): The control to apply to the robot(s). Note that this should be a flat 1D array that
                encompasses all actions to be distributed to each robot if there are multiple. For each section of the
                action space assigned to a single robot, the first @self.robots[i].controller.control_dim dimensions
                should be the desired controller actions and if the robot has a gripper, the next
                @self.robots[i].gripper.dof dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """
        # Verify that the action is the correct dimension
        assert len(action) == self.action_dim, \
            "environment got invalid action dimension -- expected {}, got {}".format(
                self.action_dim, len(action))

        # Update robot joints based on controller actions
        cutoff = 0
        for idx, robot in enumerate(self.robots):
            robot_action = action[cutoff:cutoff+robot.action_dim]
            robot.control(robot_action, policy_step=policy_step)
            cutoff += robot.action_dim

        # Also update indicator object if necessary
        if self.use_indicator_object:
            # Apply gravity compensation to indicator object too
            self.sim.data.qfrc_applied[
                self._ref_indicator_vel_low: self._ref_indicator_vel_high
                ] = self.sim.data.qfrc_bias[
                    self._ref_indicator_vel_low: self._ref_indicator_vel_high]

    def _post_action(self, action):
        """
        Run any necessary visualization after running the action

        Args:
            action (np.array): Action being passed during this timestep

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method

        """
        ret = super()._post_action(action)
        self._visualization()
        return ret

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:

            `'robot-state'`: contains robot-centric information.

            `'image'`: requires @self.use_camera_obs to be True. Contains a rendered frame from the simulation.

            `'depth'`: requires @self.use_camera_obs and @self.camera_depth to be True.
            Contains a rendered depth map from the simulation

        Returns:
            OrderedDict: Observations from the environment
        """
        di = super()._get_observation()

        # Loop through robots and update the observations
        for robot in self.robots:
            di = robot.get_observations(di)

        # Loop through cameras and update the observations
        if self.use_camera_obs:
            for (cam_name, cam_w, cam_h, cam_d) in \
                    zip(self.camera_names, self.camera_widths, self.camera_heights, self.camera_depths):

                # Add camera observations to the dict
                camera_obs = self.sim.render(
                    camera_name=cam_name,
                    width=cam_w,
                    height=cam_h,
                    depth=cam_d,
                )
                if cam_d:
                    di[cam_name + "_image"], di[cam_name + "_depth"] = camera_obs
                else:
                    di[cam_name + "_image"] = camera_obs

        return di

    def _check_gripper_contact(self):
        """
        Checks whether each gripper is in contact with an object.

        Returns:
            list of bool: True if the specific gripper is in contact with an object
        """
        collisions = [False] * self.num_robots
        for idx, robot in enumerate(self.robots):
            for contact in self.sim.data.contact[: self.sim.data.ncon]:
                # Single arm case
                if robot.arm_type == "single":
                    if (
                        self.sim.model.geom_id2name(contact.geom1)
                        in robot.gripper.contact_geoms
                        or self.sim.model.geom_id2name(contact.geom2)
                        in robot.gripper.contact_geoms
                    ):
                        collisions[idx] = True
                        break
                # Bimanual case
                else:
                    for arm in robot.arms:
                        if (
                                self.sim.model.geom_id2name(contact.geom1)
                                in robot.gripper[arm].contact_geoms
                                or self.sim.model.geom_id2name(contact.geom2)
                                in robot.gripper[arm].contact_geoms
                        ):
                            collisions[idx] = True
                            break
        return collisions

    def _check_arm_contact(self):
        """
        Checks whether each robot arm is in contact with an object.

        Returns:
            list of bool: True if the specific gripper is in contact with an object
        """
        collisions = [False] * self.num_robots
        for idx, robot in enumerate(self.robots):
            for contact in self.sim.data.contact[: self.sim.data.ncon]:
                # Single arm case and Bimanual case are the same
                if (
                    self.sim.model.geom_id2name(contact.geom1)
                    in robot.robot_model.contact_geoms
                    or self.sim.model.geom_id2name(contact.geom2)
                    in robot.robot_model.contact_geoms
                ):
                    collisions[idx] = True
                    break
        return collisions

    def _check_q_limits(self):
        """
        Check if each robot arm is either very close or at the joint limits

        Returns:
            list of bool: True if the specific arm is near its joint limits
        """
        joint_limits = [False] * self.num_robots
        tolerance = 0.1
        for idx, robot in enumerate(self.robots):
            for (qidx, (q, q_limits)) in enumerate(
                    zip(
                        self.sim.data.qpos[robot._ref_joint_pos_indexes],
                        self.sim.model.jnt_range[robot._ref_joint_indexes]
                    )
            ):
                if not (q_limits[0] + tolerance < q < q_limits[1] - tolerance):
                    print("Joint limit reached in joint " + str(qidx))
                    joint_limits[idx] = True
        return joint_limits

    def _visualization(self):
        """
        Do any needed visualization here
        """
        # Loop over robot grippers to visualize them independently
        for robot in self.robots:
            robot.visualize_gripper()

    def _load_robots(self):
        """
        Instantiates robots and stores them within the self.robots attribute
        """
        # Loop through robots and instantiate Robot object for each
        for idx, (name, config) in enumerate(zip(self.robot_names, self.robot_configs)):
            # Create the robot instance
            if not check_bimanual(name):
                self.robots[idx] = SingleArm(
                    robot_type=name,
                    idn=idx,
                    **config
                )
            else:
                self.robots[idx] = Bimanual(
                    robot_type=name,
                    idn=idx,
                    **config
                )

            # Now, load the robot models
            self.robots[idx].load_model()

    def reward(self, action):
        """
        Runs superclass method by default
        """
        return super().reward(action)

    def _check_success(self):
        """
        Runs superclass method by default
        """
        return super()._check_success()

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure inputted robots and the corresponding requested task/configuration combo is legal.
        Should be implemented in every specific task module

        Args:
            robots (str or list of str): Inputted requested robots at the task-level environment
        """
        raise NotImplementedError
