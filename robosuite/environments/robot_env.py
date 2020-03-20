import numpy as np

from robosuite.environments.base import MujocoEnv

from robosuite.agents.single_arm import SingleArm
from robosuite.agents.bimanual import Bimanual
from robosuite.models.robots.robot import check_bimanual

from robosuite.controllers.controller_factory import reset_controllers


class RobotEnv(MujocoEnv):
    """
    Initializes a robot environment in Mujoco.
    """

    def __init__(
        self,
        robots,
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        use_camera_obs=True,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderers=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        """
        Args:
            robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
                (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)

            controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
                custom controller. Else, uses the default controller for this specific task. Should either be single
                dict if same controller is to be used for all robots or else it should be a list of the same length as
                "robots" param

            gripper_types (str or list of str): type of gripper, used to instantiate
                gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
                with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
                overrides the default gripper. Should either be single str if same gripper type is to be used for all
                robots or else it should be a list of the same length as "robots" param

            gripper_visualizations (bool or list of bool): True if using gripper visualization.
                Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
                robots or else it should be a list of the same length as "robots" param

            use_camera_obs (bool or list of bool): if True, every observation for a specific robot includes a rendered
            image. Should either be single bool if camera obs value is to be used for all
                robots or else it should be a list of the same length as "robots" param

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderers (bool or list of bool): True if using off-screen rendering. Should either be single
                bool if same offscreen renderering setting is to be used for all cameras or else it should be a list of
                the same length as "robots" param

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_names (str or list of str): name of camera to be rendered. Should either be single str if
                same name is to be used for all cameras' rendering or else it should be a list of the same length as
                "robots" param. Note: Each name must be set if the corresponding @use_camera_obs value is True.

            camera_heights (int or list of int): height of camera frame. Should either be single int if
                same height is to be used for all cameras' frames or else it should be a list of the same length as
                "robots" param.

            camera_widths (int or list of int): width of camera frame. Should either be single int if
                same width is to be used for all cameras' frames or else it should be a list of the same length as
                "robots" param.

            camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
                bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
                "robots" param.

        """
        # Robot
        robots = robots if type(robots) is list else [robots]
        self.num_robots = len(robots)
        self.robot_names = robots
        self.robots = self._input2list(None)
        self.action_dim = None

        # Controller
        controller_configs = self._input2list(controller_configs)

        # Gripper
        gripper_types = self._input2list(gripper_types)
        gripper_visualizations = self._input2list(gripper_visualizations)

        # Observations -- Ground truth = object_obs, Image data = camera_obs
        self.use_camera_obs = self._input2list(use_camera_obs)

        # Camera / Rendering Settings
        self.has_offscreen_renderers = self._input2list(has_offscreen_renderers)
        self.camera_names = self._input2list(camera_names)
        self.camera_heights = self._input2list(camera_heights)
        self.camera_widths = self._input2list(camera_widths)
        self.camera_depths = self._input2list(camera_depths)

        # sanity checks for cameras
        for idx, (cam_name, use_cam_obs, has_offscreen_renderer) in \
            enumerate(zip(self.camera_names, self.use_camera_obs, self.has_offscreen_renderers)):
            if use_cam_obs and not has_offscreen_renderer:
                raise ValueError("Camera #{} observations require an offscreen renderer.".format(idx))
            if use_cam_obs and cam_name is None:
                raise ValueError("Must specify camera #{} name when using camera obs".format(idx))

        # Robot configurations
        self.robot_configs = [
            {
                "controller_config": controller_configs[idx],
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
            has_offscreen_renderer=any(self.has_offscreen_renderers),
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
        )

    @property
    def action_spec(self):
        """
        Action space (low, high) for this environment
        """
        low, high = [], []
        for robot in self.robots:
            lo, hi = robot.action_limits
            low, high = np.concatenate([low, lo]), np.concatenate([high, hi])
        return low, high

    def move_indicator(self, pos):
        """
        Sets 3d position of indicator object to @pos.
        """
        if self.use_indicator_object:
            index = self._ref_indicator_pos_low
            self.sim.data.qpos[index : index + 3] = pos

    def _input2list(self, inp):
        """
        Helper function that converts an input that is either a single value or a list into a list
        @inp (str or list): Input value to be converted to list
        """
        # convert to list if necessary
        return inp if type(inp) is list else [inp for _ in range(self.num_robots)]

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Loop through robots and load each model
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
        super()._reset_internal()

        self.action_dim = 0

        # Reset controllers
        reset_controllers()

        # Reset robot and update action space dimension along the way
        for robot in self.robots:
            robot.reset()
            self.action_dim += robot.action_dim

    def _pre_action(self, action, policy_step=False):
        """
        Overrides the superclass method to control the robot(s) within this enviornment using their respective
        controllers using the passed actions and gripper control.

        Args:
            action (numpy array): The control to apply to the robot(s). Note that this should be a flat 1D array that
                encompasses all actions to be distributed to each robot if there are multiple. For each section of the
                action space assigned to a single robot, the first @self.robots[i].controller.control_dim dimensions
                should be the desired controller actions and if the robot has a gripper, the next
                @self.robots[i].gripper.dof dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken
        """
        # Verify that the action is the correct dimension
        assert len(action) == self.action_dim, \
            "environment got invalid action dimension -- expected {}, got {}".format(
                self.action_dim, len(action))

        # Update robot joints based on controller actions
        cutoff = 0
        for robot in self.robots:
            robot.control(action[cutoff:cutoff+robot.action_dim], policy_step=policy_step)
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
        (Optional) anything after actions.
        """
        ret = super()._post_action(action)
        self._visualization()
        return ret

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()

        # Loop through all robots and their respective cameras and update the observations
        for idx, (robot, use_cam_obs, cam_name, cam_w, cam_h, cam_d) in \
                enumerate(zip(self.robots, self.use_camera_obs,
                              self.camera_names, self.camera_widths, self.camera_heights, self.camera_depths)):

            # Add robot observations to the dict
            di = robot.get_observations(di)

            # camera observations
            if use_cam_obs:
                camera_obs = self.sim.render(
                    # TODO: May need to update cameras to correspond to each robot-specific view
                    camera_name=cam_name,
                    width=cam_w,
                    height=cam_h,
                    depth=cam_d,
                )
                if cam_d:
                    di["image"], di["depth"] = camera_obs
                else:
                    di["image"] = camera_obs

        return di

    def _check_contact(self):
        """
        Returns True for any gripper is in contact with an object.
        """
        collisions = [False] * self.num_robots
        for idx, robot in enumerate(self.robots):
            for contact in self.sim.data.contact[: self.sim.data.ncon]:
                # Single arm case
                if robot.arm_type == "single":
                    if (
                        self.sim.model.geom_id2name(contact.geom1)
                        in robot.gripper.contact_geoms()
                        or self.sim.model.geom_id2name(contact.geom2)
                        in robot.gripper.contact_geoms()
                    ):
                        collisions[idx] = True
                        break
                # Bimanual case
                else:
                    for arm in robot.arms:
                        if (
                                self.sim.model.geom_id2name(contact.geom1)
                                in robot.gripper[arm].contact_geoms()
                                or self.sim.model.geom_id2name(contact.geom2)
                                in robot.gripper[arm].contact_geoms()
                        ):
                            collisions[idx] = True
                            break
        return collisions

    def _visualization(self):
        """
        Do any needed visualization here
        """
        # Loop over robot grippers to visualize them indepedently
        for robot in self.robots:
            robot.gripper_visualization

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure inputted robots and the corresponding requested task/configuration combo is legal.
        Should be implemented in every specific task module

        Args:
            robots (str or list of str): Inputted requested robots at the task-level environment
        """
        raise NotImplementedError
