from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class TwoArmTapeHandover(TwoArmEnv):
    """
    This class corresponds to a two robot arm environment with a yellow tape on the left side
    and a duct tape on the right side.

    Args:
        robots (str or list of str): Specification for specific robot(s)
            Note: Must be either 2 robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment if two robots inputted. Can be either:

            :'parallel': Sets up the two robots next to each other on the -x side of the table
            :'opposed': Sets up the two robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" "opposed" if two robots are used.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :'magnitude': The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to None or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :'type': Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        tables_boundary (3-tuple): x, y, and z dimensions of the table bounds. One table will be created at the center of
            this boundary

        table_friction (3-tuple): the three mujoco friction parameters for
            each table.

        yellow_tape_size (float): Size of the yellow tape

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (yellow tape) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

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

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        # Note: default gripper will use the franka gripper
        gripper_types="default",
        # gripper_types="Robotiq85Gripper",
        initialization_noise="default",
        tables_boundary=(0.8, 1.4, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        yellow_tape_size=0.04,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=["agentview", "all-eye_in_hand"],  # Use "all-camera_d405" if using Yam robots
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # settings for table top
        self.tables_boundary = tables_boundary
        self.table_full_size = np.array(tables_boundary)
        self.table_friction = table_friction
        self.table_offsets = np.zeros((1, 3))
        self.table_offsets[0, 2] = 0.8  # scale z offset
        self.yellow_tape_size = yellow_tape_size

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided when the yellow tape is placed on the duct tape

        Un-normalized max-wise components if using reward shaping:

            - Reaching: distance between yellow tape and gripper
            - Grasping: reward for grasping yellow tape
            - Placing: distance between yellow tape and duct tape

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        # Initialize reward
        reward = 0

        # Get positions
        yellow_tape_pos = np.array(self.sim.data.body_xpos[self.yellow_tape_body_id])
        duct_tape_pos = np.array(self.sim.data.body_xpos[self.duct_tape_body_id])

        # use a shaping reward if specified
        if self.reward_shaping:
            # Reaching reward
            gripper_site_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
            dist_to_yellow_tape = np.linalg.norm(gripper_site_pos - yellow_tape_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist_to_yellow_tape)
            reward += reaching_reward

            # Grasping reward
            yellow_tape_height = yellow_tape_pos[2]
            table_height = self.table_offsets[0, 2]
            if yellow_tape_height > table_height + 0.05:
                reward += 0.5

            # Placing reward
            dist_yellow_tape_to_duct_tape = np.linalg.norm(yellow_tape_pos - duct_tape_pos)
            placing_reward = 1 - np.tanh(10.0 * dist_yellow_tape_to_duct_tape)
            reward += placing_reward

            # Normalize
            reward = reward / 2.5

        # Else this is the sparse reward setting
        else:
            # Provide reward if yellow tape is in duct tape (within duct tape radius and at appropriate height)
            dist_xy = np.linalg.norm(yellow_tape_pos[:2] - duct_tape_pos[:2])
            if dist_xy < 0.08 and abs(yellow_tape_pos[2] - duct_tape_pos[2]) < 0.05:
                reward = 1.0

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation, offset in zip(self.robots, (np.pi / 2, -np.pi / 2), (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    xpos += np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.6, 0.6)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = MultiTableArena(
            table_offsets=self.table_offsets,
            table_rots=0,
            table_full_sizes=self.table_full_size,
            table_frictions=self.table_friction,
            has_legs=True,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        # mujoco_arena.set_camera(
        #     camera_name="agentview",
        #     pos=[1.2434677502317038, 4.965421871106301e-08, 2.091455182752329],
        #     quat=[0.6530981063842773, 0.2710406184196472, 0.27104079723358154, 0.6530979871749878],
        # )

        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[-1.2434677502317038, 4.965421871106301e-08, 2.091455182752329],
            quat=[0.65309799, 0.2710408, -0.27104062, -0.65309811],
        )


        import os
        yellow_tape_xml_path = os.path.join(os.path.dirname(__file__), "../../assets/yellow_tape/yellow_tape.xml")
        yellow_tape_decomp_xml_path = os.path.join(os.path.dirname(__file__), "../../assets/yellow_tape/yellow_tape_decomp/yellow_tape.xml")
        assert os.path.exists(yellow_tape_decomp_xml_path), "Yellow tape decomp XML path does not exist"
        self.yellow_tape = MujocoXMLObject(
            fname=yellow_tape_decomp_xml_path, # yellow_tape_xml_path,
            name="yellow_tape",
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # yellow_tape_cad_xml_path = os.path.join(os.path.dirname(__file__), "../../assets/yellow_tape_cad/yellow_tape_cad.xml")
        # self.yellow_tape = MujocoXMLObject(
        #     fname=yellow_tape_cad_xml_path,
        #     name="yellow_tape_cad",
        #     joints=[dict(type="free", damping="0.0005")],
        #     obj_type="all",
        #     duplicate_collision_geoms=True,
        # )
        duct_tape_xml_path = os.path.join(os.path.dirname(__file__), "../../assets/duct_tape/duct_tape.xml")
        self.duct_tape = MujocoXMLObject(
            fname=duct_tape_xml_path,
            name="duct_tape",
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

        # Create list of objects
        self.objects = [self.yellow_tape, self.duct_tape]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

        # Create placement initializer
        self._get_placement_initializer()

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # Calculate centers for left and right areas based on table boundary
        y_left_center = -self.tables_boundary[1] * 3 / 8
        y_right_center = self.tables_boundary[1] * 3 / 8

        # Yellow tape on left side of table
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="YellowTapeSampler",
                mujoco_objects=self.yellow_tape,
                x_range=[-0.15, 0.0],
                y_range=[y_right_center, y_right_center + 0.15],
                rotation=0,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offsets[0],
                z_offset=0.01,
                rng=self.rng,
            )
        )

        # Duct tape on right side of table
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="DuctTapeSampler",
                mujoco_objects=self.duct_tape,
                x_range=[0.0, 0.15],
                y_range=[y_left_center - 0.15, y_left_center],
                rotation=0,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offsets[0],
                z_offset=0.01,
                rng=self.rng,
            )
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional references for yellow tape and duct tape
        self.yellow_tape_body_id = self.sim.model.body_name2id(self.yellow_tape.root_body)
        self.duct_tape_body_id = self.sim.model.body_name2id(self.duct_tape.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"

            # position and rotation of yellow tape
            @sensor(modality=modality)
            def yellow_tape_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.yellow_tape_body_id])

            @sensor(modality=modality)
            def yellow_tape_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.yellow_tape_body_id], to="xyzw")

            # position and rotation of duct tape
            @sensor(modality=modality)
            def duct_tape_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.duct_tape_body_id])

            @sensor(modality=modality)
            def duct_tape_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.duct_tape_body_id], to="xyzw")

            sensors = [yellow_tape_pos, yellow_tape_quat, duct_tape_pos, duct_tape_quat]
            names = [s.__name__ for s in sensors]

            # Add gripper to object sensors
            pf = self.robots[0].robot_model.naming_prefix
            if self.env_configuration == "single-robot":
                # For bimanual robots
                if hasattr(self.robots[0], "robot_model") and "right" in dir(self.robots[0].robot_model):
                    pf_right = pf + "right_"
                    pf_left = pf + "left_"
                else:
                    pf_right = pf
                    pf_left = pf
            else:
                pf_right = self.robots[0].robot_model.naming_prefix
                pf_left = self.robots[1].robot_model.naming_prefix if len(self.robots) > 1 else pf_right

            @sensor(modality=modality)
            def gripper_to_yellow_tape(obs_cache):
                return (
                    obs_cache["yellow_tape_pos"] - obs_cache[f"{pf_right}eef_pos"]
                    if f"{pf_right}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper_to_duct_tape(obs_cache):
                return (
                    obs_cache["duct_tape_pos"] - obs_cache[f"{pf_right}eef_pos"]
                    if f"{pf_right}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors += [gripper_to_yellow_tape, gripper_to_duct_tape]
            names += [s.__name__ for s in [gripper_to_yellow_tape, gripper_to_duct_tape]]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # Set the collision object joints
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _check_success(self):
        """
        Check if yellow tape is placed in the duct tape

        Returns:
            bool: True if task has been completed
        """
        yellow_tape_pos = np.array(self.sim.data.body_xpos[self.yellow_tape_body_id])
        duct_tape_pos = np.array(self.sim.data.body_xpos[self.duct_tape_body_id])

        # Check if yellow tape is within duct tape radius and at appropriate height
        dist_xy = np.linalg.norm(yellow_tape_pos[:2] - duct_tape_pos[:2])
        return dist_xy < 0.08 and abs(yellow_tape_pos[2] - duct_tape_pos[2]) < 0.05
