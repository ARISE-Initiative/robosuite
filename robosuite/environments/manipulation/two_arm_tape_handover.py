from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import BoxObject, CylinderObject, MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class TwoArmTapeHandover(TwoArmEnv):
    """
    This class corresponds to a two robot arm environment with a red cube on the left side
    and a grey bowl on the right side.

    Args:
        robots (str or list of str): Specification for specific robot(s)
            Note: Must be either 2 robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment if two robots inputted. Can be either:

            :`'parallel'`: Sets up the two robots next to each other on the -x side of the table
            :`'opposed'`: Sets up the two robots opposed from each others on the opposite +/-y sides of the table.

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

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        tables_boundary (3-tuple): x, y, and z dimensions of the table bounds. One table will be created at the center of
            this boundary

        table_friction (3-tuple): the three mujoco friction parameters for
            each table.

        cube_size (float): Size of the cube

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
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
        gripper_types="default",
        initialization_noise="default",
        tables_boundary=(0.8, 1.2, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        cube_size=0.04,
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
        camera_names="agentview",
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
        self.cube_size = cube_size

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

            - a discrete reward of 1.0 is provided when the cube is placed in the bowl

        Un-normalized max-wise components if using reward shaping:

            - Reaching: distance between cube and gripper
            - Grasping: reward for grasping cube
            - Placing: distance between cube and bowl

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
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        bowl_pos = np.array(self.sim.data.body_xpos[self.bowl_body_id])

        # use a shaping reward if specified
        if self.reward_shaping:
            # Reaching reward
            gripper_site_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
            dist_to_cube = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist_to_cube)
            reward += reaching_reward

            # Grasping reward
            cube_height = cube_pos[2]
            table_height = self.table_offsets[0, 2]
            if cube_height > table_height + 0.05:
                reward += 0.5

            # Placing reward
            dist_cube_to_bowl = np.linalg.norm(cube_pos - bowl_pos)
            placing_reward = 1 - np.tanh(10.0 * dist_cube_to_bowl)
            reward += placing_reward

            # Normalize
            reward = reward / 2.5

        # Else this is the sparse reward setting
        else:
            # Provide reward if cube is in bowl (within bowl radius and at appropriate height)
            dist_xy = np.linalg.norm(cube_pos[:2] - bowl_pos[:2])
            if dist_xy < 0.08 and abs(cube_pos[2] - bowl_pos[2]) < 0.05:
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
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[1.2434677502317038, 4.965421871106301e-08, 2.091455182752329],
            quat=[0.6530981063842773, 0.2710406184196472, 0.27104079723358154, 0.6530979871749878],
        )

        # # TODO: these cams seem sus
        # # Add shoulder cameras
        # mujoco_arena.set_camera(
        #     camera_name="shouldercamera0",
        #     pos=[0.4430096057365183, -1.0697399743660143, 1.3639950119362048],
        #     quat=[0.804057240486145, 0.5531665086746216, 0.11286306381225586, 0.18644218146800995],
        # )
        # mujoco_arena.set_camera(
        #     camera_name="shouldercamera1",
        #     pos=[-0.40900713993039983, 0.9613722572245062, 1.3084072951772754],
        #     quat=[0.15484197437763214, 0.12077208608388901, -0.5476858019828796, -0.8133130073547363],
        # )

        # Add relevant materials
        # Red material for cube
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        red_mat = CustomMaterial(
            texture="WoodRed",
            tex_name="red_cube",
            mat_name="red_cube_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # Grey material for bowl
        grey_mat = CustomMaterial(
            texture="Metal",
            tex_name="grey_bowl",
            mat_name="grey_bowl_mat",
            tex_attrib=tex_attrib,
            mat_attrib={**mat_attrib, "rgba": "0.5 0.5 0.5 1"},
        )

        # Initialize objects of interest
        # Custom Yellow Tape Object
        yellow_tape_xml_path = "/home/karimelrafi/MV-SAM3D/visualization/yellow_tape/yellow_tape/yellow_tape_yellow_tape_2v_s1a30_s2e30_20251224_121628/yellow_tape.xml"
        self.cube = MujocoXMLObject(
            fname=yellow_tape_xml_path,
            name="yellow_tape",
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

        duct_tape_xml_path = "/home/karimelrafi/MV-SAM3D/visualization/duct_tape/duct_tape/duct_tape_duct_tape_1_s1off_s2off_20251224_122111/duct_tape.xml"
        self.bowl = MujocoXMLObject(
            fname=duct_tape_xml_path,
            name="duct_tape",
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

        # Create list of objects
        self.objects = [self.cube, self.bowl]

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

        # Cube on left side of table
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CubeSampler",
                mujoco_objects=self.cube,
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

        # Bowl on right side of table
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="BowlSampler",
                mujoco_objects=self.bowl,
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

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional references for cube and bowl
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.bowl_body_id = self.sim.model.body_name2id(self.bowl.root_body)

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

            # position and rotation of cube
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.cube_body_id], to="xyzw")

            # position and rotation of bowl
            @sensor(modality=modality)
            def bowl_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.bowl_body_id])

            @sensor(modality=modality)
            def bowl_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.bowl_body_id], to="xyzw")

            sensors = [cube_pos, cube_quat, bowl_pos, bowl_quat]
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
            def gripper_to_cube(obs_cache):
                return (
                    obs_cache["cube_pos"] - obs_cache[f"{pf_right}eef_pos"]
                    if f"{pf_right}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper_to_bowl(obs_cache):
                return (
                    obs_cache["bowl_pos"] - obs_cache[f"{pf_right}eef_pos"]
                    if f"{pf_right}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors += [gripper_to_cube, gripper_to_bowl]
            names += [s.__name__ for s in [gripper_to_cube, gripper_to_bowl]]

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
        Check if cube is placed in the bowl

        Returns:
            bool: True if task has been completed
        """
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        bowl_pos = np.array(self.sim.data.body_xpos[self.bowl_body_id])

        # Check if cube is within bowl radius and at appropriate height
        dist_xy = np.linalg.norm(cube_pos[:2] - bowl_pos[:2])
        return dist_xy < 0.08 and abs(cube_pos[2] - bowl_pos[2]) < 0.05

