from collections import OrderedDict
from mujoco_py import MjSim, MjRenderContextOffscreen
from mujoco_py import load_model_from_xml

from robosuite.utils import SimulationError, XMLError, MujocoPyRenderer
import numpy as np
import robosuite.utils.transform_utils as T

REGISTERED_ENVS = {}


def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


def make(env_name, *args, **kwargs):
    """Try to get the equivalent functionality of gym.make in a sloppy way."""
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )
    return REGISTERED_ENVS[env_name](*args, **kwargs)


class EnvMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["MujocoEnv", "SawyerEnv", "PandaEnv", "BaxterEnv",
                              "SawyerRobotArmEnv", "PandaRobotArmEnv"]

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls


class MujocoEnv(metaclass=EnvMeta):
    """Initializes a Mujoco Environment."""

    def __init__(
        self,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        use_camera_obs=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
    ):
        """
        Args:

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every simulated second. This sets the amount of simulation time
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
        """

        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.control_freq = control_freq
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.viewer = None
        self.model = None

        # settings for camera observations
        self.use_camera_obs = use_camera_obs
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Camera observations require an offscreen renderer.")
        self.camera_name = camera_name
        if self.use_camera_obs and self.camera_name is None:
            raise ValueError("Must specify camera name when using camera obs")
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_depth = camera_depth

        self._reset_internal()

    def initialize_time(self, control_freq):
        """
        Initializes the time constants used for simulation.
        """
        self.cur_time = 0
        self.model_timestep = self.sim.model.opt.timestep
        if self.model_timestep <= 0:
            raise XMLError("xml model defined non-positive time step")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError(
                "control frequency {} is invalid".format(control_freq)
            )
        self.control_timestep = 1. / control_freq

    def _load_model(self):
        """Loads an xml model, puts it in self.model"""
        pass

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        pass

    def reset(self):
        """Resets simulation."""
        # TODO(yukez): investigate black screen of death
        # if there is an active viewer window, destroy it
        self._destroy_viewer()
        self._reset_internal()
        self.sim.forward()
        return self._get_observation()

    def _reset_internal(self):
        """Resets simulation internal configurations."""
        # instantiate simulation from MJCF model
        self._load_model()
        self.mjpy_model = self.model.get_model(mode="mujoco_py")
        self.sim = MjSim(self.mjpy_model)
        self.initialize_time(self.control_freq)

        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

        elif self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(self.sim)
                self.sim.add_render_context(render_context)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.sim._render_context_offscreen.vopt.geomgroup[1] = (
                1 if self.render_visual_mesh else 0
            )

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

    def _get_observation(self):
        """Returns an OrderedDict containing observations [(name_string, np.array), ...]."""
        return OrderedDict()

    def step(self, action):
        """Takes a step in simulation with control command @action."""
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1
        policy_step = True
        for i in range(int(self.control_timestep / self.model_timestep)):
            self._pre_action(action, policy_step)
            self.sim.step()
            policy_step = False

        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)
        return self._get_observation(), reward, done, info

    def _pre_action(self, action, policy_step=False):
        """Do any preprocessing before taking an action."""
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        """Do any housekeeping after taking an action."""
        reward = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        return reward, self.done, {}

    def reward(self, action):
        """Reward should be a function of state and action."""
        return 0

    def render(self):
        """
        Renders to an on-screen window.
        """
        self.viewer.render()

    def observation_spec(self):
        """
        Returns an observation as observation specification.

        An alternative design is to return an OrderedDict where the keys
        are the observation names and the values are the shapes of observations.
        We leave this alternative implementation commented out, as we find the
        current design is easier to use in practice.
        """
        observation = self._get_observation()
        return observation

        # observation_spec = OrderedDict()
        # for k, v in observation.items():
        #     observation_spec[k] = v.shape
        # return observation_spec

    def action_spec(self):
        """
        Action specification should be implemented in subclasses.

        Action space is represented by a tuple of (low, high), which are two numpy
        vectors that specify the min/max action limits per dimension.
        """
        raise NotImplementedError

    def reset_from_xml_string(self, xml_string):
        """Reloads the environment from an XML description of the environment."""

        # if there is an active viewer window, destroy it
        self.close()

        # load model from xml
        self.mjpy_model = load_model_from_xml(xml_string)

        self.sim = MjSim(self.mjpy_model)
        self.initialize_time(self.control_freq)
        if self.has_renderer and self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

        elif self.has_offscreen_renderer:
            render_context = MjRenderContextOffscreen(self.sim)
            render_context.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            render_context.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0
            self.sim.add_render_context(render_context)

        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

        # necessary to refresh MjData
        self.sim.forward()

    def find_contacts(self, geoms_1, geoms_2):
        """
        Finds contact between two geom groups.

        Args:
            geoms_1: a list of geom names (string)
            geoms_2: another list of geom names (string)

        Returns:
            iterator of all contacts between @geoms_1 and @geoms_2
        """
        for contact in self.sim.data.contact[0 : self.sim.data.ncon]:
            # check contact geom in geoms
            c1_in_g1 = self.sim.model.geom_id2name(contact.geom1) in geoms_1
            c2_in_g2 = self.sim.model.geom_id2name(contact.geom2) in geoms_2
            # check contact geom in geoms (flipped)
            c2_in_g1 = self.sim.model.geom_id2name(contact.geom2) in geoms_1
            c1_in_g2 = self.sim.model.geom_id2name(contact.geom1) in geoms_2
            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
                yield contact

    def _check_contact(self):
        """Returns True if gripper is in contact with an object."""
        return False

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        return False

    def _destroy_viewer(self):
        # if there is an active viewer window, destroy it
        if self.viewer is not None:
            self.viewer.close()  # change this to viewer.finish()?
            self.viewer = None

    ### added some camera methods ###
    def _get_camera_intrinsic_matrix(self, camera_name=None):
        """
        Obtains camera internal matrix from other parameters. A 3X3 matrix.
        """
        if camera_name is None:
            camera_name = self.camera_name
        cam_id = self.sim.model.camera_name2id(camera_name)
        height, width = self.camera_height, self.camera_width
        fovy = self.sim.model.cam_fovy[cam_id]
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
        return K

    def _get_camera_extrinsic_matrix(self, camera_name=None):
        """
        Returns a 4x4 homogenous matrix corresponding to the camera pose in the
        world frame. MuJoCo has a weird convention for how it sets up the
        camera body axis, so we also apply a correction so that the x and y
        axis are along the camera view and the z axis points along the 
        viewpoint.

        Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        """
        if camera_name is None:
            camera_name = self.camera_name

        cam_id = self.sim.model.camera_name2id(camera_name)
        camera_pos = self.sim.model.cam_pos[cam_id]
        camera_rot = self.sim.model.cam_mat0[cam_id].reshape(3, 3)
        R = T.make_pose(camera_pos, camera_rot)

        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array([
            [1., 0., 0., 0.], 
            [0., -1., 0., 0.], 
            [0., 0., -1., 0.], 
            [0., 0., 0., 1.]]
        )
        R = camera_axis_correction @ R
        return R

    def get_camera_transform_matrix(self, camera_name=None):
        """
        Camera transform matrix to project from world coordinates to pixel coordinates.
        """
        if camera_name is None:
            camera_name = self.camera_name

        R = self._get_camera_extrinsic_matrix(camera_name=camera_name)
        K = self._get_camera_intrinsic_matrix(camera_name=camera_name)
        K_exp = np.eye(4)
        K_exp[:3, :3] = K

        # Takes a point in world, transforms to camera frame, and then projects onto image plane.
        return K_exp @ T.pose_inv(R)

    def get_camera_inverse_transform_matrix(self, camera_name=None):
        """Exposes the 4X4 homogeneous transform matrix from camera frame to world frame."""
        if camera_name is None:
            camera_name = self.camera_name
        P = self.get_camera_transform_matrix(camera_name=camera_name)
        return np.linalg.inv(P)

    def from_pixel_to_world(self, u, v, w, camera_name=None):
        """
        @input u, v: pixel
        @input w: a scale, for homogeneous transformation
        @returns X: numpy array of shape (3,); x, y, z in world coordinates
        """
        assert 0 <= u < self.camera_width
        assert 0 <= v < self.camera_height

        if camera_name is None:
            camera_name = self.camera_name

        P_inv = self.get_camera_inverse_transform_matrix(camera_name=camera_name)
        X = P_inv @ np.array([u * w, v * w, w, 1.])
        return X[:3]

    def from_world_to_pixel(self, x, camera_name=None):
        """
        @input x: numpy array of shape (3,); x, y, z in world coordinates
        @returns u, v: pixel coordinates (not rounded)
        """
        assert x.shape[0] == 3
        assert len(x.shape) == 1

        if camera_name is None:
            camera_name = self.camera_name

        P = self.get_camera_transform_matrix(camera_name=camera_name)
        pixel = P @ np.array([x[0], x[1], x[2], 1.])

        # account for homogenous coordinates
        pixel /= pixel[2]
        u, v = pixel[:2]

        assert 0 <= u < self.camera_width
        assert 0 <= v < self.camera_height
        return u, v

    def close(self):
        """Do any cleanup necessary here."""
        self._destroy_viewer()
