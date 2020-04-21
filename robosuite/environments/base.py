from collections import OrderedDict
import time
import numpy as np

from mujoco_py import MjSim, MjRenderContextOffscreen
from mujoco_py import load_model_from_xml
import robosuite.utils.transform_utils as T

from robosuite.utils import SimulationError, XMLError, MujocoPyRenderer
try:
    from mujoco_py.generated.const import RND_SEGMENT, RND_IDCOLOR
except:
    print("WARNING: could not import Mujoco 200 constants!")


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
        _unregistered_envs = ["MujocoEnv", "SawyerEnv", "PandaEnv", "BaxterEnv"]

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls

class CustomMjSim(MjSim):
    """
    Custom mjsim to overwrite the render function in order to
    workaround the mujoco-py bug of multiple env rendering issue
    """
    def __new__(cls, *args, **kw):
        return super().__new__(cls, *args, **kw)

    def __init__(self, mjpy_model):
        # super().__init__()
        pass

    def render(self, **kwargs):
        # render twice to work around the bug!
        super().render(**kwargs)
        return super().render(**kwargs)

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
        camera_real_depth=False,
        camera_segmentation=False,
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

            camera_real_depth (bool): True if convert depth to real depth in meters

            camera_segmentation (bool): True if also return semantic segmentation of the camera view
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
        self.camera_real_depth = camera_real_depth
        self.camera_segmentation = camera_segmentation

        # self._reset_internal()
        self.reset()

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

        # instantiate simulation from MJCF model
        self._load_model()
        self.mjpy_model = self.model.get_model(mode="mujoco_py")
        self.sim = CustomMjSim(self.mjpy_model)

        self._reset_internal()
        self.sim.forward()
        return self._get_observation()

    def _reset_internal(self):
        """Resets simulation internal configurations."""
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

            # make sure mujoco-py doesn't block rendering frames.
            # (see https://github.com/StanfordVL/robosuite/issues/39)
            self.viewer.viewer._render_every_frame = True

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
        di = OrderedDict()
        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
                if self.camera_real_depth:
                    di["depth"] = self.z_buffer_to_real_depth(di["depth"])
            else:
                di["image"] = camera_obs

            if self.camera_segmentation:
                di["segmentation"] = self.render_segmentation(
                    self.camera_name, camera_width=self.camera_width, camera_height=self.camera_height)
        return di

    def step(self, action):
        """Takes a step in simulation with control command @action."""
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
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
        self.sim = CustomMjSim(self.mjpy_model)

        self._reset_internal()
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
        R = R @ camera_axis_correction
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
        return T.matrix_inverse(P)

    def from_pixel_to_world(self, u, v, w, camera_name=None):
        """
        @input u, v: pixel
        @input w: depth
        @returns X: numpy array of shape (3,); x, y, z in world coordinates
        """
        assert 0 <= u < self.camera_width
        assert 0 <= v < self.camera_height

        if camera_name is None:
            camera_name = self.camera_name

        P_inv = self.get_camera_inverse_transform_matrix(camera_name=camera_name)
        X = P_inv @ np.array([u * w, v * w, w, 1.])
        return X[:3]

    def batch_from_pixel_to_world(self, pixels, depths, camera_name=None):
        assert(len(pixels.shape) == 2)
        assert(len(depths.shape) == 1)
        assert(pixels.shape[1] == 2)
        assert(depths.shape[0] == pixels.shape[0])
        assert(np.all(np.logical_and(pixels[:, 0] >= 0, pixels[:, 0] < self.camera_width)))
        assert(np.all(np.logical_and(pixels[:, 1] >= 0, pixels[:, 1] < self.camera_height)))
        assert(np.all(depths >= 0))

        if camera_name is None:
            camera_name = self.camera_name
        P_inv = self.get_camera_inverse_transform_matrix(camera_name=camera_name)
        depths = depths[:, None]
        pts = np.concatenate((pixels * depths, depths, np.ones_like(depths)), axis=1)  # [u * w, v * w, w, 1.]
        pts_3d = (P_inv @ pts.transpose()).transpose()
        return pts_3d[:, :3]

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

    def batch_from_world_to_pixel(self, pts, camera_name=None, return_valid_index=False):
        assert(pts.shape[1] == 3)
        assert(len(pts.shape) == 2)
        if camera_name is None:
            camera_name = self.camera_name

        P = self.get_camera_transform_matrix(camera_name=camera_name)
        x_hom = np.concatenate((pts, np.ones_like(pts[:, [0]])), axis=1)   # [x, y, z, 1]
        pixels = (P @ x_hom.transpose()).transpose()
        pixels /= pixels[:, [2]]
        valid_pixels_x = np.logical_and(pixels[:, 0] >= 0, pixels[:, 0] < (self.camera_width - 0.5))
        valid_pixels_y = np.logical_and(pixels[:, 1] >= 0, pixels[:, 1] < (self.camera_height - 0.5))
        valid_pixel_index = np.where(np.logical_and(valid_pixels_x, valid_pixels_y))[0]
        pixels = pixels[valid_pixel_index, :2]
        if return_valid_index:
            return pixels, valid_pixel_index
        else:
            return pixels

    def z_buffer_to_real_depth(self, z_buffer):
        """
        Converts z_buffer value (range [0, 1]) to real depth
        source: https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py#L742
        :param z_buffer: N-d array z buffer values from gl renderer
        :return: real_depth: N-d array depth in meters
        """
        extent = self.sim.model.stat.extent
        far = self.sim.model.vis.map.zfar * extent
        near = self.sim.model.vis.map.znear * extent
        return near / (1. - z_buffer * (1. - near / far))

    def render_segmentation(self, camera_name, camera_width=None, camera_height=None):
        """
        Get semantic segmentation map of a given view
        Note that this requires a fork of mujoco-py: https://github.com/StanfordVL/mujoco-py
        Ref: https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py#L751
        :param camera_name: camera name
        :return: a semantic segmentation map with each element corresponding to a body id
        """
        scn = self.sim.render_contexts[0].scn
        scn.flags[RND_SEGMENT] = True
        scn.flags[RND_IDCOLOR] = True
        if camera_width is None:
            camera_width = self.camera_width
        if camera_height is None:
            camera_height = self.camera_height
        frame = self.sim.render(camera_width, camera_height, camera_name=camera_name)
        frame = frame[..., 0] + frame[..., 1] * 2 ** 8 + frame[..., 2] * 2 ** 16
        segid2output = np.full((self.sim.model.ngeom + 1), fill_value=-1,
                               dtype=np.int32)  # Seg id cannot be > ngeom + 1.
        geoms = self.sim.render_contexts[0].get_geoms()
        mappings = np.array([(g['segid'], self.sim.model.geom_bodyid[g['objid']]) for g in geoms], dtype=np.int32)
        segid2output[mappings[:, 0] + 1] = mappings[:, 1]
        frame = segid2output[frame]
        scn.flags[RND_SEGMENT] = False
        scn.flags[RND_IDCOLOR] = False
        return frame

    def _destroy_viewer(self):
        # if there is an active viewer window, destroy it
        if self.viewer is not None:
            self.viewer.close()  # change this to viewer.finish()?
            self.viewer = None

    def close(self):
        """Do any cleanup necessary here."""
        self._destroy_viewer()
