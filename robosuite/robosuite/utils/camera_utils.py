"""
This module includes:

- Utility classes for modifying sim cameras

- Utility functions for performing common camera operations such as retrieving
camera matrices and transforming from world to camera frame or vice-versa.
"""
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import DomainRandomizationWrapper, VisualizationWrapper


def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    """
    Obtains camera intrinsic matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix(sim, camera_name):
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
    Return:
        R (np.array): 4x4 camera extrinsic matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    R = T.make_pose(camera_pos, camera_rot)

    # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
    camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    R = R @ camera_axis_correction
    return R


def get_camera_transform_matrix(sim, camera_name, camera_height, camera_width):
    """
    Camera transform matrix to project from world coordinates to pixel coordinates.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 4x4 camera matrix to project from world coordinates to pixel coordinates
    """
    R = get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    K = get_camera_intrinsic_matrix(
        sim=sim, camera_name=camera_name, camera_height=camera_height, camera_width=camera_width
    )
    K_exp = np.eye(4)
    K_exp[:3, :3] = K

    # Takes a point in world, transforms to camera frame, and then projects onto image plane.
    return K_exp @ T.pose_inv(R)


def get_camera_segmentation(sim, camera_name, camera_height, camera_width):
    """
    Obtains camera segmentation matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        im (np.array): 2-channel segmented image where the first contains the
            geom types and the second contains the geom IDs
    """
    return sim.render(camera_name=camera_name, height=camera_height, width=camera_width, segmentation=True)[::-1]


def get_real_depth_map(sim, depth_map):
    """
    By default, MuJoCo will return a depth map that is normalized in [0, 1]. This
    helper function converts the map so that the entries correspond to actual distances.

    (see https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py#L742)

    Args:
        sim (MjSim): simulator instance
        depth_map (np.array): depth map with values normalized in [0, 1] (default depth map
            returned by MuJoCo)
    Return:
        depth_map (np.array): depth map that corresponds to actual distances
    """
    # Make sure that depth values are normalized
    assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)
    extent = sim.model.stat.extent
    far = sim.model.vis.map.zfar * extent
    near = sim.model.vis.map.znear * extent
    return near / (1.0 - depth_map * (1.0 - near / far))


def project_points_from_world_to_camera(points, world_to_camera_transform, camera_height, camera_width):
    """
    Helper function to project a batch of points in the world frame
    into camera pixels using the world to camera transformation.

    Args:
        points (np.array): 3D points in world frame to project onto camera pixel locations. Should
            be shape [..., 3].
        world_to_camera_transform (np.array): 4x4 Tensor to go from robot coordinates to pixel
            coordinates.
        camera_height (int): height of the camera image
        camera_width (int): width of the camera image

    Return:
        pixels (np.array): projected pixel indices of shape [..., 2]
    """
    assert points.shape[-1] == 3  # last dimension must be 3D
    assert len(world_to_camera_transform.shape) == 2
    assert world_to_camera_transform.shape[0] == 4 and world_to_camera_transform.shape[1] == 4

    # convert points to homogenous coordinates -> (px, py, pz, 1)
    ones_pad = np.ones(points.shape[:-1] + (1,))
    points = np.concatenate((points, ones_pad), axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do robot frame to pixels transform
    mat_reshape = [1] * len(points.shape[:-1]) + [4, 4]
    cam_trans = world_to_camera_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    pixels = np.matmul(cam_trans, points[..., None])[..., 0]  # shape [..., 4]

    # re-scaling from homogenous coordinates to recover pixel values
    # (x, y, z) -> (x / z, y / z)
    pixels = pixels / pixels[..., 2:3]
    pixels = pixels[..., :2].round().astype(int)  # shape [..., 2]

    # swap first and second coordinates to get pixel indices that correspond to (height, width)
    # and also clip pixels that are out of range of the camera image
    pixels = np.concatenate(
        (
            pixels[..., 1:2].clip(0, camera_height - 1),
            pixels[..., 0:1].clip(0, camera_width - 1),
        ),
        axis=-1,
    )

    return pixels


def transform_from_pixels_to_world(pixels, depth_map, camera_to_world_transform):
    """
    Helper function to take a batch of pixel locations and the corresponding depth image
    and transform these points from the camera frame to the world frame.

    Args:
        pixels (np.array): pixel coordinates of shape [..., 2]
        depth_map (np.array): depth images of shape [..., H, W, 1]
        camera_to_world_transform (np.array): 4x4 Tensor to go from pixel coordinates to world
            coordinates.

    Return:
        points (np.array): 3D points in robot frame of shape [..., 3]
    """

    # make sure leading dimensions are consistent
    pixels_leading_shape = pixels.shape[:-1]
    depth_map_leading_shape = depth_map.shape[:-3]
    assert depth_map_leading_shape == pixels_leading_shape

    # sample from the depth map using the pixel locations with bilinear sampling
    pixels = pixels.astype(float)
    im_h, im_w = depth_map.shape[-2:]
    depth_map_reshaped = depth_map.reshape(-1, im_h, im_w, 1)
    z = bilinear_interpolate(im=depth_map_reshaped, x=pixels[..., 1:2], y=pixels[..., 0:1])
    z = z.reshape(*depth_map_leading_shape, 1)  # shape [..., 1]

    # form 4D homogenous camera vector to transform - [x * z, y * z, z, 1]
    # (note that we need to swap the first 2 dimensions of pixels to go from pixel indices
    # to camera coordinates)
    cam_pts = [pixels[..., 1:2] * z, pixels[..., 0:1] * z, z, np.ones_like(z)]
    cam_pts = np.concatenate(cam_pts, axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do camera to robot frame transform
    mat_reshape = [1] * len(cam_pts.shape[:-1]) + [4, 4]
    cam_trans = camera_to_world_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    points = np.matmul(cam_trans, cam_pts[..., None])[..., 0]  # shape [..., 4]
    return points[..., :3]


def bilinear_interpolate(im, x, y):
    """
    Bilinear sampling for pixel coordinates x and y from source image im.
    Taken from https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


class CameraMover:
    """
    A class for manipulating a camera.

    WARNING: This class will initially RE-INITIALIZE the environment.

    Args:
        env (MujocoEnv): Mujoco environment to modify camera
        camera (str): Which camera to mobilize during playback, e.g.: frontview, agentview, etc.
        init_camera_pos (None or 3-array): If specified, should be the (x,y,z) global cartesian pos to
            initialize camera to
        init_camera_quat (None or 4-array): If specified, should be the (x,y,z,w) global quaternion orientation to
            initialize camera to
    """

    def __init__(
        self,
        env,
        camera="frontview",
        init_camera_pos=None,
        init_camera_quat=None,
    ):
        # Store relevant values and initialize other values
        self.env = env
        self.camera = camera
        self.mover_body_name = f"{self.camera}_cameramover"

        # Get state
        state = self.env.sim.get_state().flatten()

        # Grab environment xml
        xml = env.sim.model.get_xml()

        # Modify xml to add mocap to move camera around
        xml = self.modify_xml_for_camera_movement(xml=xml, camera_name=self.camera)

        # Reset the environment and restore the state
        self.env.reset_from_xml_string(xml)
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()

        # Set initial camera pose
        self.set_camera_pose(pos=init_camera_pos, quat=init_camera_quat)

    def set_camera_pose(self, pos=None, quat=None):
        """
        Sets the camera pose, which optionally includes position and / or quaternion

        Args:
            pos (None or 3-array): If specified, should be the (x,y,z) global cartesian pos to set camera to
            quat (None or 4-array): If specified, should be the (x,y,z,w) global quaternion orientation to set camera to
        """
        if pos is not None:
            self.env.sim.data.set_mocap_pos(self.mover_body_name, pos)
        if quat is not None:
            self.env.sim.data.set_mocap_quat(self.mover_body_name, T.convert_quat(quat, to="wxyz"))

        # Make sure changes propagate in sim
        self.env.sim.forward()

    def get_camera_pose(self):
        """
        Grab the current camera pose, which optionally includes position and / or quaternion

        Returns:
            2-tuple:
                - 3-array: (x,y,z) camera global cartesian pos
                - 4-array: (x,y,z,w) camera global quaternion orientation
        """
        # Grab values from sim
        pos = self.env.sim.data.get_mocap_pos(self.mover_body_name)
        quat = T.convert_quat(self.env.sim.data.get_mocap_quat(self.mover_body_name), to="xyzw")

        return pos, quat

    def modify_xml_for_camera_movement(self, xml, camera_name):
        """
        Cameras in mujoco are 'fixed', so they can't be moved by default.
        Although it's possible to hack position movement, rotation movement
        does not work. An alternative is to attach a camera to a mocap body,
        and move the mocap body.

        This function modifies the camera with name @camera_name in the xml
        by attaching it to a mocap body that can move around freely. In this
        way, we can move the camera by moving the mocap body.

        See http://www.mujoco.org/forum/index.php?threads/move-camera.2201/ for
        further details.

        Args:
            xml (str): Mujoco sim XML file as a string
            camera_name (str): Name of camera to tune
        """
        tree = ET.fromstring(xml)
        wb = tree.find("worldbody")

        # find the correct camera
        camera_elem = None
        cameras = wb.findall("camera")
        for camera in cameras:
            if camera.get("name") == camera_name:
                camera_elem = camera
                break
        assert camera_elem is not None

        # add mocap body
        mocap = ET.SubElement(wb, "body")
        mocap.set("name", self.mover_body_name)
        mocap.set("mocap", "true")
        mocap.set("pos", camera.get("pos"))
        mocap.set("quat", camera.get("quat"))
        new_camera = ET.SubElement(mocap, "camera")
        new_camera.set("mode", "fixed")
        new_camera.set("name", camera.get("name"))
        new_camera.set("pos", "0 0 0")

        # remove old camera element
        wb.remove(camera_elem)

        return ET.tostring(tree, encoding="utf8").decode("utf8")

    def rotate_camera(self, point, axis, angle):
        """
        Rotate the camera view about a direction (in the camera frame).

        Args:
            point (None or 3-array): (x,y,z) cartesian coordinates about which to rotate camera in camera frame. If None,
                assumes the point is the current location of the camera
            axis (3-array): (ax,ay,az) axis about which to rotate camera in camera frame
            angle (float): how much to rotate about that direction

        Returns:
            2-tuple:
                pos: (x,y,z) updated camera position
                quat: (x,y,z,w) updated camera quaternion orientation
        """
        # current camera rotation + pos
        camera_pos = np.array(self.env.sim.data.get_mocap_pos(self.mover_body_name))
        camera_rot = T.quat2mat(T.convert_quat(self.env.sim.data.get_mocap_quat(self.mover_body_name), to="xyzw"))

        # rotate by angle and direction to get new camera rotation
        rad = np.pi * angle / 180.0
        R = T.rotation_matrix(rad, axis, point=point)
        camera_pose = np.zeros((4, 4))
        camera_pose[:3, :3] = camera_rot
        camera_pose[:3, 3] = camera_pos
        camera_pose = camera_pose @ R

        # Update camera pose
        pos, quat = camera_pose[:3, 3], T.mat2quat(camera_pose[:3, :3])
        self.set_camera_pose(pos=pos, quat=quat)

        return pos, quat

    def move_camera(self, direction, scale):
        """
        Move the camera view along a direction (in the camera frame).

        Args:
            direction (3-array): direction vector for where to move camera in camera frame
            scale (float): how much to move along that direction
        """
        # current camera rotation + pos
        camera_pos = np.array(self.env.sim.data.get_mocap_pos(self.mover_body_name))
        camera_quat = self.env.sim.data.get_mocap_quat(self.mover_body_name)
        camera_rot = T.quat2mat(T.convert_quat(camera_quat, to="xyzw"))

        # move along camera frame axis and set new position
        camera_pos += scale * camera_rot.dot(direction)
        self.set_camera_pose(pos=camera_pos)

        return camera_pos, camera_quat


class DemoPlaybackCameraMover(CameraMover):
    """
    A class for playing back demonstrations and recording the resulting frames with the flexibility of a mobile camera
    that can be set manually or panned automatically frame-by-frame

    Note: domain randomization is also supported for playback!

    Args:
        demo (str): absolute fpath to .hdf5 demo
        env_config (None or dict): (optional) values to override inferred environment information from demonstration.
            (e.g.: camera h / w, depths, segmentations, etc...)
            Any value not specified will be inferred from the extracted demonstration metadata
            Note that there are some specific arguments that MUST be set a certain way, if any of these values
            are specified with @env_config, an error will be raised
        replay_from_actions (bool): If True, will replay demonstration's actions. Otherwise, replays will be hardcoded
            from the demonstration states
        visualize_sites (bool): If True, will visualize sites during playback. Note that this CANNOT be paired
            simultaneously with camera segmentations
        camera (str): Which camera to mobilize during playback, e.g.: frontview, agentview, etc.
        init_camera_pos (None or 3-array): If specified, should be the (x,y,z) global cartesian pos to
            initialize camera to
        init_camera_quat (None or 4-array): If specified, should be the (x,y,z,w) global quaternion orientation to
            initialize camera to
        use_dr (bool): If True, will use domain randomization during playback
        dr_args (None or dict): If specified, will set the domain randomization wrapper arguments if using dr
    """

    def __init__(
        self,
        demo,
        env_config=None,
        replay_from_actions=False,
        visualize_sites=False,
        camera="frontview",
        init_camera_pos=None,
        init_camera_quat=None,
        use_dr=False,
        dr_args=None,
    ):
        # Store relevant values and initialize other values
        self.camera_id = None
        self.replay_from_actions = replay_from_actions
        self.states = None
        self.actions = None
        self.step = None
        self.n_steps = None
        self.current_ep = None
        self.started = False

        # Load the demo
        self.f = h5py.File(demo, "r")

        # Extract relevant info
        env_info = json.loads(self.f["data"].attrs["env_info"])

        # Construct default env arguments
        default_args = {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_camera_obs": True,
            "reward_shaping": True,
            "hard_reset": False,
            "camera_names": camera,
        }

        # If custom env_config is specified, make sure that there's no overlap with default args and merge with config
        if env_config is not None:
            for k in env_config.keys():
                assert k not in default_args, f"Key {k} cannot be specified in env_config!"
            env_info.update(env_config)

        # Merge in default args
        env_info.update(default_args)

        # Create env
        env = robosuite.make(**env_info)

        # Optionally wrap with visualization wrapper
        if visualize_sites:
            env = VisualizationWrapper(env=self.env)

        # Optionally use domain randomization if specified
        self.use_dr = use_dr
        if self.use_dr:
            default_dr_args = {
                "seed": 1,
                "randomize_camera": False,
                "randomize_every_n_steps": 10,
            }
            default_dr_args.update(dr_args)
            env = DomainRandomizationWrapper(
                env=self.env,
                **default_dr_args,
            )

        # list of all demonstrations episodes
        self.demos = list(self.f["data"].keys())

        # Run super init
        super().__init__(
            env=env,
            camera=camera,
            init_camera_pos=init_camera_pos,
            init_camera_quat=init_camera_quat,
        )

        # Load episode 0 by default
        self.load_episode_xml(demo_num=0)

    def load_episode_xml(self, demo_num):
        """
        Loads demo episode with specified @demo_num into the simulator.

        Args:
            demo_num (int): Demonstration number to load
        """
        # Grab raw xml file
        ep = self.demos[demo_num]
        model_xml = self.f[f"data/{ep}"].attrs["model_file"]

        # Reset environment
        self.env.reset()
        xml = self.env.edit_model_xml(model_xml)
        xml = self.modify_xml_for_camera_movement(xml, camera_name=self.camera)
        self.env.reset_from_xml_string(xml)
        self.env.sim.reset()

        # Update camera info
        self.camera_id = self.env.sim.model.camera_name2id(self.camera)

        # Load states and actions
        self.states = self.f[f"data/{ep}/states"].value
        self.actions = np.array(self.f[f"data/{ep}/actions"].value)

        # Set initial state
        self.env.sim.set_state_from_flattened(self.states[0])

        # Reset step count and set current episode number
        self.step = 0
        self.n_steps = len(self.actions)
        self.current_ep = demo_num

        # Notify user of loaded episode
        print(f"Loaded episode {demo_num}.")

    def grab_next_frame(self):
        """
        Grabs the next frame in the demo sequence by stepping the simulation and returning the resulting value(s)

        Returns:
            dict: Keyword-mapped np.arrays from the demonstration sequence, corresponding to all image modalities used
                in the playback environment (e.g.: "image", "depth", "segmentation_instance")
        """
        # Make sure the episode isn't completed yet, if so, we load the next episode
        if self.step == self.n_steps:
            self.load_episode_xml(demo_num=self.current_ep + 1)

        # Step the environment and grab obs
        if self.replay_from_actions:
            obs, _, _, _ = self.env.step(self.actions[self.step])
        else:  # replay from states
            self.env.sim.set_state_from_flattened(self.states[self.step + 1])
            if self.use_dr:
                self.env.step_randomization()
            self.env.sim.forward()
            obs = self.env._get_observation()

        # Increment the step counter
        self.step += 1

        # Return all relevant frames
        return {k.split(f"{self.camera}_")[-1]: obs[k] for k in obs if self.camera in k}

    def grab_episode_frames(self, demo_num, pan_point=(0, 0, 0.8), pan_axis=(0, 0, 1), pan_rate=0.01):
        """
        Playback entire episode @demo_num, while optionally rotating the camera about point @pan_point and
            axis @pan_axis if @pan_rate > 0

        Args:
            demo_num (int): Demonstration episode number to load for playback
            pan_point (3-array): (x,y,z) cartesian coordinates about which to rotate camera in camera frame
            pan_direction (3-array): (ax,ay,az) axis about which to rotate camera in camera frame
            pan_rate (float): how quickly to pan camera if not 0

        Returns:
            dict: Keyword-mapped stacked np.arrays from the demonstration sequence, corresponding to all image
                modalities used in the playback environment (e.g.: "image", "depth", "segmentation_instance")

        """
        # First, load env
        self.load_episode_xml(demo_num=demo_num)

        # Initialize dict to return
        obs = self.env._get_observation()
        frames_dict = {k.split(f"{self.camera}_")[-1]: [] for k in obs if self.camera in k}

        # Continue to loop playback steps while there are still frames left in the episode
        while self.step < self.n_steps:
            # Take playback step and add frames
            for k, frame in self.grab_next_frame().items():
                frames_dict[k].append(frame)

            # Update camera pose
            self.rotate_camera(point=pan_point, axis=pan_axis, angle=pan_rate)

        # Stack all frames and return
        return {k: np.stack(frames) for k, frames in frames_dict.items()}
