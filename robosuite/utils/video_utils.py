"""
A set of utilities for generating nice robosuite videos :D

NOTE: This requires cv2 package to be installed
"""

import numpy as np
import json
import h5py
import robosuite
from robosuite.wrappers import VisualizationWrapper, DomainRandomizationWrapper
from robosuite.utils.mjcf_utils import postprocess_model_xml
import robosuite.utils.transform_utils as T
import xml.etree.ElementTree as ET


class DemoPlaybackCamera:
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
        self.camera = camera
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
        env_name = self.f["data"].attrs["env"]
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
        self.env = robosuite.make(**env_info)

        # Optionally wrap with visualization wrapper
        if visualize_sites:
            self.env = VisualizationWrapper(env=self.env)

        # Optionally use domain randomization if specified
        self.use_dr = use_dr
        if self.use_dr:
            default_dr_args = {
                "seed": 1,
                "randomize_camera": False,
                "randomize_every_n_steps": 10,
            }
            default_dr_args.update(dr_args)
            self.env = DomainRandomizationWrapper(
                env=self.env,
                **default_dr_args,
            )

        # list of all demonstrations episodes
        self.demos = list(self.f["data"].keys())

        # Load episode 0 by default
        self.load_episode_xml(demo_num=0)

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
            self.env.sim.data.set_mocap_pos("cameramover", pos)
        if quat is not None:
            self.env.sim.data.set_mocap_quat("cameramover", T.convert_quat(quat, to='wxyz'))

        # Make sure changes propagate in sim
        self.env.sim.forward()

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
        xml = postprocess_model_xml(model_xml)
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
        else:           # replay from states
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
        assert (camera_elem is not None)

        # add mocap body
        mocap = ET.SubElement(wb, "body")
        mocap.set("name", "cameramover")
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
            point (3-array): (x,y,z) cartesian coordinates about which to rotate camera in camera frame
            axis (3-array): (ax,ay,az) axis about which to rotate camera in camera frame
            angle (float): how much to rotate about that direction

        Returns:
            2-tuple:
                pos: (x,y,z) updated camera position
                quat: (x,y,z,w) updated camera quaternion orientation
        """

        # current camera rotation + pos
        camera_pos = np.array(self.env.sim.data.get_mocap_pos("cameramover"))
        camera_rot = T.quat2mat(T.convert_quat(self.env.sim.data.get_mocap_quat("cameramover"), to='xyzw'))

        # rotate by angle and direction to get new camera rotation
        rad = np.pi * angle / 180.0
        R = T.rotation_matrix(rad, axis, point=point)
        camera_pose = np.zeros((4, 4))
        camera_pose[:3, :3] = camera_rot
        camera_pose[:3, 3] = camera_pos
        camera_pose = R @ camera_pose

        # Update camera pose
        pos, quat = camera_pose[:3, 3], T.mat2quat(camera_pose[:3, :3])
        self.set_camera_pose(pos=pos, quat=quat)

        return pos, quat
