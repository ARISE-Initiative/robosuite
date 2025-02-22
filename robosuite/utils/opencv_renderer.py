"""
opencv renderer class.
"""
import numpy as np
import cv2
import platform


class OpenCVRenderer:
    def __init__(self, sim):
        # TODO: update this appropriately - need to get screen dimensions
        self.width = 1280
        self.height = 800

        self.sim = sim
        self.set_camera(camera_id=0)
        self._window_name = "offscreen render"
        self._has_window = False
        self.keypress_callback = None

    def set_camera(self, camera_id=None, camera_name=None):
        """
        Set the camera view to the specified camera ID.

        Args:
            camera_id (int or list): id(s) of the camera to set the current viewer to
            camera_name (str or list or None): name(s) of the camera to set the current viewer to
        """

        # enforce exactly one arg
        assert (camera_id is not None) or (camera_name is not None)
        assert (camera_id is None) or (camera_name is None)

        if camera_id is not None:
            if isinstance(camera_id, int):
                camera_id = [camera_id]
            self.camera_names = [self.sim.model.camera_id2name(cam_id) for cam_id in camera_id]
        else:
            if isinstance(camera_name, str):
                camera_name = [camera_name]
            self.camera_names = list(camera_name)

    def render(self):
        # get frame with offscreen renderer (assumes that the renderer already exists)
        im = [
            self.sim.render(camera_name=cam_name, height=self.height, width=self.width)[..., ::-1]
            for cam_name in self.camera_names
        ]
        im = np.concatenate(im, axis=1) # concatenate horizontally

        # write frame to window
        im = np.flip(im, axis=0)
        cv2.imshow(self._window_name, im)
        if (platform.system() != "Darwin") and (not self._has_window):
            # move window to top left of screen, and ensure we only move window on creation
            cv2.moveWindow(self._window_name, 0, 0)
        key = cv2.waitKey(1)
        if self.keypress_callback is not None:
            self.keypress_callback(key)
        self._has_window = True

    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback

    def close_window(self):
        """
        Helper method to close the active window.
        """
        if self._has_window:
            cv2.destroyWindow(self._window_name)
            cv2.waitKey(1)
        self._has_window = False

    def close(self):
        """
        Any cleanup to close renderer.
        """

        # NOTE: assume that @sim will get cleaned up outside the renderer - just delete the reference
        self.sim = None

        # close window
        self.close_window()