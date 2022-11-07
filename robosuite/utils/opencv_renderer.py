"""
opencv renderer class.
"""
import cv2
import numpy as np


class OpenCVRenderer:
    def __init__(self, sim):
        # TODO: update this appropriately - need to get screen dimensions
        self.width = 1280
        self.height = 800

        self.sim = sim
        self.camera_name = self.sim.model.camera_id2name(0)

        self.keypress_callback = None

    def set_camera(self, camera_id):
        """
        Set the camera view to the specified camera ID.
        Args:
            camera_id (int): id of the camera to set the current viewer to
        """
        self.camera_name = self.sim.model.camera_id2name(camera_id)

    def render(self):
        # get frame with offscreen renderer (assumes that the renderer already exists)
        im = self.sim.render(camera_name=self.camera_name, height=self.height, width=self.width)[..., ::-1]

        # write frame to window
        im = np.flip(im, axis=0)
        cv2.imshow("offscreen render", im)
        key = cv2.waitKey(1)
        if self.keypress_callback:
            self.keypress_callback(key)

    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback

    def close(self):
        """
        Any cleanup to close renderer.
        """

        # NOTE: assume that @sim will get cleaned up outside the renderer - just delete the reference
        self.sim = None

        # close window
        cv2.destroyAllWindows()
