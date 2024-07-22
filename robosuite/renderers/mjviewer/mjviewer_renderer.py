import cv2
import numpy as np
from mujoco import viewer


class MjviewerRenderer:
    def __init__(self, env, camera_id=None, cam_config=None, sim=None):
        self.env = env
        self.camera_id = camera_id
        self.viewer = None
        self.camera_config = cam_config

        self.width = 1280
        self.height = 800
        self.sim = sim
        self.camera_name = self.sim.model.camera_id2name(0)

    def render(self):
        # get frame with offscreen renderer (assumes that the renderer already exists)
        im = self.sim.render( height=self.height, width=self.width)[..., ::-1]

        # write frame to window
        im = np.flip(im, axis=0)
        # cv2.imshow("offscreen render", im)
        return im

    def set_camera(self, camera_id):
        self.camera_id = camera_id

    def update(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(
                self.env.sim.model._model,
                self.env.sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )

            self.viewer.opt.geomgroup[0] = 0

            if self.camera_config is not None:
                self.viewer.cam.lookat = self.camera_config["lookat"]
                self.viewer.cam.distance = self.camera_config["distance"]
                self.viewer.cam.azimuth = self.camera_config["azimuth"]
                self.viewer.cam.elevation = self.camera_config["elevation"]

            if self.camera_id is not None:
                self.viewer.cam.type = 2
                self.viewer.cam.fixedcamid = self.camera_id

        self.viewer.sync()

    def reset(self):
        pass

    def close(self):

        self.sim = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback
