from mujoco import viewer


DEFAULT_CAM = {
   "lookat": [2.25, -1, 1.05312667],
   "distance": 5,
   "azimuth": 89.70301806083651,
   "elevation":-18.02177994296577
}




class MjviewerRenderer:
    def __init__(self, env, camera_id=None, cam_config=None):
        self.env = env
        self.camera_id = camera_id
        self.viewer = None
        self.camera_config = cam_config if cam_config is not None else DEFAULT_CAM

    def render(self):
        pass

    def update(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(
                self.env.sim.model._model,
                self.env.sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )

            self.viewer.opt.geomgroup[0] = 0

            if self.camera_id is not None:
                self.viewer.cam.type = 2
                self.viewer.cam.fixedcamid = self.camera_id
            else:
                self.viewer.cam.lookat = self.camera_config["lookat"]
                self.viewer.cam.distance = self.camera_config["distance"]
                self.viewer.cam.azimuth = self.camera_config["azimuth"]
                self.viewer.cam.elevation = self.camera_config["elevation"]

        # temporary fix for "M" key conflict
        self.viewer.opt.flags[-5] = 0
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
