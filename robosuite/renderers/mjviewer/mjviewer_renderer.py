from mujoco import viewer

DEFAULT_CAM = dict(lookat=[2.25, -1, 1.05312667], distance=5, azimuth=89.70301806083651, elevation=-18.02177994296577)


class MjviewerRenderer:
    def __init__(self, env, camera_id=None, camera_config=None):
        self.env = env
        self.camera_id = camera_id
        if camera_config is None or camera_config == {}:
            self.camera_config = DEFAULT_CAM
        else:
            self.camera_config = camera_config
        self.viewer = None

    def render(self):
        # no need to do anything for mujoco viewer
        pass

    def update(self):
        if self.viewer is None:
            # creates passive mujoco viewer
            self.viewer = viewer.launch_passive(
                self.env.sim.model._model,
                self.env.sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )

            # disables geom group 0
            self.viewer.opt.geomgroup[0] = 0

            self.viewer.cam.lookat = self.camera_config["lookat"]
            self.viewer.cam.distance = self.camera_config["distance"]
            self.viewer.cam.azimuth = self.camera_config["azimuth"]
            self.viewer.cam.elevation = self.camera_config["elevation"]

            if self.camera_id is not None:
                # set camera type to fixed
                self.viewer.cam.type = 2
                # robot0_agentview
                self.viewer.cam.fixedcamid = self.camera_id

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
