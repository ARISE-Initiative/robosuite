from mujoco import viewer


DEFAULT_CAM = [
    [2.25, -1, 1.05312667],
    5,
    89.70301806083651,
    -18.02177994296577
]




class MjviewerRenderer:
    def __init__(self, env, camera_id=None):
        self.env = env
        self.camera_id = camera_id
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

            camera_config = DEFAULT_CAM

            self.viewer.cam.lookat = camera_config[0]
            self.viewer.cam.distance = camera_config[1]
            self.viewer.cam.azimuth = camera_config[2]
            self.viewer.cam.elevation = camera_config[3]

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
        # for finding a suitable default cam position
        # print(
        #     self.viewer.cam.lookat, 
        #     self.viewer.cam.distance, 
        #     self.viewer.cam.azimuth, 
        #     self.viewer.cam.elevation
        # )

        self.sim = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback
