from mujoco import viewer
from robocasa.environments.manipulation.kitchen import Kitchen

DEFAULT_CAM = [[2.25, -1, 1.05312667], 5, 89.70301806083651, -18.02177994296577]

# default free cameras for different kitchen layouts
KITCHEN_CAMS = {
    0: [[2.26593463, -1.00037131, 1.38769295], 3.0505089839567323, 90.71563812375285, -12.63948837207208],
    1: [[2.66147999, -1.00162429, 1.2425155], 3.7958766287746255, 89.75784013699234, -15.177406642875091],
    2: [[3.02344359, -1.48874618, 1.2412914], 3.6684844368165512, 51.67880851867874, -13.302619131542388],
    3: [[1.44842548, -1.47664723, 1.24115989], 3.923271794728187, 127.12928449329333, -16.495686334624907],
    4: [[1.6, -1.0, 1.0], 5, 89.70301806083651, -18.02177994296577],
}


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

            # orient camera
            # adjust free camera
            if isinstance(self.env, Kitchen):
                if self.env.layout_id in KITCHEN_CAMS:
                    camera_config = KITCHEN_CAMS[self.env.layout_id]
                else:
                    # print("Cannot find default free camera for layout, using default")
                    camera_config = DEFAULT_CAM
            else:
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
