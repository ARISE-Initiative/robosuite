from robosuite.utils import MujocoPyRenderer

from robosuite.renderers.base import Renderer

class MujocoRenderer(Renderer):
    def __init__(self,
                 sim,
                 render_camera="frontview",
                 render_collision_mesh=False,
                 render_visual_mesh=True):
        
        self.sim = sim        
        self.render_camera = render_camera
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        
        self.viewer = MujocoPyRenderer(self.sim)
        self.viewer.viewer.vopt.geomgroup[0] = (1 if self.render_collision_mesh else 0)
        self.viewer.viewer.vopt.geomgroup[1] = (1 if self.render_visual_mesh else 0)

        # hiding the overlay speeds up rendering significantly
        self.viewer.viewer._hide_overlay = True

        # make sure mujoco-py doesn't block rendering frames
        # (see https://github.com/StanfordVL/robosuite/issues/39)
        self.viewer.viewer._render_every_frame = True

        # Set the camera angle for viewing
        if self.render_camera is not None:
            self.viewer.set_camera(camera_id=self.sim.model.camera_name2id(self.render_camera))        

    def reset(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.initialize_renderer()

    def update(self):
        pass

    def update_with_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()

    def render(self):
        self.viewer.render()

    def close(self):
        self.viewer.close()
        self.viewer = None