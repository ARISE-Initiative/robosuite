"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.
"""

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder


class DRWrapper(Wrapper):
    env = None

    def __init__(self, env):
        super().__init__(env)
        # TODO: Do not hardcode SawyerLift env update
        cube_geom = self.env.model.worldbody.findall("./body/[@name='cube']/geom")[0]
        cube_geom.set('material', 'arm_mat')
        self.modded_xml = self.env.model.get_xml()
        self.reset()

    def reset(self):
        self.env.reset_from_xml_string(self.modded_xml)
        self.modder = TextureModder(self.env.sim)  # modder has to access updated model
        self.modder.randomize()

    def render(self, **kwargs):
        self.modder.randomize()
        super().render(**kwargs)

