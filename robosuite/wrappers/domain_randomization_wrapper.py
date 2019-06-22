"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.
"""

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, MaterialModder


class DRWrapper(Wrapper):
    env = None

    def __init__(self, env):
        super().__init__(env)
        # TODO: Move material/texture implementation to objects.py
        cube_geom = self.env.model.worldbody.findall("./body/[@name='cube']/geom")[0]
        cube_geom.set('material', 'arm_mat')
        self.modded_xml = self.env.model.get_xml()
        self.reset()

    def reset(self):
        self.env.reset_from_xml_string(self.modded_xml)

        # modder has to access updated model
        self.tex_modder = TextureModder(self.env.sim)
        self.light_modder = LightingModder(self.env.sim)
        self.mat_modder = MaterialModder(self.env.sim)

        for modder in (self.tex_modder, self.light_modder, self.mat_modder):
            modder.randomize()

    def render(self, **kwargs):
        super().render(**kwargs)
