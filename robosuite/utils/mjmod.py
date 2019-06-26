import copy
import numpy as np
from mujoco_py import modder


class PhysicalParameterModder:
    """
    Modder for various physical parameters of the mujoco model

    can use to modify parameters stored in MjModel (ie friction, damping, etc.) as
    well as optimizer parameters like global friction multipliers (eg solimp, solref, etc)

    To modify a parameteter, use the parameter to be changed as a keyword argument to
    self.mod and the new value as the value for that argument. Supports arbitray many
    modifications in a single step.

    NOTE: It is necesary to perform sim.forward after performing the moddification. 
    NOTE: Some parameters might not be able to be changed. users are to verify that
          after the call to forward that the parameter is indeed changed.
    """
    def __init__(self, sim):
        self.sim = sim

    @property
    def model(self):
        return self.sim.model

    @property
    def opt(self):
        return self.sim.model.opt

    def __getattr__(self, name):
        try:
            opt_attr = getattr(self.opt, name)
        except AttributeError:
            opt_attr = None
            
        try:
            model_attr = getattr(self.model, name)
        except AttributeError:
            model_attr = None
                    
        ret = opt_attr if opt_attr is not None else model_attr
        if callable(ret):
            def r(*args):
                return ret(*args)
            return r

        return ret

    def mod(self, **kwargs):
        """
        Method to actually mod. Assumes passing in keword arguments with key being the parameter to
        modify and the value being the value to set

        Feel free to add more as we see fit.
        """
        for to_mod in kwargs:
            val = kwargs[to_mod]

            param = to_mod
            ind = None
            if 'geom_friction' in param:
                joint = param.replace('_geom_friction', '')
                ind = self.geom_name2id(joint)
                param = 'geom_friction'
            elif 'dof_damping' in param:
                joint = param.replace('_dof_damping', '')
                param = 'dof_damping'
                joint = self.joint_name2id(joint)
                ind = np.zeros(self.nv)
                
                for i in range(self.model.nv):
                    if self.dof_jntid[i] == joint:
                        ind[i] = 1

            if ind is None:
                setattr(self, param, val)
            else:
                self.__getattr__(param)[ind] = val


class CameraModder(modder.CameraModder):
    """
    For randomization of camera locations

    TODO: add support for changing orientation or simply define the camera to look at a
          site in the location where the camera gaze is to be located.
    """
    def __init__(self, random_state=None, sim=None, \
                 pos_ranges=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)],
                 axis=[0, 0, 1], angle_range=(-0.25, 0.25), \
                 camera_name=None, seed=None):

        assert camera_name is not None

        if seed is not None: np.random.seed(seed)

        super().__init__(sim, random_state=random_state)
        
        self.base_pos = copy.copy(self.get_pos(camera_name))
        self.pos_ranges = pos_ranges
        self.camera_name = camera_name
        self.axis = axis
        self.angle_range = angle_range


    def randomize(self):
        # TODO:
        #  Use axis angle to change the rotation of the camera or use quats.
        #  Randomize field of view of camera
        delta_pos = np.stack([np.random.uniform(*self.pos_ranges[i]) for i in range(3)])
        self.set_pos(self.camera_name, self.base_pos + delta_pos)

    def whiten_materials(self, *args, **kargs):
        pass


class MaterialModder(modder.MaterialModder):
    """
    Extension of the MaterialModder in MujocoPy to support randomization of all materials in a model
    """
    def __init__(self, sim, seed=None):
        if seed is not None: np.random.seed(seed)
        super().__init__(sim, random_state=seed)

    def randomize(self):
        # randomize all the materials ie properties like reflectance
        for geom in self.model.geom_names:
            self.rand_all(geom)

    def whiten_materials(self, *args, **kargs):
        pass


class LightingModder(modder.LightModder):
    """
    Extension of the LightModder in MujocoPy to support randomization of all the lights in a model
    """
    def __init__(self, sim, seed=None):
        if seed is not None: np.random.seed(seed)
        super().__init__(sim, random_state=seed)

    def randomize(self):
        # randomize all the lights
        for light in self.model.light_names:
            self.rand_all(light)

    def rand3(self):
        return np.random.randn(3)

    def randbool(self):
        rand = np.random.choice([0,1], 1)
        rand = rand[0]
        return rand

    def rand_all(self, light):
        self.set_pos(light, self.rand3())
        self.set_dir(light, self.rand3())
        self.set_active(light, self.randbool())
        self.set_specular(light, self.rand3())
        self.set_ambient(light, self.rand3())
        self.set_diffuse(light, self.rand3())
        self.set_castshadow(light, self.randbool())
        
    def whiten_materials(self, *args, **kargs):
        pass


class TextureModder(modder.TextureModder):
    """
    Extension of the TextureModder in MujocoPy
    """
    def __init__(self, sim, seed=None):
        if seed is not None: np.random.seed(seed)
        super().__init__(sim, random_state=seed)

    def whiten_materials(self, geom_names=None):
        """
        Extends modder.TextureModder to also whiten geom_rgba

        Args:
        - geom_names (list): list of geom names whose materials should be
            set to white. If omitted, all materials will be changed.
        """
        geom_names = geom_names or []
        if geom_names:
            for name in geom_names:
                geom_id = self.model.geom_name2id(name)
                mat_id = self.model.geom_matid[geom_id]
                self.model.mat_rgba[mat_id, :] = 1.0
                self.model.geom_rgba[geom_id, :] = 1.0
        else:
            self.model.mat_rgba[:] = 1.0
            self.model.geom_rgba[:] = 1.0

    def randomize(self):
        self.whiten_materials()
        super().randomize()
        super().rand_all("skybox")
