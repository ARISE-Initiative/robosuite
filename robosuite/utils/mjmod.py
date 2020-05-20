"""
Modder classes used for domain randomization. Largely bsased off of the mujoco-py
implementation below.

https://github.com/openai/mujoco-py/blob/1fe312b09ae7365f0dd9d4d0e453f8da59fae0bf/mujoco_py/modder.py
"""

import os
import numpy as np

from collections import defaultdict
from PIL import Image
from mujoco_py import cymj

import robosuite
import robosuite.utils.transform_utils as trans

class BaseModder():
    def __init__(self, sim):
        self.sim = sim

    @property
    def model(self):
        # Available for quick convenience access
        return self.sim.model

class LightingModder(BaseModder):

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

    def set_pos(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_pos[lightid] = value

    def set_dir(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_dir[lightid] = value

    def set_active(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        self.model.light_active[lightid] = value

    def set_specular(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_specular[lightid] = value

    def set_ambient(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_ambient[lightid] = value

    def set_diffuse(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_diffuse[lightid] = value

    def set_castshadow(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name
        self.model.light_castshadow[lightid] = value

    def get_lightid(self, name):
        return self.model.light_name2id(name)

    def randomize(self):
        # randomize all the lights
        for light in self.model.light_names:
            self.rand_all(light)


class CameraModder(BaseModder):
    def __init__(
        self,
        sim,
        camera_names,
        perturb_position=True,
        perturb_rotation=True,
        perturb_fovy=True,
        position_perturbation_size=0.01,
        rotation_perturbation_size=0.087,
        fovy_perturbation_size=5.,
    ):
        """
        Args:
            sim (MjSim): MjSim object
            camera_names ([string]): list of camera names to use for randomization
            perturb_position (bool): if True, randomize camera position
            perturb_rotation (bool): if True, randomize camera rotation
            perturb_fovy (bool): if True, randomize camera fovy
            position_perturbation_size (float): size of camera position perturbations to each dimension
            rotation_perturbation_size (float): magnitude of camera rotation perturbations in axis-angle.
                Default corresponds to around 5 degrees.
            fovy_perturbation_size (float): magnitude of camera fovy perturbations (corresponds to focusing)
        """
        super().__init__(sim)

        assert perturb_position or perturb_rotation or perturb_fovy

        self.camera_names = camera_names
        self.perturb_position = perturb_position
        self.perturb_rotation = perturb_rotation
        self.perturb_fovy = perturb_fovy
        self.position_perturbation_size = position_perturbation_size
        self.rotation_perturbation_size = rotation_perturbation_size
        self.fovy_perturbation_size = fovy_perturbation_size
        self.set_defaults()

    def set_defaults(self):
        """
        Uses the current MjSim state and model to save default parameter values. 
        """
        self.base_pos = {}
        self.base_quat = {}
        self.base_fovy = {}
        for camera_name in self.camera_names:
            self.base_pos[camera_name] = np.array(self.get_pos(camera_name))
            self.base_quat[camera_name] = np.array(self.get_quat(camera_name))
            self.base_fovy[camera_name] = self.get_fovy(camera_name)

    def restore_defaults(self):
        """
        Reloads the saved parameter values.
        """
        for camera_name in self.camera_names:
            self.set_pos(camera_name, self.base_pos[camera_name])
            self.set_quat(camera_name, self.base_quat[camera_name])
            self.set_fovy(camera_name, self.base_fovy[camera_name])

    def randomize(self):
        for camera_name in self.camera_names:

            if self.perturb_position:
                delta_pos = np.random.uniform(
                    low=-self.position_perturbation_size, 
                    high=self.position_perturbation_size, 
                    size=3,
                )
                self.set_pos(
                    camera_name, 
                    self.base_pos[camera_name] + delta_pos,
                )

            if self.perturb_rotation:
                # sample random axis using a normalized sample from spherical Gaussian.
                # see (http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/)
                # for why it works.
                random_axis = np.random.randn(3)
                random_axis /= np.linalg.norm(random_axis)
                random_angle = np.random.uniform(low=0., high=self.rotation_perturbation_size)
                random_delta_rot = trans.quat2mat(trans.axisangle2quat(axis=random_axis, angle=random_angle))
                
                # compute new rotation and set it
                base_rot = trans.quat2mat(trans.convert_quat(self.base_quat[camera_name], to='xyzw'))
                new_rot = random_delta_rot.T.dot(base_rot)
                new_quat = trans.convert_quat(trans.mat2quat(new_rot), to='wxyz')
                self.set_quat(
                    camera_name,
                    new_quat,
                    # self.base_quat[camera_name]
                )

            if self.perturb_fovy:
                delta_fovy = np.random.uniform(
                    low=-self.fovy_perturbation_size,
                    high=self.fovy_perturbation_size,
                )
                self.set_fovy(
                    camera_name,
                    self.base_fovy[camera_name] + delta_fovy,
                )

    def get_fovy(self, name):
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        return self.model.cam_fovy[camid]

    def set_fovy(self, name, value):
        camid = self.get_camid(name)
        assert 0 < value < 180
        assert camid > -1, "Unknown camera %s" % name
        self.model.cam_fovy[camid] = value

    def get_quat(self, name):
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        return self.model.cam_quat[camid]

    def set_quat(self, name, value):
        value = list(value)
        assert len(value) == 4, (
            "Expectd value of length 3, instead got %s" % value)
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        self.model.cam_quat[camid] = value

    def get_pos(self, name):
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        return self.model.cam_pos[camid]

    def set_pos(self, name, value):
        value = list(value)
        assert len(value) == 3, (
            "Expected value of length 3, instead got %s" % value)
        camid = self.get_camid(name)
        assert camid > -1
        self.model.cam_pos[camid] = value

    def get_camid(self, name):
        return self.model.camera_name2id(name)


class TextureModder(BaseModder):
    """
    Modify textures in model. Example use:
        sim = MjSim(...)
        modder = TextureModder(sim)
        modder.whiten_materials()  # ensures materials won't impact colors
        modder.set_checker('some_geom', (255, 0, 0), (0, 0, 0))
        modder.rand_all('another_geom')
    Note: in order for the textures to take full effect, you'll need to set
    the rgba values for all materials to [1, 1, 1, 1], otherwise the texture
    colors will be modulated by the material colors. Call the
    `whiten_materials` helper method to set all material colors to white.
    """

    def __init__(self, sim, path=None):
        super().__init__(sim)

        self.path = path

        self.textures = [Texture(self.model, i)
                         for i in range(self.model.ntex)]
        self._build_tex_geom_map()

        # These matrices will be used to rapidly synthesize
        # checker pattern bitmaps
        self._cache_checker_matrices()

    def randomize(self):
        """
        Overrides mujoco-py implementation to also randomize color
        for geoms that have no material.
        """
        self.whiten_materials()
        for name in self.sim.model.geom_names:
            try:
                self.rand_all(name)
            except:
                self.model.geom_rgba[self.model.geom_name2id(name), :3] = np.random.uniform(0, 1, size=3)
        self.rand_all("skybox")

    def rand_all(self, name):
        choices = [
            self.rand_checker,
            self.rand_gradient,
            self.rand_rgb,
            self.rand_noise,
        ]
        choice = np.random.randint(len(choices))
        return choices[choice](name)

    def rand_checker(self, name):
        rgb1, rgb2 = self.get_rand_rgb(2)
        return self.set_checker(name, rgb1, rgb2)

    def rand_gradient(self, name):
        rgb1, rgb2 = self.get_rand_rgb(2)
        vertical = bool(np.random.uniform() > 0.5)
        return self.set_gradient(name, rgb1, rgb2, vertical=vertical)

    def rand_rgb(self, name):
        rgb = self.get_rand_rgb()
        return self.set_rgb(name, rgb)

    def rand_noise(self, name):
        fraction = 0.1 + np.random.uniform() * 0.8
        rgb1, rgb2 = self.get_rand_rgb(2)
        return self.set_noise(name, rgb1, rgb2, fraction)

    def whiten_materials(self, geom_names=None):
        """
        Extends modder.TextureModder to also whiten geom_rgba.

        Helper method for setting all material colors to white, otherwise
        the texture modifications won't take full effect.
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
        else:
            self.model.mat_rgba[:] = 1.0

    def get_rand_rgb(self, n=1):
        def _rand_rgb():
            return np.array(np.random.uniform(size=3) * 255,
                            dtype=np.uint8)

        if n == 1:
            return _rand_rgb()
        else:
            return tuple(_rand_rgb() for _ in range(n))

    def set_existing_texture(self, name):
        bitmap = self.get_texture(name).bitmap
        img = Image.new("RGB", (bitmap.shape[1], bitmap.shape[0]))

        img_path = self.path + '/' + np.random.choice(os.listdir(self.path))
        img = Image.open(img_path).convert('RGB')
        img = img.convert('RGB')
        if name == 'skybox':
            img = img.resize((256, 256))
        elif name =='table_visual':
            img = img.resize((512, 512))
        else:
            img = img.resize((32, 32))
        img = np.array(img)
        img = img.astype(np.uint8)
        img = np.concatenate([img] * int(bitmap.shape[0] / img.shape[0]), 0)
        img.resize(bitmap.shape)
        bitmap[..., :] = img

        self.upload_texture(name)

    def get_texture(self, name):
        if name == 'skybox':
            tex_id = -1
            for i in range(self.model.ntex):
                # TODO: Don't hardcode this
                skybox_textype = 2
                if self.model.tex_type[i] == skybox_textype:
                    tex_id = i
            assert tex_id >= 0, "Model has no skybox"
        else:
            geom_id = self.model.geom_name2id(name)
            mat_id = self.model.geom_matid[geom_id]
            assert mat_id >= 0, "Geom has no assigned material"
            tex_id = self.model.mat_texid[mat_id]
            assert tex_id >= 0, "Material has no assigned texture"

        texture = self.textures[tex_id]

        return texture

    def get_checker_matrices(self, name):
        if name == 'skybox':
            return self._skybox_checker_mat
        else:
            geom_id = self.model.geom_name2id(name)
            return self._geom_checker_mats[geom_id]

    def set_checker(self, name, rgb1, rgb2):
        bitmap = self.get_texture(name).bitmap
        cbd1, cbd2 = self.get_checker_matrices(name)

        rgb1 = np.asarray(rgb1).reshape([1, 1, -1])
        rgb2 = np.asarray(rgb2).reshape([1, 1, -1])
        bitmap[:] = rgb1 * cbd1 + rgb2 * cbd2

        self.upload_texture(name)
        return bitmap

    def set_gradient(self, name, rgb1, rgb2, vertical=True):
        """
        Creates a linear gradient from rgb1 to rgb2.
        Args:
        - rgb1 (array): start color
        - rgb2 (array): end color
        - vertical (bool): if True, the gradient in the positive
            y-direction, if False it's in the positive x-direction.
        """
        # NOTE: MuJoCo's gradient uses a sigmoid. Here we simplify
        # and just use a linear gradient... We could change this
        # to just use a tanh-sigmoid if needed.
        bitmap = self.get_texture(name).bitmap
        h, w = bitmap.shape[:2]
        if vertical:
            p = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
        else:
            p = np.tile(np.linspace(0, 1, w), (h, 1))

        for i in range(3):
            bitmap[..., i] = rgb2[i] * p + rgb1[i] * (1.0 - p)

        self.upload_texture(name)
        return bitmap

    def set_rgb(self, name, rgb):
        bitmap = self.get_texture(name).bitmap
        bitmap[..., :] = np.asarray(rgb)

        self.upload_texture(name)
        return bitmap

    def set_noise(self, name, rgb1, rgb2, fraction=0.9):
        """
        Args:
        - name (str): name of geom
        - rgb1 (array): background color
        - rgb2 (array): color of random noise foreground color
        - fraction (float): fraction of pixels with foreground color
        """
        bitmap = self.get_texture(name).bitmap
        h, w = bitmap.shape[:2]
        mask = np.random.uniform(size=(h, w)) < fraction

        bitmap[..., :] = np.asarray(rgb1)
        bitmap[mask, :] = np.asarray(rgb2)

        self.upload_texture(name)
        return bitmap

    def upload_texture(self, name):
        """
        Uploads the texture to the GPU so it's available in the rendering.
        """
        texture = self.get_texture(name)
        if not self.sim.render_contexts:
            cymj.MjRenderContextOffscreen(self.sim)
        for render_context in self.sim.render_contexts:
            render_context.upload_texture(texture.id)

    def _build_tex_geom_map(self):
        # Build a map from tex_id to geom_ids, so we can check
        # for collisions.
        self._geom_ids_by_tex_id = defaultdict(list)
        for geom_id in range(self.model.ngeom):
            mat_id = self.model.geom_matid[geom_id]
            if mat_id >= 0:
                tex_id = self.model.mat_texid[mat_id]
                if tex_id >= 0:
                    self._geom_ids_by_tex_id[tex_id].append(geom_id)

    def _cache_checker_matrices(self):
        """
        Cache two matrices of the form [[1, 0, 1, ...],
                                        [0, 1, 0, ...],
                                        ...]
        and                            [[0, 1, 0, ...],
                                        [1, 0, 1, ...],
                                        ...]
        for each texture. To use for fast creation of checkerboard patterns
        """
        self._geom_checker_mats = []
        for geom_id in range(self.model.ngeom):
            mat_id = self.model.geom_matid[geom_id]
            tex_id = self.model.mat_texid[mat_id]
            texture = self.textures[tex_id]
            h, w = texture.bitmap.shape[:2]
            self._geom_checker_mats.append(self._make_checker_matrices(h, w))

        # add skybox
        skybox_tex_id = -1
        for tex_id in range(self.model.ntex):
            skybox_textype = 2
            if self.model.tex_type[tex_id] == skybox_textype:
                skybox_tex_id = tex_id
        if skybox_tex_id >= 0:
            texture = self.textures[skybox_tex_id]
            h, w = texture.bitmap.shape[:2]
            self._skybox_checker_mat = self._make_checker_matrices(h, w)
        else:
            self._skybox_checker_mat = None

    def _make_checker_matrices(self, h, w):
        re = np.r_[((w + 1) // 2) * [0, 1]]
        ro = np.r_[((w + 1) // 2) * [1, 0]]
        cbd1 = np.expand_dims(np.row_stack(((h + 1) // 2) * [re, ro]), -1)[:h, :w]
        cbd2 = np.expand_dims(np.row_stack(((h + 1) // 2) * [ro, re]), -1)[:h, :w]
        return cbd1, cbd2


# From mjtTexture
MJT_TEXTURE_ENUM = ['2d', 'cube', 'skybox']

class Texture():
    """
    Helper class for operating on the MuJoCo textures.
    """

    __slots__ = ['id', 'type', 'height', 'width', 'tex_adr', 'tex_rgb']

    def __init__(self, model, tex_id):
        self.id = tex_id
        self.type = MJT_TEXTURE_ENUM[model.tex_type[tex_id]]
        self.height = model.tex_height[tex_id]
        self.width = model.tex_width[tex_id]
        self.tex_adr = model.tex_adr[tex_id]
        self.tex_rgb = model.tex_rgb

    @property
    def bitmap(self):
        size = self.height * self.width * 3
        data = self.tex_rgb[self.tex_adr:self.tex_adr + size]
        return data.reshape((self.height, self.width, 3))

class PhysicalParameterModder:
    """
    Modder for various physical parameters of the mujoco model
    can use to modify parameters stored in MjModel (ie friction, damping, etc.) as
    well as optimizer parameters like global friction multipliers (eg solimp, solref, etc)
    To modify a parameteter, use the parameter to be changed as a keyword argument to
    self.mod and the new value as the value for that argument. Supports arbitray many
    modifications in a single step.
    NOTE: It is necesary to perform sim.forward after performing the modification.
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
        Method to actually mod. Assumes passing in keyword arguments with key being the parameter to
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
