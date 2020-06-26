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
    def __init__(self, sim, random_state=None):
        """
        Using @random_state ensures that sampling here won't be affected
        by sampling that happens outside of the modders.

        Args:
            sim (MjSim): MjSim object

            random_state (RandomState): instance of np.random.RandomState
        """
        self.sim = sim
        if random_state is None:
            # default to global RandomState instance
            self.random_state = np.random.mtrand._rand
        else:
            self.random_state = random_state

    @property
    def model(self):
        # Available for quick convenience access
        return self.sim.model

class LightingModder(BaseModder):

    def __init__(
        self,
        sim,
        random_state=None,
        light_names=None,
        randomize_position=True,
        randomize_direction=True,
        randomize_specular=True,
        randomize_ambient=True,
        randomize_diffuse=True,
        randomize_active=True,
        position_perturbation_size=0.1,
        direction_perturbation_size=0.35, # 20 degrees
        specular_perturbation_size=0.1,
        ambient_perturbation_size=0.1,
        diffuse_perturbation_size=0.1,
    ):
        """
        Args:
            sim (MjSim): MjSim object

            random_state (RandomState): instance of np.random.RandomState

            light_names ([string]): list of lights to use for randomization. If not provided, all
                lights in the model are randomized.
        """
        super().__init__(sim, random_state=random_state)

        if light_names is None:
            light_names = self.sim.model.light_names
        self.light_names = light_names

        self.randomize_position = randomize_position
        self.randomize_direction = randomize_direction
        self.randomize_specular = randomize_specular
        self.randomize_ambient = randomize_ambient
        self.randomize_diffuse = randomize_diffuse
        self.randomize_active = randomize_active

        self.position_perturbation_size = position_perturbation_size
        self.direction_perturbation_size = direction_perturbation_size
        self.specular_perturbation_size = specular_perturbation_size
        self.ambient_perturbation_size = ambient_perturbation_size
        self.diffuse_perturbation_size = diffuse_perturbation_size

        self.save_defaults()

    def save_defaults(self):
        """
        Uses the current MjSim state and model to save default parameter values. 
        """
        self._defaults = { k : {} for k in self.light_names }
        for name in self.light_names:
            self._defaults[name]['pos'] = np.array(self.get_pos(name))
            self._defaults[name]['dir'] = np.array(self.get_dir(name))
            self._defaults[name]['specular'] = np.array(self.get_specular(name))
            self._defaults[name]['ambient'] = np.array(self.get_ambient(name))
            self._defaults[name]['diffuse'] = np.array(self.get_diffuse(name))
            self._defaults[name]['active'] = self.get_active(name)

    def restore_defaults(self):
        """
        Reloads the saved parameter values.
        """
        for name in self.light_names:
            self.set_pos(name, self._defaults[name]['pos'])
            self.set_dir(name, self._defaults[name]['dir'])
            self.set_specular(name, self._defaults[name]['specular'])
            self.set_ambient(name, self._defaults[name]['ambient'])
            self.set_diffuse(name, self._defaults[name]['diffuse'])
            self.set_active(name, self._defaults[name]['active'])

    def randomize(self):
        for name in self.light_names:
            if self.randomize_position:
                self._randomize_position(name)

            if self.randomize_direction:
                self._randomize_direction(name)

            if self.randomize_specular:
                self._randomize_specular(name)

            if self.randomize_ambient:
                self._randomize_ambient(name)

            if self.randomize_diffuse:
                self._randomize_diffuse(name)

            if self.randomize_active:
                self._randomize_active(name)

    def _randomize_position(self, name):
        delta_pos = self.random_state.uniform(
            low=-self.position_perturbation_size, 
            high=self.position_perturbation_size, 
            size=3,
        )
        self.set_pos(
            name, 
            self._defaults[name]['pos'] + delta_pos,
        )

    def _randomize_direction(self, name):
        # sample a small, random axis-angle delta rotation
        random_axis, random_angle = trans.random_axis_angle(angle_limit=self.direction_perturbation_size, random_state=self.random_state)
        random_delta_rot = trans.quat2mat(trans.axisangle2quat(axis=random_axis, angle=random_angle))
        
        # rotate direction by this delta rotation and set the new direction
        new_dir = random_delta_rot.dot(self._defaults[name]['dir'])
        self.set_dir(
            name,
            new_dir,
        )

    def _randomize_specular(self, name):
        delta = self.random_state.uniform(
            low=-self.specular_perturbation_size, 
            high=self.specular_perturbation_size, 
            size=3,
        )
        self.set_specular(
            name, 
            self._defaults[name]['specular'] + delta,
        )

    def _randomize_ambient(self, name):
        delta = self.random_state.uniform(
            low=-self.ambient_perturbation_size, 
            high=self.ambient_perturbation_size, 
            size=3,
        )
        self.set_ambient(
            name, 
            self._defaults[name]['ambient'] + delta,
        )

    def _randomize_diffuse(self, name):
        delta = self.random_state.uniform(
            low=-self.diffuse_perturbation_size, 
            high=self.diffuse_perturbation_size, 
            size=3,
        )
        self.set_diffuse(
            name, 
            self._defaults[name]['diffuse'] + delta,
        )

    def _randomize_active(self, name):
        active = int(self.random_state.uniform() > 0.5)
        self.set_active(
            name,
            active
        )

    def get_pos(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_pos[lightid]

    def set_pos(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_pos[lightid] = value

    def get_dir(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_dir[lightid] 

    def set_dir(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_dir[lightid] = value

    def get_active(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_active[lightid]

    def set_active(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        self.model.light_active[lightid] = value

    def get_specular(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_specular[lightid]

    def set_specular(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_specular[lightid] = value

    def get_ambient(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_ambient[lightid]

    def set_ambient(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_ambient[lightid] = value

    def get_diffuse(self, name):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_diffuse[lightid]

    def set_diffuse(self, name, value):
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_diffuse[lightid] = value

    # def set_castshadow(self, name, value):
    #     lightid = self.get_lightid(name)
    #     assert lightid > -1, "Unkwnown light %s" % name
    #     self.model.light_castshadow[lightid] = value

    def get_lightid(self, name):
        return self.model.light_name2id(name)

class CameraModder(BaseModder):
    def __init__(
        self,
        sim,
        random_state=None,
        camera_names=None,
        randomize_position=True,
        randomize_rotation=True,
        randomize_fovy=True,
        position_perturbation_size=0.01,
        rotation_perturbation_size=0.087,
        fovy_perturbation_size=5.,
    ):
        """
        Args:
            sim (MjSim): MjSim object
            random_state (RandomState): instance of np.random.RandomState
            camera_names ([string]): list of camera names to use for randomization. If not provided,
                all cameras are used for randomization.
            perturb_position (bool): if True, randomize camera position
            perturb_rotation (bool): if True, randomize camera rotation
            perturb_fovy (bool): if True, randomize camera fovy
            position_perturbation_size (float): size of camera position perturbations to each dimension
            rotation_perturbation_size (float): magnitude of camera rotation perturbations in axis-angle.
                Default corresponds to around 5 degrees.
            fovy_perturbation_size (float): magnitude of camera fovy perturbations (corresponds to focusing)
        """
        super().__init__(sim, random_state=random_state)

        assert randomize_position or randomize_rotation or randomize_fovy

        if camera_names is None:
            camera_names = self.sim.model.camera_names
        self.camera_names = camera_names

        self.randomize_position = randomize_position
        self.randomize_rotation = randomize_rotation
        self.randomize_fovy = randomize_fovy

        self.position_perturbation_size = position_perturbation_size
        self.rotation_perturbation_size = rotation_perturbation_size
        self.fovy_perturbation_size = fovy_perturbation_size

        self.save_defaults()

    def save_defaults(self):
        """
        Uses the current MjSim state and model to save default parameter values. 
        """
        self._defaults = { k : {} for k in self.camera_names }
        for camera_name in self.camera_names:
            self._defaults[camera_name]['pos'] = np.array(self.get_pos(camera_name))
            self._defaults[camera_name]['quat'] = np.array(self.get_quat(camera_name))
            self._defaults[camera_name]['fovy'] = self.get_fovy(camera_name)

    def restore_defaults(self):
        """
        Reloads the saved parameter values.
        """
        for camera_name in self.camera_names:
            self.set_pos(camera_name, self._defaults[camera_name]['pos'])
            self.set_quat(camera_name, self._defaults[camera_name]['quat'])
            self.set_fovy(camera_name, self._defaults[camera_name]['fovy'])

    def randomize(self):
        for camera_name in self.camera_names:
            if self.randomize_position:
                self._randomize_position(camera_name)

            if self.randomize_rotation:
                self._randomize_rotation(camera_name)

            if self.randomize_fovy:
                self._randomize_fovy(camera_name)

    def _randomize_position(self, name):
        delta_pos = self.random_state.uniform(
            low=-self.position_perturbation_size, 
            high=self.position_perturbation_size, 
            size=3,
        )
        self.set_pos(
            name, 
            self._defaults[name]['pos'] + delta_pos,
        )

    def _randomize_rotation(self, name):
        # sample a small, random axis-angle delta rotation
        random_axis, random_angle = trans.random_axis_angle(angle_limit=self.rotation_perturbation_size, random_state=self.random_state)
        random_delta_rot = trans.quat2mat(trans.axisangle2quat(axis=random_axis, angle=random_angle))
        
        # compute new rotation and set it
        base_rot = trans.quat2mat(trans.convert_quat(self._defaults[name]['quat'], to='xyzw'))
        new_rot = random_delta_rot.T.dot(base_rot)
        new_quat = trans.convert_quat(trans.mat2quat(new_rot), to='wxyz')
        self.set_quat(
            name,
            new_quat,
        )

    def _randomize_fovy(self, name):
        delta_fovy = self.random_state.uniform(
            low=-self.fovy_perturbation_size,
            high=self.fovy_perturbation_size,
        )
        self.set_fovy(
            name,
            self._defaults[name]['fovy'] + delta_fovy,
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

    def __init__(
        self, 
        sim,
        random_state=None,
        geom_names=None,
        randomize_local=False,
        randomize_material=False,
        local_rgb_interpolation=0.1,
        local_material_interpolation=0.2,
        texture_variations=['rgb', 'checker', 'noise', 'gradient'],
        randomize_skybox=True,
        # texture_path=None,
    ):
        """
        Args:
            sim (MjSim): MjSim object

            random_state (RandomState): instance of np.random.RandomState

            geom_names ([string]): list of geom names to use for randomization. If not provided,
                all geoms are used for randomization.

            randomize_local (bool): if True, constrain RGB color variations to be close to the
                original RGB colors per geom and texture. Otherwise, RGB color values will
                be sampled uniformly at random.

            randomize_material (bool): if True, randomizes material properties associated with a
                given texture (reflectance, shininess, specular)

            local_rgb_interpolation (float): determines the size of color variations from
                the base geom colors when @randomize_local is True.

            local_material_interpolation (float): determines the size of material variations from
                the base material when @randomize_local and @randomize_material are both True.

            texture_variations ([string]): a list of texture variation strings. Each string
                must be either 'rgb', 'checker', 'noise', or 'gradient' and corresponds to
                a specific kind of texture randomization. For each geom that has a material
                and texture, a random variation from this list is sampled and applied.

            randomize_skybox (bool): if True, apply texture variations to the skybox as well.

            texture_path (string): optional path to texture images files to use for random texture
                generation
        """
        super().__init__(sim, random_state=random_state)

        if geom_names is None:
            geom_names = self.sim.model.geom_names
        self.geom_names = geom_names

        self.randomize_local = randomize_local
        self.randomize_material = randomize_material
        self.local_rgb_interpolation = local_rgb_interpolation
        self.local_material_interpolation = local_material_interpolation
        self.texture_variations = list(texture_variations)
        self.randomize_skybox = randomize_skybox

        self._all_texture_variation_callbacks = {
            'rgb' : self.rand_rgb,
            'checker' : self.rand_checker,
            'noise' : self.rand_noise,
            'gradient' : self.rand_gradient,
        }
        self._texture_variation_callbacks = { 
            k : self._all_texture_variation_callbacks[k] 
            for k in self.texture_variations 
        }

        self.save_defaults()

    def save_defaults(self):
        """
        Uses the current MjSim state and model to save default parameter values. 
        """
        self.textures = [Texture(self.model, i)
                         for i in range(self.model.ntex)]
        # self._build_tex_geom_map()

        # save copy of original texture bitmaps
        self._default_texture_bitmaps = [np.array(text.bitmap) for text in self.textures]

        # These matrices will be used to rapidly synthesize
        # checker pattern bitmaps
        self._cache_checker_matrices()

        self._defaults = { k : {} for k in self.geom_names }
        if self.randomize_skybox:
            self._defaults['skybox'] = {}
        for name in self.geom_names:
            if self._check_geom_for_texture(name):
                # store the texture bitmap for this geom
                tex_id = self._name_to_tex_id(name)
                self._defaults[name]['texture'] = self._default_texture_bitmaps[tex_id]
                # store material properties as well (in tuple (reflectance, shininess, specular) form)
                self._defaults[name]['material'] = self.get_material(name)
            else:
                # store geom color
                self._defaults[name]['rgb'] = np.array(self.get_geom_rgb(name))

        if self.randomize_skybox:
            tex_id = self._name_to_tex_id('skybox')
            self._defaults['skybox']['texture'] = self._default_texture_bitmaps[tex_id]

    def restore_defaults(self):
        """
        Reloads the saved parameter values.
        """
        for name in self.geom_names:
            if self._check_geom_for_texture(name):
                self.set_texture(name, self._defaults[name]['texture'], perturb=False)
                self.set_material(name, self._defaults[name]['material'], perturb=False)
            else:
                self.set_geom_rgb(name, self._defaults[name]['rgb'])

        if self.randomize_skybox:
            self.set_texture('skybox', self._defaults['skybox']['texture'], perturb=False)

    def randomize(self):
        """
        Overrides mujoco-py implementation to also randomize color
        for geoms that have no material.
        """
        self.whiten_materials()
        for name in self.geom_names:
            if self._check_geom_for_texture(name):
                # geom has valid texture that can be randomized
                self._randomize_texture(name)
                # randomize material if requested
                if self.randomize_material:
                    self._randomize_material(name)
            else:
                # randomize geom color
                self._randomize_geom_color(name)

        if self.randomize_skybox:
            self._randomize_texture("skybox")

    def _randomize_geom_color(self, name):
        if self.randomize_local:
            random_color = self.random_state.uniform(0, 1, size=3)
            rgb = (1. - self.local_rgb_interpolation) * self._defaults[name]['rgb'] + self.local_rgb_interpolation * random_color
        else:
            rgb = self.random_state.uniform(0, 1, size=3)
        self.set_geom_rgb(name, rgb)

    def _randomize_texture(self, name):
        keys = list(self._texture_variation_callbacks.keys())
        choice = keys[self.random_state.randint(len(keys))]
        self._texture_variation_callbacks[choice](name)

    def _randomize_material(self, name):
        # Return immediately if this is the skybox
        if name == 'skybox':
            return
        # Grab material id
        mat_id = self._name_to_mat_id(name)
        # Randomize reflectance, shininess, and specular
        material = self.random_state.uniform(0, 1, size=3)   # (reflectance, shininess, specular)
        self.set_material(name, material, perturb=self.randomize_local)

    def rand_checker(self, name):
        rgb1, rgb2 = self.get_rand_rgb(2)
        self.set_checker(name, rgb1, rgb2, perturb=self.randomize_local)

    def rand_gradient(self, name):
        rgb1, rgb2 = self.get_rand_rgb(2)
        vertical = bool(self.random_state.uniform() > 0.5)
        self.set_gradient(name, rgb1, rgb2, vertical=vertical, perturb=self.randomize_local)

    def rand_rgb(self, name):
        rgb = self.get_rand_rgb()
        self.set_rgb(name, rgb, perturb=self.randomize_local)

    def rand_noise(self, name):
        fraction = 0.1 + self.random_state.uniform() * 0.8
        rgb1, rgb2 = self.get_rand_rgb(2)
        self.set_noise(name, rgb1, rgb2, fraction, perturb=self.randomize_local)

    def whiten_materials(self):
        """
        Extends modder.TextureModder to also whiten geom_rgba

        Helper method for setting all material colors to white, otherwise
        the texture modifications won't take full effect.
        """
        for name in self.geom_names:
            # whiten geom
            geom_id = self.model.geom_name2id(name)
            self.model.geom_rgba[geom_id, :] = 1.0

            if self._check_geom_for_texture(name):
                # whiten material
                mat_id = self.model.geom_matid[geom_id]
                self.model.mat_rgba[mat_id, :] = 1.0

        # self.model.mat_rgba[:] = 1.0
        # self.model.geom_rgba[:] = 1.0

    def get_geom_rgb(self, name):
        geom_id = self.model.geom_name2id(name)
        return self.model.geom_rgba[geom_id, :3]

    def set_geom_rgb(self, name, rgb):
        geom_id = self.model.geom_name2id(name)
        self.model.geom_rgba[geom_id, :3] = rgb

    def get_rand_rgb(self, n=1):
        def _rand_rgb():
            return np.array(self.random_state.uniform(size=3) * 255,
                            dtype=np.uint8)

        if n == 1:
            return _rand_rgb()
        else:
            return tuple(_rand_rgb() for _ in range(n))

    # def set_existing_texture(self, name):
    #     bitmap = self.get_texture(name).bitmap
    #     img = Image.new("RGB", (bitmap.shape[1], bitmap.shape[0]))

    #     img_path = self.texture_path + '/' + self.random_state.choice(os.listdir(self.texture_path))
    #     img = Image.open(img_path).convert('RGB')
    #     img = img.convert('RGB')
    #     if name == 'skybox':
    #         img = img.resize((256, 256))
    #     elif name =='table_visual':
    #         img = img.resize((512, 512))
    #     else:
    #         img = img.resize((32, 32))
    #     img = np.array(img)
    #     img = img.astype(np.uint8)
    #     img = np.concatenate([img] * int(bitmap.shape[0] / img.shape[0]), 0)
    #     img.resize(bitmap.shape)
    #     bitmap[..., :] = img

    #     self.upload_texture(name)

    def get_texture(self, name):
        tex_id = self._name_to_tex_id(name)
        texture = self.textures[tex_id]
        return texture

    def set_texture(self, name, bitmap, perturb=False):
        """
        Sets the bitmap for the texture that corresponds
        to geom @name.

        If @perturb is True, then use the computed bitmap
        to perturb the default bitmap slightly, instead
        of replacing it.
        """
        bitmap_to_set = self.get_texture(name).bitmap
        if perturb:
            bitmap = (1. - self.local_rgb_interpolation) * self._defaults[name]['texture'] + self.local_rgb_interpolation * bitmap
        bitmap_to_set[:] = bitmap
        self.upload_texture(name)

    def get_material(self, name):
        mat_id = self._name_to_mat_id(name)
        # Material is in tuple form (reflectance, shininess, specular)
        material = np.array((self.model.mat_reflectance[mat_id],
                             self.model.mat_shininess[mat_id],
                             self.model.mat_specular[mat_id]))
        return material

    def set_material(self, name, material, perturb=False):
        """
        Sets the material that corresponds to geom @name.

        If @perturb is True, then use the computed material
        to perturb the default material slightly, instead
        of replacing it.
        """
        mat_id = self._name_to_mat_id(name)
        if perturb:
            material = (1. - self.local_material_interpolation) * self._defaults[name]['material'] + \
                       self.local_material_interpolation * material
        self.model.mat_reflectance[mat_id] = material[0]
        self.model.mat_shininess[mat_id] = material[1]
        self.model.mat_specular[mat_id] = material[2]

    def get_checker_matrices(self, name):
        tex_id = self._name_to_tex_id(name)
        return self._texture_checker_mats[tex_id]

    def set_checker(self, name, rgb1, rgb2, perturb=False):
        """
        Use the two checker matrices to create a checker
        pattern from the two colors, and set it as 
        the texture for geom @name.
        """
        cbd1, cbd2 = self.get_checker_matrices(name)
        rgb1 = np.asarray(rgb1).reshape([1, 1, -1])
        rgb2 = np.asarray(rgb2).reshape([1, 1, -1])
        bitmap = rgb1 * cbd1 + rgb2 * cbd2

        self.set_texture(name, bitmap, perturb=perturb)

    def set_gradient(self, name, rgb1, rgb2, vertical=True, perturb=False):
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

        new_bitmap = np.zeros_like(bitmap)
        for i in range(3):
            new_bitmap[..., i] = rgb2[i] * p + rgb1[i] * (1.0 - p)

        self.set_texture(name, new_bitmap, perturb=perturb)

    def set_rgb(self, name, rgb, perturb=False):
        """
        Just set the texture bitmap for geom @name
        to a constant rgb value.
        """
        bitmap = self.get_texture(name).bitmap
        new_bitmap = np.zeros_like(bitmap)
        new_bitmap[..., :] = np.asarray(rgb)

        self.set_texture(name, new_bitmap, perturb=perturb)

    def set_noise(self, name, rgb1, rgb2, fraction=0.9, perturb=False):
        """
        Args:
        - name (str): name of geom
        - rgb1 (array): background color
        - rgb2 (array): color of random noise foreground color
        - fraction (float): fraction of pixels with foreground color
        """
        bitmap = self.get_texture(name).bitmap
        h, w = bitmap.shape[:2]
        mask = self.random_state.uniform(size=(h, w)) < fraction

        new_bitmap = np.zeros_like(bitmap)
        new_bitmap[..., :] = np.asarray(rgb1)
        new_bitmap[mask, :] = np.asarray(rgb2)

        self.set_texture(name, new_bitmap, perturb=perturb)

    def upload_texture(self, name):
        """
        Uploads the texture to the GPU so it's available in the rendering.
        """
        texture = self.get_texture(name)
        if not self.sim.render_contexts:
            cymj.MjRenderContextOffscreen(self.sim)
        for render_context in self.sim.render_contexts:
            render_context.upload_texture(texture.id)

    def _check_geom_for_texture(self, name):
        """
        Helper function to determined if the geom @name has
        an assigned material and that the material has
        an assigned texture.
        """
        geom_id = self.model.geom_name2id(name)
        mat_id = self.model.geom_matid[geom_id]
        if mat_id < 0:
            return False
        tex_id = self.model.mat_texid[mat_id]
        if tex_id < 0:
            return False
        return True

    def _name_to_tex_id(self, name):
        """
        Helper function to get texture id from geom name.
        """

        # handle skybox separately
        if name == 'skybox':
            skybox_tex_id = -1
            for tex_id in range(self.model.ntex):
                skybox_textype = 2
                if self.model.tex_type[tex_id] == skybox_textype:
                    skybox_tex_id = tex_id
            assert skybox_tex_id >= 0
            return skybox_tex_id

        assert self._check_geom_for_texture(name)
        geom_id = self.model.geom_name2id(name)
        mat_id = self.model.geom_matid[geom_id]
        tex_id = self.model.mat_texid[mat_id]
        return tex_id

    def _name_to_mat_id(self, name):
        """
        Helper function to get material id from geom name.
        """

        # handle skybox separately
        if name == 'skybox':
            raise ValueError("Error: skybox has no material!")

        assert self._check_geom_for_texture(name)
        geom_id = self.model.geom_name2id(name)
        mat_id = self.model.geom_matid[geom_id]
        return mat_id

    # def _build_tex_geom_map(self):
    #     # Build a map from tex_id to geom_ids, so we can check
    #     # for collisions.
    #     self._geom_ids_by_tex_id = defaultdict(list)
    #     for geom_id in range(self.model.ngeom):
    #         mat_id = self.model.geom_matid[geom_id]
    #         if mat_id >= 0:
    #             tex_id = self.model.mat_texid[mat_id]
    #             if tex_id >= 0:
    #                 self._geom_ids_by_tex_id[tex_id].append(geom_id)

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
        self._texture_checker_mats = []
        for tex_id in range(self.model.ntex):
            texture = self.textures[tex_id]
            h, w = texture.bitmap.shape[:2]
            self._texture_checker_mats.append(self._make_checker_matrices(h, w))

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
