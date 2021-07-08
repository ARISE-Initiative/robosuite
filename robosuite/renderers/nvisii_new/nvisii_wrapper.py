import numpy as np
import robosuite as suite
import nvisii
import nvisii_utils as utils
import open3d as o3d

from robosuite.wrappers import Wrapper
from robosuite.utils import transform_utils as T

from parser import Parser

class NViSIIWrapper(Wrapper):
    def __init__(self,
                 env,
                 img_path,
                 width=500,
                 height=500,
                 spp=256,
                 use_noise=False,
                 debug_mode=False):
        """
        Initializes the nvisii wrapper.
        Args:
            env (MujocoEnv instance): The environment to wrap
            img_path (string): path to images
            width (int, optional): Width of the rendered image. Defaults to 500.
            height (int, optional): Height of the rendered image. Defaults to 500.
            spp (int, optional): sample-per-pixel for each image
            use_noise (bool, optional): Use noise or denoise
            debug_mode (bool, optional): Use debug mode for nvisii
        """

        super().__init__(env)

        self.env = env
        self.img_path = img_path
        self.width = width
        self.height = height
        self.spp = spp
        self.use_noise = use_noise

        self.img_cntr = 0

        # enable interactive mode when debugging
        if debug_mode:
            nvisii.initialize_interactive()
        else:
            nvisii.initialize(headless = True)

        # add denoiser to nvisii if not using noise
        if not use_noise: 
            nvisii.configure_denoiser()
            nvisii.enable_denoiser()
            nvisii.configure_denoiser(True,True,False)

        # Intiailizes the lighting
        self.light_1 = nvisii.entity.create(
            name      = "light",
            mesh      = nvisii.mesh.create_sphere("light"),
            transform = nvisii.transform.create("light"),
        )

        self.light_1.set_light(
            nvisii.light.create("light")
        )

        self.light_1.get_light().set_intensity(150)
        self.light_1.get_transform().set_scale(nvisii.vec3(0.3))
        self.light_1.get_transform().set_position(nvisii.vec3(3, 3, 4))

        self._init_floor(image="wood-floor-4k.jpg")
        self._init_walls(image="marble-4k.jpg")
        self._init_camera()
        # Sets the primary camera of the renderer to the camera entity
        nvisii.set_camera_entity(self.camera)
        self._camera_configuration(pos_vec = nvisii.vec3(0, 0, 1), 
                                   at_vec  = nvisii.vec3(0, 0, 1), 
                                   up_vec  = nvisii.vec3(0, 0, 1),
                                   eye_vec = nvisii.vec3(1.5, 0, 1.5))
        
        # Environment configuration
        self._dome_light_intensity = 1
        nvisii.set_dome_light_intensity(self._dome_light_intensity)
        nvisii.set_max_bounce_depth(4)

        self._load()

    def _init_floor(self, image):
        """
        Intiailizes the floor
        """
        floor_mesh = nvisii.mesh.create_plane(name = "plane",
                                              size = nvisii.vec2(3, 3))

        floor_entity = nvisii.entity.create(
            name      = "floor",
            mesh      = floor_mesh,
            material  = nvisii.material.create("plane"),
            transform = nvisii.transform.create("plane")
        )
        floor_entity.get_transform().set_scale(nvisii.vec3(1))
        floor_entity.get_transform().set_position(nvisii.vec3(0, 0, 0))

        texture_image = f'../../models/assets/textures/{image}'
        texture = nvisii.texture.create_from_file(name = 'floor_texture',
                                                  path = texture_image)

        floor_entity.get_material().set_base_color_texture(texture)
        floor_entity.get_material().set_roughness(0.4)
        floor_entity.get_material().set_specular(0)

    def _init_walls(self, image):
        """
        Intiailizes the walls
        """
        texture_image = f'../../models/assets/textures/{image}'
        texture = nvisii.texture.create_from_file(name = 'wall_texture',
                                                  path = texture_image)

        for wall in self.env.model.mujoco_arena.worldbody.findall("./geom[@material='walls_mat']"):

            name = wall.get('name')
            size = [float(x) for x in wall.get('size').split(' ')]

            pos, quat = self._get_orientation_geom(name)

            wall_entity = nvisii.entity.create(
                        name = name,
                        mesh = nvisii.mesh.create_box(name = name,
                                                      size = nvisii.vec3(size[0],
                                                                         size[1],
                                                                         size[2])),
                        transform = nvisii.transform.create(name),
                        material = nvisii.material.create(name)
                    )

            wall_entity.get_transform().set_position(nvisii.vec3(pos[0],
                                                                 pos[1],
                                                                 pos[2]))

            wall_entity.get_transform().set_rotation(nvisii.quat(quat[0],
                                                                 quat[1],
                                                                 quat[2],
                                                                 quat[3]))

            wall_entity.get_material().set_base_color_texture(texture)

    def _init_camera(self):

        # intializes the camera
        self.camera = nvisii.entity.create(
            name = "camera",
            transform = nvisii.transform.create("camera_transform"),
        )

        self.camera.set_camera(
            nvisii.camera.create_from_fov(
                name = "camera_camera", 
                field_of_view = 1, 
                aspect = float(self.width)/float(self.height)
            )
        )

    def _camera_configuration(self, pos_vec, at_vec, up_vec, eye_vec):

        # configures the camera
        self.camera.get_transform().set_position(pos_vec)

        self.camera.get_transform().look_at(
            at  = at_vec, # look at (world coordinate)
            up  = up_vec, # up vector
            eye = eye_vec,
            previous = False
        )

    def _get_orientation_geom(self, name):

        pos = self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(name)]
        R = self.env.sim.data.geom_xmat[self.env.sim.model.geom_name2id(name)].reshape(3, 3)

        quat_xyzw = utils.quaternion_from_matrix3(R)
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return pos, quat

    def _load(self):        
        parser = Parser(self.env)
        parser.parse_textures()
        parser.parse_materials()
        parser.parse_geometries()
        self.components = parser.components

    def step(self, action):
        """
        Updates the states for the wrapper given a certain action
        Args:
            action (np-array): the action the robot should take
        """
        ret_val = super().step(action)
        for key, value in self.components.items():
            self._update_orientation(name=key, component=value)

        return ret_val

    def _update_orientation(self, name, component):
        """
        Update position for an object or a robot in renderer.
        :param component: object in renderer and parent body name
        """
        obj = component[0]
        parent_body = component[1]
        geom_quat = component[2]
        dynamic = component[3]

        if not dynamic:
            return
        
        self.body_tags = ['robot', 'pedestal', 'gripper', 'peg']

        if parent_body != 'worldbody':
            if self.tag_in_name(name):
                pos = self.env.sim.data.get_body_xpos(parent_body)
            else:
                pos = self.env.sim.data.get_geom_xpos(name)

            B = self.env.sim.data.body_xmat[self.env.sim.model.body_name2id(parent_body)].reshape((3, 3))      
            quat_xyzw_body = utils.quaternion_from_matrix3(B)
            quat_wxyz_body = np.array([quat_xyzw_body[3], quat_xyzw_body[0], quat_xyzw_body[1], quat_xyzw_body[2]]) # wxyz
            nvisii_quat = nvisii.quat(*geom_quat) * nvisii.quat(*quat_wxyz_body)
        else:
            pos = [0,0,0]
            nvisii_quat = nvisii.quat([1,0,0,0]) # wxyz

        if isinstance(obj, nvisii.scene):
            obj.transforms[0].set_position(nvisii.vec3(pos[0],
                                                       pos[1],
                                                       pos[2]))
            obj.transforms[0].set_rotation(nvisii_quat)
        else:
            obj.get_transform().set_position(nvisii.vec3(pos[0],
                                                         pos[1],
                                                         pos[2]))
            obj.get_transform().set_rotation(nvisii_quat)

    def tag_in_name(self, name):
        for tag in self.body_tags:
            if tag in name:
                return True
        return False

    def render(self, render_type="png"):
        self.img_cntr += 1
        nvisii.render_to_file(
            width             = self.width,
            height            = self.height, 
            samples_per_pixel = self.spp,   
            file_path         = f'{self.img_path}/image_{self.img_cntr}.{render_type}'
        )

    def close(self):
        """
        Deinitializes the nvisii rendering environment
        """
        nvisii.deinitialize()