import os
import numpy as np
import robosuite as suite
import nvisii
import robosuite.renderers.nvisii.nvisii_utils as utils
import open3d as o3d
import cv2

from robosuite.wrappers import Wrapper
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv

from robosuite.renderers.nvisii.parser import Parser
from robosuite.utils.transform_utils import mat2quat

class NViSIIWrapper(Wrapper):
    def __init__(self,
                 env,
                 img_path='images/',
                 width=500,
                 height=500,
                 spp=256,
                 use_noise=False,
                 debug_mode=False,
                 video_mode=False,
                 video_path='videos/',
                 video_name='robosuite_video_0.mp4',
                 video_fps=60,
                 verbose=1,
                 image_options=None):
        """
        Initializes the nvisii wrapper. Wrapping any MuJoCo environment in this 
        wrapper will use the NViSII wrapper for rendering.

        Args:
            env (MujocoEnv instance): The environment to wrap.

            img_path (string): Path to images.

            width (int, optional): Width of the rendered image. Defaults to 500.

            height (int, optional): Height of the rendered image. Defaults to 500.

            spp (int, optional): Sample-per-pixel for each image. Higher spp will result
                                 in higher quality images but will take more time to render
                                 each image. Higher quality images typically use an spp of
                                 around 512.

            use_noise (bool, optional): Use noise or denoise. Deafults to false.

            debug_mode (bool, optional): Use debug mode for nvisii. Deafults to false.

            video_mode (bool, optional): By deafult, the NViSII wrapper saves the results as 
                                         images. If video_mode is set to true, a video is
                                         produced and will be stored in the directory defined
                                         by video_path. Defaults to false.
            
            video_path (string, optional): Path to store the video. Required if video_mode is 
                                           set to true. Defaults to 'videos/'.

            video_name (string, optional): Name for the file for the video. Defaults to
                                           'robosuite_video_0.mp4'.
            
            video_fps (int, optional): Frames per second for video. Defaults to 60.

            verbose (int, optional): If verbose is set to 1, the wrapper will print the image
                                     number for each image rendered. If verbose is set to 0, 
                                     nothing will be printed. Defaults to 1.

            image_options (string, optional): Options to render image with different ground truths
                                              for NViSII. Options include "normal", "texture_coordinates",
                                              "position"
        """

        super().__init__(env)

        self.env = env
        self.img_path = img_path
        self.width = width
        self.height = height
        self.spp = spp
        self.use_noise = use_noise

        self.video_mode = video_mode
        self.video_path = video_path
        self.video_name = video_name
        self.video_fps = video_fps

        self.verbose = verbose
        self.image_options = image_options

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

        if not os.path.exists(img_path):
            os.makedirs(img_path)

        if video_mode:             
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            self.video = cv2.VideoWriter(video_path + video_name, cv2.VideoWriter_fourcc(*'MP4V'), video_fps, (self.width, self.height))
            print(f'video mode enabled')

        if image_options is None:
            nvisii.sample_pixel_area(
                x_sample_interval = (0.0, 1.0), 
                y_sample_interval = (0.0, 1.0)
            )
        else:
            nvisii.sample_pixel_area(
                x_sample_interval = (.5,.5), 
                y_sample_interval = (.5, .5)
            )

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

        self._init_floor(image="plywood-4k.jpg")
        self._init_walls(image="plaster-wall-4k.jpg")
        self._init_camera()
        # Sets the primary camera of the renderer to the camera entity
        nvisii.set_camera_entity(self.camera)
        self._camera_configuration(at_vec  = nvisii.vec3(0, 0, 1), 
                                   up_vec  = nvisii.vec3(0, 0, 1),
                                   eye_vec = nvisii.vec3(1.5, 0, 1.5),
                                   quat    = nvisii.quat(-1, 0, 0, 0))
        
        # Environment configuration
        self._dome_light_intensity = 1
        nvisii.set_dome_light_intensity(self._dome_light_intensity)
        nvisii.set_max_bounce_depth(4)

        self._load()

    def _init_floor(self, image):
        """
        Intiailizes the floor

        Args:
            image (string): String for the file to use as an image for the floor

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

        texture_image = xml_path_completion("textures/" + image)
        texture = nvisii.texture.create_from_file(name = 'floor_texture',
                                                  path = texture_image)

        floor_entity.get_material().set_base_color_texture(texture)
        floor_entity.get_material().set_roughness(0.4)
        floor_entity.get_material().set_specular(0)

    def _init_walls(self, image):
        """
        Intiailizes the walls

        Args:
            image (string): String for the file to use as an image for the walls
        """
        texture_image = xml_path_completion("textures/" + image)
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
        """
        Intializes the camera for the NViSII renderer
        """

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

    def _camera_configuration(self, at_vec, up_vec, eye_vec, quat):
        """
        Sets the configuration for the NViSII camera. Configuration
        is dependent on where the camera is located and where it 
        looks at
        """
        # configures the camera
        self.camera.get_transform().look_at(
            at  = at_vec, # look at (world coordinate)
            up  = up_vec, # up vector
            eye = eye_vec,
            previous = False
        )

        self.camera.get_transform().rotate_around(eye_vec, quat)

    def set_camera_pos_quat(self, pos, quat):
        self.camera.get_transform().set_position(pos)
        self.camera.get_transform().look_at(
            at  = (0, 0, 1), # look at (world coordinate)
            up  = (0, 0, 1), # up vector
            eye = pos,
            previous = False
        )
        # self.camera.get_transform().rotate_around(pos, quat)

    def _get_orientation_geom(self, name):
        """
        Gets the position and quaternion for a geom
        """

        pos = self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(name)]
        R = self.env.sim.data.geom_xmat[self.env.sim.model.geom_name2id(name)].reshape(3, 3)

        quat_xyzw = mat2quat(R)
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return pos, quat

    def _load(self):
        """
        Loads the nessecary textures, materials, and geoms into the
        NViSII renderer
        """   
        parser = Parser(self.env)
        parser.parse_textures()
        parser.parse_materials()
        parser.parse_geometries()
        self.components = parser.components

    def step(self, action):
        """
        Updates the states for the wrapper given a certain action

        Args:
            action (np-array): The action the robot should take
        """
        ret_val = super().step(action)
        for key, value in self.components.items():
            self._update_orientation(name=key, component=value)

        return ret_val

    def _update_orientation(self, name, component):
        """
        Update position for an object or a robot in renderer.
        
        Args:
            name (string): name of component
            component (nvisii entity or scene): Object in renderer and other info
                                                for object.
        """

        obj = component.obj
        parent_body_name = component.parent_body_name
        geom_pos = component.geom_pos
        geom_quat = component.geom_quat
        dynamic = component.dynamic
        
        if not dynamic:
            return
        
        self.body_tags = ['robot', 'pedestal', 'gripper', 'peg']

        if parent_body_name != 'worldbody':
            if self.tag_in_name(name):
                pos = self.env.sim.data.get_body_xpos(parent_body_name)
            else:
                pos = self.env.sim.data.get_geom_xpos(name)

            B = self.env.sim.data.body_xmat[self.env.sim.model.body_name2id(parent_body_name)].reshape((3, 3))      
            quat_xyzw_body = mat2quat(B)
            quat_wxyz_body = np.array([quat_xyzw_body[3], quat_xyzw_body[0], quat_xyzw_body[1], quat_xyzw_body[2]]) # wxyz
            nvisii_quat = nvisii.quat(*quat_wxyz_body) * nvisii.quat(*geom_quat)

            if self.tag_in_name(name):
                # Add position offset if there are position offset defined in the geom tag
                homo_mat = T.pose2mat((np.zeros((1, 3), dtype=np.float32), quat_xyzw_body))
                pos_offset = homo_mat @ np.array([geom_pos[0], geom_pos[1], geom_pos[2], 1.]).transpose()
                pos = pos + pos_offset[:3]
            
        else:
            pos = [0,0,0]
            nvisii_quat = nvisii.quat(1,0,0,0) # wxyz

        if isinstance(obj, nvisii.scene):

            # temp fix -- look into XML file for correct quat
            if 's_visual' in name:
                # single robot
                if isinstance(self.env, SingleArmEnv):
                    nvisii_quat = nvisii.quat(0, 0.5, 0, 0)
                # two robots - 0
                elif isinstance(self.env, TwoArmEnv) and 'robot_0' in name:
                    nvisii_quat = nvisii.quat(-0, 0.5, 0.5, 0)
                # two robots - 1
                else:
                    nvisii_quat = nvisii.quat(-0, 0.5, -0.5, 0)

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
        """
        Checks if one of the tags in body tags in the name

        Args:
            name (string): Name of component
        """
        for tag in self.body_tags:
            if tag in name:
                return True
        return False

    def render(self, render_type="png"):
        """
        Renders an image of the NViSII renderer
        
        Args:
            render_type (string, optional): Type of file to save as. Defaults to 'png'
        """

        self.img_cntr += 1
        verbose_word = 'frame' if self.video_mode else 'image'

        if self.video_mode:
            img_file = f'{self.img_path}/image_0.{render_type}'
            if self.image_options is None:
                self.render_to_file(img_file)
            else:
                self.render_data_to_file(img_file)

            self.video.write(cv2.imread(img_file))
        else:
            img_file = f'{self.img_path}/image_{self.img_cntr}.{render_type}'
            if self.image_options is None:
                self.render_to_file(img_file)
            else:
                self.render_data_to_file(img_file)

        if self.verbose == 1:
            print(f'Rendering {verbose_word}... {self.img_cntr}')

    def render_to_file(self, img_file):
        nvisii.render_to_file(
            width = self.width,
            height = self.height,
            samples_per_pixel = self.spp,
            file_path = img_file
        )

    def render_data_to_file(self, img_file):
        nvisii.render_data_to_file(
            width = self.width,
            height = self.height, 
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options=self.image_options,
            file_path=img_file
        )


    def close(self):
        """
        Deinitializes the nvisii rendering environment
        """
        nvisii.deinitialize()
