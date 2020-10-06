"""

This file implements a wrapper for using the ViSII imaging interface
for robosuite. 

"""

import numpy as np
import sys
import visii
import robosuite as suite
from robosuite.wrappers import Wrapper
from robosuite.models.robots import Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
import open3d as o3d

class VirtualWrapper(Wrapper):
    """

    Initializes the ViSII wrapper.

    Args:
        env (MujocoEnv instance): The environment to wrap.
        width (int, optional)
        height (int, optional)
        use_noise (bool, optional): Use noise or denoise
        debug_mode (bool, optional): Use debug mode for visii
    """
    def __init__(self,
                 env,
                 width=500,
                 height=500,
                 use_noise=True,
                 debug_mode=False):
        super().__init__(env)

        # Camera variables
        self.width = width
        self.height = height

        if debug_mode:
            visii.initialize_interactive()
        else:
            visii.initialize_headless()

        if not use_noise: 
            visii.enable_denoiser()

        # Creates an entity
        self.camera = visii.entity.create(
            name = "camera",
            transform = visii.transform.create("camera_transform"),
        )

        # Adds a camera to the entity
        self.camera.set_camera(
            visii.camera.create_perspective_from_fov(
                name = "camera_camera", 
                field_of_view = 1, 
                aspect = float(self.width)/float(self.height)
            )
        )

        # Sets the primary camera to the renderer to the camera entity
        visii.set_camera_entity(self.camera)

        # Lets place our camera to look at the scene
        # all the position are defined by visii.vector3  

        # TODO (yifeng): 1. Put camera intiialization to a seprate
        # method
        # TODO (yifeng): 2. Create another function to configure the camera
        # parameters when needed

        self.camera.get_transform().set_position(visii.vec3(0, 0, 1))

        self.camera.get_transform().look_at(
            at = visii.vec3(0,0,0) , # look at (world coordinate)
            up = visii.vec3(0,0,1), # up vector
            eye = visii.vec3(1.5, 1.5, 1.5),
            previous = False
        )

        # (Yifeng): These two lines are breaking things
        # self.camera.get_camera().set_aperture_diameter(2000)
        # self.camera.get_camera().set_focal_distance(500)

        # Environment configuration
        self._dome_light_intensity = 1
        visii.set_dome_light_intensity(self._dome_light_intensity)

        visii.set_max_bounce_depth(2)
        
        self.robots = []
        self.robots_names = self.env.robot_names

        for robot in self.robot_names:
            self.robots.append(self.str_to_class(robot)())

    def close(self):
        visii.deinitialize()

    def str_to_class(self, str):
        return getattr(sys.modules[__name__], str)

    def reset(self):
        self.obs_dict = self.env.reset()

    def step(self, action):

        """
        Updates the states for the wrapper given a certain action
        """

        obs_dict, reward, done, info = self.env.step(action)
        self.update(obs_dict, reward, done, info)

        return obs_dict, reward, done, info

    def update(self, obs_dict, reward, done, info):
        self.obs_dict = obs_dict
        self.reward   = reward
        self.done     = done
        self.info     = info

    def render(self, render_type):
        """
        Renders the scene
        Arg:
            render_type: tells the method whether to save to png or save to hdr
        """

        # For now I am only rendering it as a png

        # TODO (yifeng): Put static objects intitialization in the
        # beginning.

        light_1 = visii.entity.create(
            name="light_1",
            mesh=visii.mesh.create_plane("light_1"),
            transform=visii.transform.create("light_1"),
        )

        light_1.set_light(
            visii.light.create("light_1")
        )

        light_1.get_light().set_intensity(20000)
        light_1.get_transform().set_scale(visii.vec3(0.3))
        light_1.get_transform().set_position(visii.vec3(-3, -3, 2))
        
        floor = visii.entity.create(
            name = "floor",
            mesh = visii.mesh.create_plane("plane"),
            material = visii.material.create("plane"),
            transform = visii.transform.create("plane")
        )
        floor.get_transform().set_scale(visii.vec3(10))
        floor.get_transform().set_position(visii.vec3(0, 0, -5))
        floor.get_material().set_base_color(visii.vec3(0.8, 1, 0.8))
        floor.get_material().set_roughness(0.4)
        floor.get_material().set_specular(0)

        # adds the object to the scene

        # We decide if we use obj files or stl files based on the
        # robot description file
        
        # obj file extension
        # obj = visii.import_obj(
        #     "obj", # prefix name
        #     '../models/assets/robots/sawyer/meshes/head.obj', #obj path
        #     '../models/assets/robots/sawyer/meshes/', # mtl folder 
        #     visii.vec3(0,0,0), # translation 
        #     visii.vec3(1), # scale here
        #     visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) #rotation here
        # )
        # obj[0].get_transform().set_position(visii.vec3(0,0,1))
        # obj[0].get_transform().set_scale(visii.vec3(1.0))
        # obj[0].get_material().set_base_color(visii.vec3(0.9, 0.9, 0.9)) 

        # stl file extension
        mesh = o3d.io.read_triangle_mesh('../models/assets/robots/sawyer/meshes/head.stl')
        link_name = 'head'

        normals = np.array(mesh.vertex_normals).flatten().tolist()
        vertices = np.array(mesh.vertices).flatten().tolist() 

        mesh = visii.mesh.create_from_data(f'{link_name}_mesh', positions=vertices, normals=normals)

        link_entity = visii.entity.create(
            name=link_name,
            mesh=mesh,
            transform=visii.transform.create(link_name),
            material=visii.material.create(link_name)
        )

        link_entity.get_transform().set_position(visii.vec3(0, 0, 0.2))

        link_entity.get_material().set_base_color(visii.vec3(0.2, 0.2, 0.2))
        link_entity.get_material().set_metallic(0)
        link_entity.get_material().set_transmission(0)
        link_entity.get_material().set_roughness(0.3)

        visii.render_to_png(
            width = self.width,
            height = self.height, 
            samples_per_pixel = 500,   
            image_path = 'temp.png'
        )
        
    #def parse_mjcf_files(self):


    def printState(self): # For testing purposes
        print(self.obs_dict)

    def get_camera_intrinsics(self):
        """Get camera intrinsics matrix

        """
        raise NotImplementedError

    def get_camera_extrinsics(self):
        """Get camera extrinsics matrix

        """
        raise NotImplementedError

    def set_camera_intrinsics(self):
        """Get camera extrinsics matrix

        """
        raise NotImplementedError

    def set_camera_extrinsics(self):
        """Get camera extrinsics matrix

        """
        raise NotImplementedError
        
        
if __name__ == '__main__':

    env = VirtualWrapper(
        env = suite.make(
                "Lift",
                robots = "Sawyer",
                reward_shaping=True,
                has_renderer=False,       # no on-screen renderer
                has_offscreen_renderer=False, # no off-screen renderer
                ignore_done=True,
                use_object_obs=True,      # use object-centric feature
                use_camera_obs=False,     # no camera observations
                control_freq=10,
            ),
        use_noise=False,
    )

    env.reset()

    action = np.random.randn(8)
    obs, reward, done, info = env.step(action)

    #env.printState() # For testing purposes

    env.render(render_type = "png")

    env.close()

    print('Done.')

