import os
import numpy as np
import sys
import visii
import robosuite as suite
import dynamic_object_initialization as dyn_init
import static_object_initialization as static_init
import rendering_objects as ren
import visii_utils as vutils
from mujoco_py import MjSim, load_model_from_path
from robosuite.wrappers import Wrapper
from robosuite.models.robots import Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
import open3d as o3d
import math
import xml.etree.ElementTree as ET

class VirtualWrapper(Wrapper):
    """Initializes the ViSII wrapper.
    Args:
        env (MujocoEnv instance): The environment to wrap.
        width (int, optional)
        height (int, optional)
        spp (int, optional): sample-per-pixel for each image
        use_noise (bool, optional): Use noise or denoise
        debug_mode (bool, optional): Use debug mode for visii
    """

    def __init__(self,
                 env,
                 width=1000,
                 height=1000,
                 spp=50,
                 use_noise=True,
                 debug_mode=False):

        super().__init__(env)

        if not os.path.exists('images'):
            os.makedirs('images')

        # Camera variables
        self.width             = width
        self.height            = height
        self.samples_per_pixel = spp

        # Counter for images
        self.image_counter = 0

        # Entity variables
        self.entities               = {}
        self.gripper_entities       = {}
        self.static_object_entities = {}

        # Stores info for robot(s) and gripper(s)
        self.robot_info = {}
        self.gripper_info = {}

        self.num_meshes_per_robot = []

        if debug_mode:
            visii.initialize_interactive()
        else:
            visii.initialize(headless = True)

        if not use_noise: 
            visii.enable_denoiser()

        # Intiailizes the lighting
        light_1 = visii.entity.create(
            name      = "light_1",
            mesh      = visii.mesh.create_sphere("light_1"),
            transform = visii.transform.create("light_1"),
        )

        light_1.set_light(
            visii.light.create("light_1")
        )

        light_1.get_light().set_intensity(150)
        light_1.get_transform().set_scale(visii.vec3(0.3))
        light_1.get_transform().set_position(visii.vec3(3, 3, 4))
        
        # Intiailizes the floor
        floor_mesh = visii.mesh.create_plane(name = "plane",
                                             size = visii.vec2(3, 3))

        floor_entity = visii.entity.create(
            name      = "floor",
            mesh      = floor_mesh,
            material  = visii.material.create("plane"),
            transform = visii.transform.create("plane")
        )
        floor_entity.get_transform().set_scale(visii.vec3(1))
        floor_entity.get_transform().set_position(visii.vec3(0, 0, 0))

        wood_floor_image = '../../models/assets/textures/wood-floor-4k.png'
        wood_floor_texture = visii.texture.create_from_file(name = 'flooring_texture',
                                                            path = wood_floor_image)

        floor_entity.get_material().set_base_color_texture(wood_floor_texture)

        floor_entity.get_material().set_roughness(0.4)
        floor_entity.get_material().set_specular(0)

        gray_plaster_image = '../../models/assets/textures/gray-plaster-rough-4k.png'
        gray_plaster_texture = visii.texture.create_from_file(name = 'walls_texture',
                                                              path = gray_plaster_image)
        
        # Intiailizes the walls
        for wall in self.env.model.mujoco_arena.worldbody.findall("./geom[@material='walls_mat']"):

            name = wall.get('name')
            size = [float(x) for x in wall.get('size').split(' ')]

            pos, quat = self._get_pos_quat(name)

            wall_entity = visii.entity.create(
                        name = name,
                        mesh = visii.mesh.create_box(name = name,
                                                     size = visii.vec3(size[0],
                                                                       size[1],
                                                                       size[2])),
                        transform = visii.transform.create(name),
                        material = visii.material.create(name)
                    )

            wall_entity.get_transform().set_position(visii.vec3(pos[0],
                                                                pos[1],
                                                                pos[2]))

            wall_entity.get_transform().set_rotation(visii.quat(quat[0],
                                                                quat[1],
                                                                quat[2],
                                                                quat[3]))

            wall_entity.get_material().set_base_color_texture(gray_plaster_texture)


        self._camera_init()

        # Sets the primary camera of the renderer to the camera entity
        visii.set_camera_entity(self.camera) 

        self._camera_configuration(pos_vec = visii.vec3(0, 0, 1), 
                                  at_vec  = visii.vec3(0, 0, 1), 
                                  up_vec  = visii.vec3(0, 0, 1),
                                  eye_vec = visii.vec3(1.5, 0, 1.5))
        
        # Environment configuration
        self._dome_light_intensity = 1
        visii.set_dome_light_intensity(self._dome_light_intensity)

        visii.set_max_bounce_depth(4)

        robot_count = 0

        for robot in self.env.robots:

            robot_xml_filepath = robot.robot_model.file
            gripper_xml_filepath = robot.gripper.file

            robot_name = robot.name
            gripper_name = gripper_xml_filepath[gripper_xml_filepath.rfind('/')+1:gripper_xml_filepath.rfind('.')]

            self._initalize_simulation(robot_xml_filepath)

            root_robot = ET.parse(robot_xml_filepath).getroot()
            root_gripper = ET.parse(gripper_xml_filepath).getroot()

            robot_meshes, robot_positions, robot_quats, robot_part_count, robot_entities = dyn_init.dynamic_robot_init(env=self.env,
                                                                                                                       root=root_robot,
                                                                                                                       robot_num=robot_count,
                                                                                                                       robot_name=robot_name)

            gripper_meshes, gripper_positions, gripper_quats, gripper_geom_quats, gripper_part_count, gripper_entities = dyn_init.dynamic_gripper_init(env=self.env,
                                                                                                                                   root=root_gripper,
                                                                                                                                   robot_num=robot_count,
                                                                                                                                   gripper_name=gripper_name)

            robot_id = f'{robot_name}_{robot_count}'

            self.robot_info[robot_id] = [robot_meshes, robot_positions, robot_quats, robot_part_count, robot_entities]
            self.gripper_info[robot_id] = [gripper_meshes, gripper_positions, gripper_quats, gripper_part_count, gripper_entities, gripper_geom_quats]

            robot_count += 1

        self._init_objects_in_visii()

    def _init_objects_in_visii(self):

        static_init.init_arena_visii(self.env)
        static_init.init_pedestals(self.env)

        self.obj_entity_dict = static_init.init_objects_visii(self.env)

        # print(self.obj_entity_dict)

    def _camera_init(self):

        # intializes the camera
        self.camera = visii.entity.create(
            name = "camera",
            transform = visii.transform.create("camera_transform"),
        )

        self.camera.set_camera(
            visii.camera.create_from_fov(
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

    def _initalize_simulation(self, xml_file = None):
        
        # Load the model from the xml file
        self.mjpy_model = load_model_from_path(xml_file) if xml_file else self.model.get_model(mode="mujoco_py")

        # Creates the simulation
        self.env.sim.step()

    def _get_pos_quat(self, name):

        pos = self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(name)]
        R = self.env.sim.data.geom_xmat[self.env.sim.model.geom_name2id(name)].reshape(3, 3)

        quat_xyzw = vutils._quaternion_from_matrix3(R)
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return pos, quat

    def step(self, action):
        """Updates the states for the wrapper given a certain action
        Args:
            action (np-array): the action the robot should take
        """

        obs_dict, reward, done, info = self.env.step(action)
        self._update(obs_dict, reward, done, info)

        return obs_dict, reward, done, info

    def _update(self, obs_dict, reward, done, info):
        
        self.obs_dict = obs_dict
        self.reward = reward
        self.done = done
        self.info = info

        robot_num = 0
        for robot in self.robot_info.keys():

            robot_meshes = self.robot_info[robot][0]
            gripper_meshes = self.gripper_info[robot][0]
            
            self.robot_info[robot][1] = vutils.get_positions(env=self.env,
                                                             part_type='robot',
                                                             parts=robot_meshes, 
                                                             robot_num=robot_num)

            self.robot_info[robot][2] = vutils.get_quaternions(env=self.env,
                                                               part_type='robot',
                                                               parts=robot_meshes, 
                                                               robot_num=robot_num)

            self.gripper_info[robot][1] = vutils.get_positions(env=self.env,
                                                               part_type='gripper',
                                                               parts=gripper_meshes, 
                                                               robot_num=robot_num)

            self.gripper_info[robot][2] = vutils.get_quaternions(env=self.env,
                                                                 part_type='gripper',
                                                                 parts=gripper_meshes, 
                                                                 robot_num=robot_num)

            robot_num += 1

    def render(self, render_type='png'):
        """ Renders the environment using the ViSII renderer
        Args: 
            render_type (str, optional): renders the image as either 'png' or 'hdr'
        """

        self.image_counter+=1

        # render the robots
        ren.render_robots(self.env, self.robot_info)
        ren.render_grippers(self.env, self.gripper_info)
        ren.render_objects(self.env, self.obj_entity_dict)

        visii.render_to_file(
            width             = self.width,
            height            = self.height, 
            samples_per_pixel = self.samples_per_pixel,   
            file_path         = f'images/image_{self.image_counter}.png'
        )

    def close(self):
        """Deinitializes the ViSII rendering environment
        """
        visii.deinitialize()

    def get_camera_intrinsics(self):
        """Get camera intrinsics matrix
        """
        return self.camera.get_transform.get_world_to_local_matrix()

    def get_camera_extrinsics(self):
        """Get camera extrinsics matrix
        """
        return self.camera.get_transform.get_local_to_world_matrix()

    def set_camera_intrinsics(self, fov, width, height):
        """Set camera intrinsics matrix
        """
        self.width = width
        self.height = height

        self.camera.set_camera(
            visii.camera.create_from_fov(
                name = "camera_camera", 
                field_of_view = fov, 
                aspect = float(width)/float(height)
            )
        )

    def set_camera_extrinsics(self, at_vec, up_vec, eye_vec):
        """Set camera extrinsics matrix
        """
        self.camera.get_transform().look_at(
            at  = at_vec, # look at (world coordinate)
            up  = up_vec, # up vector
            eye = eye_vec,
            previous = False
        )

if __name__ == '__main__':

    # Registered environments: Lift :), Stack :), NutAssembly :), NutAssemblySingle :), NutAssemblySquare :), NutAssemblyRound :),
    #                          PickPlace :), PickPlaceSingle :), PickPlaceMilk :), PickPlaceBread :), PickPlaceCereal :), 
    #                          PickPlaceCan :), Door :), Wipe :), TwoArmLift :), TwoArmPegInHole :), TwoArmHandover :)

    # Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e

    env = VirtualWrapper(
        env = suite.make(
                "TwoArmLift",
                robots = ["Panda", "Sawyer"],
                reward_shaping=True,
                has_renderer=False,           # no on-screen renderer
                has_offscreen_renderer=False, # no off-screen renderer
                ignore_done=True,
                use_object_obs=True,          # use object-centric feature
                use_camera_obs=False,         # no camera observations
                control_freq=10, 
            ),
        spp=10,
        use_noise=False,
        debug_mode=False,
    )

    env.reset()

    for i in range(200):
        action = np.random.randn(16)
        obs, reward, done, info = env.step(action)

        if i%100 == 0:
            env.render(render_type = "png")

    env.close()
    
    print('Done.')
