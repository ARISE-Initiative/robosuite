"""
This file implements a wrapper for using the ViSII imaging interface
for robosuite. 
"""

import numpy as np
import sys
import visii
import robosuite as suite
from mujoco_py import MjSim
from mujoco_py import load_model_from_path
from robosuite.wrappers import Wrapper
from robosuite.models.robots import Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
from robosuite.models.robots import create_robot
import open3d as o3d
import math
import xml.etree.ElementTree as ET

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
        self.width             = width
        self.height            = height
        self.samples_per_pixel = 10
        self.image_counter     = 0
        self.render_number     = 1
        self.entities          = []

        if debug_mode:
            visii.initialize_interactive()
        else:
            visii.initialize_headless()

        if not use_noise: 
            visii.enable_denoiser()

        light_1 = visii.entity.create(
            name      = "light_1",
            mesh      = visii.mesh.create_plane("light_1"),
            transform = visii.transform.create("light_1"),
        )

        light_1.set_light(
            visii.light.create("light_1")
        )

        light_1.get_light().set_intensity(20000)
        light_1.get_transform().set_scale(visii.vec3(0.3))
        light_1.get_transform().set_position(visii.vec3(-3, -3, 2))
        
        floor = visii.entity.create(
            name      = "floor",
            mesh      = visii.mesh.create_plane("plane"),
            material  = visii.material.create("plane"),
            transform = visii.transform.create("plane")
        )
        floor.get_transform().set_scale(visii.vec3(10))
        floor.get_transform().set_position(visii.vec3(0, 0, -5))
        floor.get_material().set_base_color(visii.vec3(0.8, 1, 0.8))
        floor.get_material().set_roughness(0.4)
        floor.get_material().set_specular(0)

        self.camera_init()

        # Sets the primary camera to the renderer to the camera entity
        visii.set_camera_entity(self.camera) 

        self.camera_configuration(pos_vec = visii.vec3(0, 0, 1), 
                                  at_vec  = visii.vec3(0,0,0), 
                                  up_vec  = visii.vec3(0,0,1),
                                  eye_vec = visii.vec3(1.5, 1.5, 1.5))
        
        # Environment configuration
        self._dome_light_intensity = 1
        visii.set_dome_light_intensity(self._dome_light_intensity)

        visii.set_max_bounce_depth(2)

        self.model = None
        
        self.robots = []
        self.robots_names = self.env.robot_names

        idNum = 1
        for robot in self.robot_names:
            robot_model = create_robot(robot, idn = idNum) # only taking the first robot for now
            self.robots.append(robot_model)
            idNum+=1

        # Passes the xml file based on the robot
        robot_xml_filepath = f'../models/assets/robots/{self.robot_names[0].lower()}/robot.xml'
        self.initalize_simulation(robot_xml_filepath)

        # Available "body" names = ('world', 'base', 'controller_box', 'pedestal_feet', 'torso', 'pedestal', 'right_arm_base_link', 
        #                           'right_l0', 'head', 'screen', 'head_camera', 'right_torso_itb', 'right_l1', 'right_l2', 'right_l3', 
        #                           'right_l4', 'right_arm_itb', 'right_l5', 'right_hand_camera', 'right_wrist', 'right_l6', 'right_hand',
        #                           'right_l4_2', 'right_l2_2', 'right_l1_2').

        # TODO (Yifeng): You should be reading these parts' names from
        # the xml files directly

        tree = ET.parse(robot_xml_filepath)
        root = tree.getroot()
        
        self.dynamic_obj_init(root)

        # initialize the static objects
        #self.static_obj_init()

    def close(self):
        visii.deinitialize()

    def camera_init(self):

        # intializes the camera
        self.camera = visii.entity.create(
            name = "camera",
            transform = visii.transform.create("camera_transform"),
        )

        self.camera.set_camera(
            visii.camera.create_perspective_from_fov(
                name = "camera_camera", 
                field_of_view = 1, 
                aspect = float(self.width)/float(self.height)
            )
        )

    def camera_configuration(self, pos_vec, at_vec, up_vec, eye_vec):

        # configures the camera
        self.camera.get_transform().set_position(pos_vec)

        self.camera.get_transform().look_at(
            at  = at_vec, # look at (world coordinate)
            up  = up_vec, # up vector
            eye = eye_vec,
            previous = False
        )

    def static_obj_init(self):

        # create the tables and walls

        raise NotImplementedError

    def dynamic_obj_init(self, root):

        self.parts      = []
        self.meshes     = []
        self.mesh_parts = []

        # Stores all the meshes required for the robot
        for body in root.iter('body'):
            self.parts.append(body.get('name'))
            for geom in body.findall('geom'):
                geom_mesh = geom.get('mesh')
                if geom_mesh != None:
                    self.meshes.append(geom_mesh)
                    self.mesh_parts.append(body.get('name'))

        # TODO (Yifeng): Try to create a list of objects, one of which
        # contains: geom info, position info, orientation info
        self.positions = self.get_positions(self.mesh_parts) # position information for the robot
        self.geoms     = self.get_geoms(root) # geometry information for the robot
        self.quats     = self.get_quats() # orientation information for the robot

        # print(self.parts)
        # print(self.meshes)
        # print(self.mesh_parts)
        # print(self.positions) # testing
        # print(self.quats)

    def initalize_simulation(self, xml_file = None):
        
        self.mjpy_model = load_model_from_path(xml_file) if xml_file else self.model.get_model(mode="mujoco_py")

        # Creates the simulation
        self.sim = MjSim(self.mjpy_model)
        self.sim.step()

    def str_to_class(self, str):
        return getattr(sys.modules[__name__], str)

    def reset(self):
        self.obs_dict = self.env.reset()

    def step(self, action):

        """
        Updates the states for the wrapper given a certain action
        """

        obs_dict, reward, done, info = self.env.step(action)
        self.sim.step()
        self.update(obs_dict, reward, done, info)

        return obs_dict, reward, done, info

    def update(self, obs_dict, reward, done, info):
        self.obs_dict = obs_dict
        self.reward   = reward
        self.done     = done
        self.info     = info

        self.image_counter += 1

        # changing the angles to the new angles of the joints
        self.positions = self.get_positions(self.mesh_parts)
        self.quats = self.get_quats()

        # call the render function to update the states in the window
        self.render(render_type = "png")


    def render(self, render_type):
        """
        Renders the scene
        Arg:
            render_type: tells the method whether to save to png or save to hdr
        """

        # For now I am only rendering it as a png

        # stl file extension
        for i in range(len(self.meshes)):

            link_entity = None

            if(self.render_number == 1):

                part_mesh = self.meshes[i]

                if(part_mesh == 'pedestal'):
                    mesh = o3d.io.read_triangle_mesh(f'../models/assets/robots/common_meshes/{part_mesh}.stl') # change
                else:
                    mesh = o3d.io.read_triangle_mesh(f'../models/assets/robots/{self.robot_names[0].lower()}/meshes/{part_mesh}.stl') # change
          
                link_name = part_mesh

                # print(f'Succesfully read: {part_mesh}')

                normals  = np.array(mesh.vertex_normals).flatten().tolist()
                vertices = np.array(mesh.vertices).flatten().tolist() 

                mesh = visii.mesh.create_from_data(f'{link_name}_mesh', positions=vertices, normals=normals)

                link_entity = visii.entity.create(
                    name      = link_name,
                    mesh      = mesh,
                    transform = visii.transform.create(link_name),
                    material  = visii.material.create(link_name)
                )

                self.entities.append(link_entity)

            else:
                link_entity = self.entities[i]

            part_position = self.positions[i]

            link_entity.get_transform().set_position(visii.vec3(part_position[0], part_position[1], part_position[2]))

            part_quaternion = self.quats[i]
            link_entity.get_transform().set_rotation(visii.quat(part_quaternion[0],
                                                                part_quaternion[1],
                                                                part_quaternion[2],
                                                                part_quaternion[3]))

            link_entity.get_material().set_base_color(visii.vec3(0.2, 0.2, 0.2))
            link_entity.get_material().set_metallic(0)
            link_entity.get_material().set_transmission(0)
            link_entity.get_material().set_roughness(0.3)

        visii.render_to_png(
            width             = self.width,
            height            = self.height, 
            samples_per_pixel = self.samples_per_pixel,   
            image_path        = f'images/temp_{self.image_counter}.png'
        )

        self.render_number += 1
        
    def quaternion_from_matrix3(self, matrix3):
            """Return quaternion from 3x3 rotation matrix.
            >>> R = rotation_matrix4(0.123, (1, 2, 3))
            >>> q = quaternion_from_matrix4(R)
            >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
            True
            """
            EPS = 1e-6
            q = np.empty((4, ), dtype=np.float64)
            M = np.array(matrix3, dtype=np.float64, copy=False)[:3, :3]
            t = np.trace(M) + 1
            if t <= -EPS:
                warnings.warn('Numerical warning of [t = np.trace(M) + 1 = {}]'\
                        .format(t))
            t = max(t, EPS)
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
            q *= 0.5 / math.sqrt(t)
            return q

    def get_positions(self, parts):

        positions = []

        for part in parts:
            positions.append(np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(part)]))

        return positions

    def get_geoms(self, root):

        geoms = []

        for body in root.iter('body'):
            self.parts.append(body.get('name'))
            for geom in body.findall('geom'):
                geoms.append(geom)

        return geoms

    def get_quats(self):
        quats = []
        for part in self.mesh_parts:
            R = self.sim.data.body_xmat[self.sim.model.body_name2id(part)].reshape(3, 3)
            # print(part, R)
            quat_xyzw = self.quaternion_from_matrix3(R)
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            quats.append(quat_wxyz)

        return quats

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
        """Set camera intrinsics matrix
        """
        raise NotImplementedError

    def set_camera_extrinsics(self):
        """Set camera extrinsics matrix
        """
        raise NotImplementedError
        
        
if __name__ == '__main__':

    # Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
    #                          PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal, 
    #                          PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandoff

    # Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e

    env = VirtualWrapper(
        env = suite.make(
                "Stack",
                robots = "Sawyer",
                reward_shaping=True,
                has_renderer=False,           # no on-screen renderer
                has_offscreen_renderer=False, # no off-screen renderer
                ignore_done=True,
                use_object_obs=True,          # use object-centric feature
                use_camera_obs=False,         # no camera observations
                control_freq=10, 
            ),
        use_noise=False,
    )

    env.reset()

    #env.render(render_type = "png") # initial rendering of the robot

    action = np.random.randn(8)

    for i in range(5):
        obs, reward, done, info = env.step(action)
        #env.printState() # for testing purposes

    env.close()

    print('Done.')
