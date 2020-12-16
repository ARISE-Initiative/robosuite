"""
This file implements a wrapper for using the ViSII imaging interface
for robosuite. 
"""

import numpy as np
import sys
import visii
import robosuite as suite
from mujoco_py import MjSim, load_model_from_path
from robosuite.wrappers import Wrapper
from robosuite.models.robots import Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
from robosuite.models.robots import create_robot
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
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
                 width=1000,
                 height=1000,
                 use_noise=True,
                 debug_mode=False):

        super().__init__(env)

        # print(env.robots[0].gripper.file)
        # quit()

        # Camera variables
        self.width                  = width
        self.height                 = height
        self.samples_per_pixel      = 50
        self.image_counter          = 0
        self.render_number          = 1
        self.entities               = []
        self.gripper_entities       = {}
        self.static_objects         = []
        self.static_object_entities = []

        if debug_mode:
            visii.initialize_interactive()
        else:
            visii.initialize_headless()

        if not use_noise: 
            visii.enable_denoiser()

        light_1 = visii.entity.create(
            name      = "light_1",
            mesh      = visii.mesh.create_sphere("light_1"),
            transform = visii.transform.create("light_1"),
        )

        light_1.set_light(
            visii.light.create("light_1")
        )

        light_1.get_light().set_intensity(200)
        light_1.get_transform().set_scale(visii.vec3(0.3))
        light_1.get_transform().set_position(visii.vec3(3, 3, 4))
        
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
                                  at_vec  = visii.vec3(0, 0, 1), 
                                  up_vec  = visii.vec3(0, 0, 1),
                                  eye_vec = visii.vec3(0.6, 0.0, 1.0))
        
        # Environment configuration
        self._dome_light_intensity = 1
        visii.set_dome_light_intensity(self._dome_light_intensity)

        # visii.set_dome_light_sky(
        #     sun_position=visii.vec3(10, 0, 10),
        #     sky_tint=visii.vec3(1, 0.1, 0.8),
        # )

        visii.set_max_bounce_depth(2)

        self.model = None
        
        self.robots_names = self.env.robot_names

        # idNum = 1
        # for robot in self.robot_names:
        #     robot_model = create_robot(robot, idn = idNum) # only taking the first robot for now
        #     self.robots.append(robot_model)
        #     idNum+=1

        # Passes the xml file based on the robot
        robot_xml_filepath   = f'../models/assets/robots/{self.env.robot_names[0].lower()}/robot.xml'
        gripper_xml_filepath = env.robots[0].gripper.file
        self.initalize_simulation(robot_xml_filepath)

        # TODO (Yifeng): You should be reading these parts' names from
        # the xml files directly

        tree = ET.parse(robot_xml_filepath)
        root = tree.getroot()
        
        tree_gripper = ET.parse(gripper_xml_filepath)
        root_gripper = tree_gripper.getroot()

        self.dynamic_obj_init(root, root_gripper)

        # initialize the static objects
        # self.static_obj_init()

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

    def dynamic_obj_init(self, root, root_gripper):

        self.parts       = []
        self.meshes      = []
        self.geom_names  = {}
        self.mesh_parts  = []
        self.mesh_colors = {}

        # Stores all the meshes and their colors required for the robot
        for body in root.iter('body'):
            self.parts.append(body.get('name'))
            meshBoolean = False
            prev_mesh = None
            for geom in body.findall('geom'):
                geom_mesh  = geom.get('mesh')
                geom_name  = geom.get('name')
                mesh_color = geom.get('rgba')
                if geom_mesh != None:
                    meshBoolean = True
                    self.meshes.append(geom_mesh)
                    self.geom_names[geom_mesh] = geom_name
                    self.mesh_parts.append(body.get('name'))
                    self.mesh_colors[geom_mesh] = None
                    prev_mesh = geom_mesh
                if meshBoolean:
                    self.mesh_colors[prev_mesh] = mesh_color

        print(self.meshes)
        # print(self.mesh_colors)

        # TODO (Yifeng): Try to create a list of objects, one of which
        # contains: geom info, position info, orientation info
        self.positions = self.get_positions(part_type = 'robot', 
                                            parts = self.mesh_parts) # position information for the robot
        self.geoms     = self.get_geoms(root) # geometry information for the robot
        self.quats     = self.get_quats() # orientation information for the robot

        self.gripper_parts      = []
        self.gripper_mesh_types = {}
        self.gripper_mesh_files = {}
        self.gripper_mesh_quats = {}
        self.gripper_colors = {}

        for asset in root_gripper.iter('asset'):

            for mesh in asset.findall('mesh'):
                self.gripper_mesh_files[mesh.get('name')] = mesh.get('file')

        # getting all the meshes and other information for the grippers
        for body in root_gripper.iter('body'):
            self.gripper_parts.append(body.get('name'))
            for geom in body.findall('geom'):
                geom_mesh = geom.get('mesh')
                if geom_mesh != None:
                    geom_quat = geom.get('quat')
                    if geom_quat is None:
                        geom_quat = [1, 0, 0, 0]
                    else:
                        geom_quat = [float(element) for element in geom_quat.split(' ')]
                    if body.get('name') not in self.gripper_mesh_types:
                        self.gripper_mesh_types[body.get('name')] = [geom_mesh]
                        self.gripper_mesh_quats[body.get('name')] = [geom_quat]
                    else:
                        self.gripper_mesh_types[body.get('name')].append(geom_mesh)
                        self.gripper_mesh_quats[body.get('name')].append(geom_quat)

        self.gripper_positions = self.get_positions(part_type = 'gripper', 
                                                    parts = self.gripper_parts)
        self.gripper_quats = self.get_quats_gripper()

        print(f'Gripper Parts: {self.gripper_parts}\n')
        print(f'Gripper Mesh Types: {self.gripper_mesh_types}\n')
        print(f'Gripper STL Files: {self.gripper_mesh_files}\n')
        print(f'Gripper Part Positions: {self.gripper_positions}\n')
        print(f'Gripper Quats: {self.gripper_quats}\n')
        print('-----\n\n')

        # quit()

    def initalize_simulation(self, xml_file = None):
        
        self.mjpy_model = load_model_from_path(xml_file) if xml_file else self.model.get_model(mode="mujoco_py")

        # Creates the simulation
        # self.sim = MjSim(self.mjpy_model)
        # self.sim = self.env.sim
        self.env.sim.step()

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

        # changing the angles to the new angles of the joints
        self.positions = self.get_positions(part_type = 'robot', 
                                            parts = self.mesh_parts)
        self.quats = self.get_quats()

        self.gripper_positions = self.get_positions(part_type = 'gripper', 
                                                    parts = self.gripper_parts)
        self.gripper_quats = self.get_quats_gripper()

    def render(self, render_type):
        """
        Renders the scene
        Arg:
            render_type: tells the method whether to save to png or save to hdr
        """
        self.image_counter += 1
        # Creating a table/base
        if self.render_number == 1:

            table_size_x = self.mujoco_arena.table_full_size[0]/2
            table_size_y = self.mujoco_arena.table_full_size[1]/2
            table_size_z = self.mujoco_arena.table_full_size[2]/2

            table = visii.entity.create(
                name="table",
                mesh = visii.mesh.create_box(name = "table",
                                             size = visii.vec3(table_size_x,
                                                               table_size_y,
                                                               table_size_z)),
                transform = visii.transform.create("table"),
                material = visii.material.create("table")
            )

            table.get_transform().set_position(
                visii.vec3(self.mujoco_arena.center_pos[0], 
                           self.mujoco_arena.center_pos[1], 
                           self.mujoco_arena.center_pos[2]))
            table.get_material().set_base_color(
                visii.vec3(0.1, 0.1, 0.1))  
            table.get_material().set_roughness(0.7)   
            table.get_material().set_specular(1)

        # Initialize the other static objects
        counter = 0
        for i in self.mujoco_objects:

            obj = None

            if self.render_number == 1:

                obj = visii.entity.create(
                    name=f"obj_{i}",
                    mesh = visii.mesh.create_box(name = f"obj_{i}",
                                                 size = visii.vec3(self.mujoco_objects[i].size[0],
                                                                   self.mujoco_objects[i].size[1],
                                                                   self.mujoco_objects[i].size[2])),
                    transform = visii.transform.create(f"obj_{i}"),
                    material = visii.material.create(f"obj_{i}")
                )

                self.static_object_entities.append(obj)

            else:

                obj = self.static_object_entities[counter]

            counter+=1

            obj.get_transform().set_position(visii.vec3(self.obs_dict[f'{i}_pos'][0],
                                                        self.obs_dict[f'{i}_pos'][1],
                                                        self.obs_dict[f'{i}_pos'][2]))

            obj.get_transform().set_rotation(visii.quat(self.obs_dict[f'{i}_quat'][3],
                                                        self.obs_dict[f'{i}_quat'][0],
                                                        self.obs_dict[f'{i}_quat'][1],
                                                        self.obs_dict[f'{i}_quat'][2]))

            obj.get_material().set_base_color(visii.vec3(self.mujoco_objects[i].rgba[0], 
                                                         self.mujoco_objects[i].rgba[1],
                                                         self.mujoco_objects[i].rgba[2]))
            obj.get_material().set_metallic(0.2)
            obj.get_material().set_transmission(0.2)
            obj.get_material().set_roughness(0.3)
            
        # For now we are only rendering it as a png
        # stl file extension
        mesh_color_arr = None
        # print(self.geom_names)
        for i in range(len(self.meshes)):

            link_entity = None
            part_mesh = self.meshes[i]
            part_name = self.geom_names[part_mesh]

            if 'vis' not in part_mesh and 'vis' not in part_name and part_mesh != 'pedestal':
                self.entities.append(None)
                continue

            mesh = None

            if(self.render_number == 1):

                # if(part_mesh == 'pedestal'):
                #     mesh = o3d.io.read_triangle_mesh(f'../models/assets/robots/common_meshes/{part_mesh}.stl') # change

                # else:
                #     # robots/{self.robot_names[0].lower()}
                #     mesh = o3d.io.read_triangle_mesh(f'../models/assets/extra_meshes/{part_mesh}.stl') # change
                
                link_name = part_mesh
                
                # print(f'Succesfully read: {part_mesh}')
                # normals  = np.array(mesh.vertex_normals).flatten().tolist()
                # vertices = np.array(mesh.vertices).flatten().tolist() 

                if(part_mesh == 'pedestal'):
                    mesh = visii.mesh.create_from_obj(name = part_mesh, path = f'../models/assets/robots/common_meshes/{part_mesh}.obj') # change

                else:
                    # robots/{self.robot_names[0].lower()}
                    mesh = visii.mesh.create_from_obj(name = part_mesh, path = f'../models/assets/robots/{self.robot_names[0].lower()}/meshes/{part_mesh}.obj') # change

                # mesh = visii.mesh.create_from_data(f'{link_name}_mesh', positions=vertices, normals=normals)
                link_entity = visii.entity.create(
                    name      = link_name,
                    mesh      = mesh,
                    transform = visii.transform.create(link_name),
                    material  = visii.material.create(link_name)
                )
                
                self.entities.append(link_entity)
            
            else:
                link_entity = self.entities[i]

            # print(self.quats[i])
            part_position = self.positions[i]
            link_entity.get_transform().set_position(visii.vec3(part_position[0], part_position[1], part_position[2]))

            part_quaternion = self.quats[i]
            link_entity.get_transform().set_rotation(visii.quat(part_quaternion[0],
                                                                part_quaternion[1],
                                                                part_quaternion[2],
                                                                part_quaternion[3]))
            # print(link_entity.get_transform().get_rotation())

            if part_mesh in self.mesh_colors and self.mesh_colors[part_mesh] != None:
                mesh_color_arr = self.mesh_colors[part_mesh].split(' ')
                link_entity.get_material().set_base_color(visii.vec3(float(mesh_color_arr[0]),
                                                                     float(mesh_color_arr[1]),
                                                                     float(mesh_color_arr[2])))
            link_entity.get_material().set_metallic(0)
            link_entity.get_material().set_transmission(0)
            link_entity.get_material().set_roughness(0.3)

        # print(self.gripper_positions)
        for key in self.gripper_mesh_types:
            gripper_entity = None

            gripper_mesh_arr = self.gripper_mesh_types[key]
            gripper_mesh_quat = self.gripper_mesh_quats[key]

            for (mesh, mesh_quat) in zip(gripper_mesh_arr, gripper_mesh_quat):
                # if 'vis' not in mesh:
                #     continue
                
                if self.render_number == 1:
                    print(f'rendering... {key} => {mesh}')
                    mesh_gripper = o3d.io.read_triangle_mesh(f'../models/assets/grippers/{self.gripper_mesh_files[mesh]}')
                    mesh_name = f'{key}_{mesh}_mesh'

                    normals  = np.array(mesh_gripper.vertex_normals).flatten().tolist()
                    vertices = np.array(mesh_gripper.vertices).flatten().tolist()
                    mesh_gripper = visii.mesh.create_from_data(mesh_name, positions=vertices, normals=normals)
                    gripper_entity = visii.entity.create(
                        name      = mesh_name,
                        mesh      = mesh_gripper,
                        transform = visii.transform.create(mesh_name),
                        material  = visii.material.create(mesh_name)
                    )

                    self.gripper_entities[f'{key}_{mesh}'] = gripper_entity

                else:
                    gripper_entity = self.gripper_entities[f'{key}_{mesh}']

                part_position = self.gripper_positions[key]
                # print(f'{key} ==> {mesh} ==> {self.gripper_quats[key]}, {mesh_quat}')

                gripper_entity.get_transform().set_position(visii.vec3(part_position[0], part_position[1], part_position[2]))

                part_quaternion = self.gripper_quats[key]
                visii_part_quat = visii.quat(*part_quaternion) * visii.quat(*mesh_quat)
                gripper_entity.get_transform().set_rotation(visii_part_quat)
                # gripper_entity.get_transform().set_rotation(visii.quat(part_quaternion[0],
                #                                                        part_quaternion[1],
                #                                                        part_quaternion[2],
                #                                                        part_quaternion[3]))

                if mesh == 'finger_vis':
                    gripper_entity.get_material().set_base_color(visii.vec3(0.5, 0.5, 0.5))
                
                gripper_entity.get_material().set_metallic(0)
                gripper_entity.get_material().set_transmission(0)
                gripper_entity.get_material().set_roughness(0.3)

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

    def get_positions(self, part_type, parts):

        positions = None

        if part_type == 'robot':
            positions = []
            for part in parts:
                positions.append(np.array(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(f'robot0_{part}')]))
        elif part_type == 'gripper':
            positions = {}
            for part in parts:
                positions[part] = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(f'gripper0_{part}')]
                # print(part)

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
            R = self.env.sim.data.body_xmat[self.env.sim.model.body_name2id(f'robot0_{part}')].reshape(3, 3)
            # print(part, R)
            quat_xyzw = self.quaternion_from_matrix3(R)
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            quats.append(quat_wxyz)

        return quats

    def get_quats_gripper(self):
        quats = {}
        for part in self.gripper_parts:
            R = self.env.sim.data.body_xmat[self.env.sim.model.body_name2id(f'gripper0_{part}')].reshape(3, 3)
            quat_xyzw = self.quaternion_from_matrix3(R)
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            quats[part] = quat_wxyz
            # print(f'gripper0_{part}: ', R, quat_wxyz)

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
                robots = "Panda",
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

    # env.render(render_type = "png") # initial rendering of the robot

    env.printState()

    for i in range(100):
        action = np.random.randn(8)
        obs, reward, done, info = env.step(action)

        if i%100 == 0:
            env.render(render_type = "png")

        # env.printState() # for testing purposes
    
    env.close()
    
    print('Done.')
