import os
import numpy as np
import sys
import visii
import robosuite as suite
from mujoco_py import MjSim, load_model_from_path
from robosuite.wrappers import Wrapper
from robosuite.models.robots import Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
from robosuite.models.robots import create_robot
from robosuite.models.arenas import TableArena, BinsArena, PegsArena, EmptyArena, WipeArena
from robosuite.models.objects import BoxObject, CylinderObject, BallObject, CapsuleObject, DoorObject, SquareNutObject, RoundNutObject
from robosuite.models.objects import CanVisualObject, MilkVisualObject, BreadVisualObject, CerealVisualObject
from robosuite.models.objects import CanObject, MilkObject, BreadObject, CerealObject
from robosuite.models.objects import PotWithHandlesObject, HammerObject
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite.utils.mjcf_utils import CustomMaterial
import open3d as o3d
import math
import xml.etree.ElementTree as ET

class VirtualWrapper(Wrapper):
    """Initializes the ViSII wrapper.
    Args:
        env (MujocoEnv instance): The environment to wrap.
        width (int, optional)
        height (int, optional)
        use_noise (bool, optional): Use noise or denoise
        debug_mode (bool, optional): Use debug mode for visii
        spp (int, optional): sample-per-pixel for each image
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

        # Image variables
        self.image_counter = 0

        # Entity variables
        self.entities               = []
        self.gripper_entities       = {}
        self.static_object_entities = {}

        # Mesh and entity variables for robot
        self.meshes      = []
        self.mesh_parts  = []
        self.geom_names  = {}
        self.mesh_colors = {}

        # Mesh parts for the gripper
        self.gripper_parts      = {}
        self.gripper_mesh_files = {}

        # Extra variables
        self.extra_entities   = {}
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

        light_1.get_light().set_intensity(200)
        light_1.get_transform().set_scale(visii.vec3(0.3))
        light_1.get_transform().set_position(visii.vec3(3, 3, 4))
        
        # Intiailizes the floor
        floor_mesh = visii.mesh.create_plane(name = "plane",
                                             size = visii.vec2(3, 3))

        floor = visii.entity.create(
            name      = "floor",
            mesh      = floor_mesh,
            material  = visii.material.create("plane"),
            transform = visii.transform.create("plane")
        )
        floor.get_transform().set_scale(visii.vec3(1))
        floor.get_transform().set_position(visii.vec3(0, 0, 0))
        floor.get_material().set_roughness(0.4)
        floor.get_material().set_specular(0)

        image = '../models/assets/textures/kitchen_wood_diff_4k.jpg'
        texture = visii.texture.create_from_file(name = 'flooring_texture',
                                                 path = image)

        floor.get_material().set_base_color_texture(texture)

        image = '../models/assets/textures/grey_plaster_rough_4k.jpg'
        texture = visii.texture.create_from_file(name = 'walls_texture',
                                                 path = image)
        
        # Intiailizes the walls
        wall_count = 0
        for wall in self.env.model.mujoco_arena.worldbody.findall("./geom[@material='walls_mat']"):

            name = wall.get('name')
            size = [float(x) for x in wall.get('size').split(' ')]

            pos = self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(name)]
            R = self.env.sim.data.geom_xmat[self.env.sim.model.geom_name2id(name)].reshape(3, 3)

            quat_xyzw = self.quaternion_from_matrix3(R)
            quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

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
            wall_entity.get_material().set_base_color_texture(texture)

            wall_count+=1


        self.camera_init()

        # Sets the primary camera to the renderer to the camera entity
        visii.set_camera_entity(self.camera) 

        self.camera_configuration(pos_vec = visii.vec3(0, 0, 1), 
                                  at_vec  = visii.vec3(0, 0, 1), 
                                  up_vec  = visii.vec3(0, 0, 1),
                                  eye_vec = visii.vec3(1.8, 0, 1.25))
        
        # Environment configuration
        self._dome_light_intensity = 1
        visii.set_dome_light_intensity(self._dome_light_intensity)

        visii.set_max_bounce_depth(2)

        robot_count = 0
        for robot in self.env.robots:

            name = robot.name

            robot_xml_filepath   = f'../models/assets/robots/{name.lower()}/robot.xml'
            gripper_xml_filepath = robot.gripper.file

            self.initalize_simulation(robot_xml_filepath)

            tree = ET.parse(robot_xml_filepath)
            root = tree.getroot()
            
            tree_gripper = ET.parse(gripper_xml_filepath)
            root_gripper = tree_gripper.getroot()

            self.dynamic_obj_init(root, root_gripper, robot_count)

            robot_count += 1

        self.init_objects()

    def dynamic_obj_init(self, root, root_gripper, robot_num):

        count = 0
        # Stores all the meshes and their colors required for the robot
        for body in root.iter('body'):
            body_name = body.get('name')
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
                    self.mesh_parts.append(f'robot{robot_num}_{body_name}')
                    self.mesh_colors[geom_mesh] = None
                    prev_mesh = geom_mesh
                    count+=1
                if meshBoolean:
                    self.mesh_colors[prev_mesh] = mesh_color

        self.num_meshes_per_robot.append(count)

        self.positions = self.get_positions(part_type = 'robot', 
                                            parts = self.mesh_parts,
                                            robot_count = robot_num) # position information for the robot
        self.geoms     = self.get_geoms(root)                        # geometry information for the robot
        self.quats     = self.get_quats(part_type = 'robot')         # orientation information for the robot

        for asset in root_gripper.iter('asset'):

            for mesh in asset.findall('mesh'):
                self.gripper_mesh_files[mesh.get('name')] = mesh.get('file')

        # getting all the meshes and other information for the grippers
        name = None
        for body in root_gripper.iter('body'):
            mesh_types = []
            mesh_quats = []
            name = body.get('name')

            for geom in body.findall('geom'):
                geom_mesh = geom.get('mesh')
                if geom_mesh != None:
                    geom_quat = geom.get('quat')
                    if geom_quat is None:
                        geom_quat = [1, 0, 0, 0]
                    else:
                        geom_quat = [float(element) for element in geom_quat.split(' ')]

                    mesh_types.append(geom_mesh)
                    mesh_quats.append(geom_quat)

            self.gripper_parts[f'gripper{robot_num}_{name}'] = (mesh_types, mesh_quats)

        self.gripper_positions = self.get_positions(part_type = 'gripper', 
                                                    parts = self.gripper_parts,
                                                    robot_count = robot_num)
        self.gripper_quats = self.get_quats(part_type = 'gripper')

    def get_positions(self, part_type, parts, robot_count):

        positions = None

        if part_type == 'robot':
            positions = []
            for part in parts:
                positions.append(np.array(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(part)]))
        elif part_type == 'gripper':
            positions = {}
            for part in parts:
                positions[part] = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(part)]

        return positions

    def get_geoms(self, root):

        geoms = []

        for body in root.iter('body'):
            # self.parts.append(body.get('name'))
            for geom in body.findall('geom'):
                geoms.append(geom)

        return geoms

    def get_quats(self, part_type):

        quats = None
        parts = None
        if part_type == 'robot':
            quats = []
            parts = self.mesh_parts
        elif part_type == 'gripper':
            quats = {}
            parts = self.gripper_parts

        for part in parts:
            R = self.env.sim.data.body_xmat[self.env.sim.model.body_name2id(part)].reshape(3, 3)
            quat_xyzw = self.quaternion_from_matrix3(R)
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            if part_type == 'robot':
                quats.append(quat_wxyz)
            elif part_type == 'gripper':
                quats[part] = quat_wxyz

        return quats

    def initalize_simulation(self, xml_file = None):
        
        # Load the model from the xml file
        self.mjpy_model = load_model_from_path(xml_file) if xml_file else self.model.get_model(mode="mujoco_py")

        # Creates the simulation
        self.env.sim.step()

    def camera_init(self):

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

    def camera_configuration(self, pos_vec, at_vec, up_vec, eye_vec):

        # configures the camera
        self.camera.get_transform().set_position(pos_vec)

        self.camera.get_transform().look_at(
            at  = at_vec, # look at (world coordinate)
            up  = up_vec, # up vector
            eye = eye_vec,
            previous = False
        )

    def init_objects(self):

        self.init_arena()
        self.init_static_objects()
        self.init_robot_parts()
        self.init_gripper_parts()

    def init_arena(self):

        table = None

        self.mujoco_arena = self.env.model.mujoco_arena

        if isinstance(self.mujoco_arena, TableArena):

            table_pos = self.mujoco_arena.table_body.get('pos').split(' ')
            
            table_size_x = self.mujoco_arena.table_half_size[0]
            table_size_y = self.mujoco_arena.table_half_size[1]
            table_size_z = self.mujoco_arena.table_half_size[2]

            table = visii.entity.create(
                        name = "table",
                        mesh = visii.mesh.create_box(name = "table",
                                                     size = visii.vec3(table_size_x,
                                                                       table_size_y,
                                                                       table_size_z)),
                        transform = visii.transform.create("table"),
                        material = visii.material.create("table")
                    )

            image = '../models/assets/textures/ceramic.png'
            texture = visii.texture.create_from_file(name = 'ceramic_table_texture',
                                                     path = image)

            table.get_material().set_base_color_texture(texture)

            table.get_transform().set_position(visii.vec3(float(table_pos[0]),
                                                          float(table_pos[1]),
                                                          float(table_pos[2])))
            
            self.set_entity_material(table)

            # iterate through the legs of the table
            leg_count = 1
            image = '../models/assets/textures/steel-brushed.png'
            texture = visii.texture.create_from_file(name = 'steel_legs_texture',
                                                     path = image)

            for leg in self.mujoco_arena.table_legs_visual:

                size = leg.get('size').split(' ')
                relative_pos = leg.get('pos').split(' ')

                leg_entity = visii.entity.create(
                        name = f"table_leg_{leg_count}",
                        mesh = visii.mesh.create_cylinder(name   = f"table_leg_{leg_count}",
                                                          radius = float(size[0]),
                                                          size   = float(size[1])),
                        transform = visii.transform.create(f"table_leg_{leg_count}"),
                        material = visii.material.create(f"table_leg_{leg_count}")
                    )

                leg_entity.get_material().set_base_color_texture(texture)

                pos = []
                for i in range(len(relative_pos)):
                    pos.append(float(relative_pos[i]) + float(table_pos[i]))

                leg_entity.get_transform().set_position(visii.vec3(pos[0], 
                                                                   pos[1],
                                                                   pos[2]))

                self.set_entity_material(leg_entity)

                leg_count += 1

            if isinstance(self.mujoco_arena, WipeArena):

                image = '../models/assets/textures/dirt.png'
                texture_dirt = visii.texture.create_from_file(name = 'dirt_texture',
                                                               path = image)

                for sensor in range(self.mujoco_arena.num_markers):
                    # self.sim.model.geom_name2id(sensor_name)

                    sensor_name = f'contact{sensor}_g0_vis'

                    if self.env.sim.model.geom_name2id(sensor_name) not in self.env.wiped_markers:
                        
                        pos = np.array(self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(sensor_name)])

                        radius = self.mujoco_arena.line_width / 2
                        half_length = 0.001

                        obj_mesh = visii.mesh.create_cylinder(name   = sensor_name,
                                                              radius = radius,
                                                              size   = half_length)

                        dirt_entity = visii.entity.create(
                            name=sensor_name,
                            mesh = obj_mesh,
                            transform = visii.transform.create(sensor_name),
                            material = visii.material.create(sensor_name)
                        )

                        dirt_entity.get_transform().set_position(visii.vec3(pos[0],
                                                                            pos[1],
                                                                            pos[2]))

                        dirt_entity.get_material().set_base_color_texture(texture_dirt)

                        self.set_entity_material(dirt_entity)

            if isinstance(self.mujoco_arena, PegsArena):

                geom_count = 0

                image = '../models/assets/textures/brass-ambra.png'
                texture = visii.texture.get(image)

                texture = visii.texture.create_from_file(name = 'peg1_texture',
                                                          path = image)

                pos = [float(x) for x in self.mujoco_arena.peg1_body.get('pos').split(' ')]
                geom = self.mujoco_arena.peg1_body.find("./geom[@group='1']")
                size = [float(x) for x in geom.get('size').split(' ')]
                name = f'peg1'

                obj_mesh = visii.mesh.create_box(name = name,
                                                 size = visii.vec3(size[0],
                                                                   size[1],
                                                                   size[2]))

                obj_entity = visii.entity.create(
                    name=name,
                    mesh = obj_mesh,
                    transform = visii.transform.create(name),
                    material = visii.material.create(name)
                )

                obj_entity.get_transform().set_position(visii.vec3(pos[0],
                                                                   pos[1],
                                                                   pos[2]))
                obj_entity.get_material().set_base_color_texture(texture)

                self.set_entity_material(obj_entity)

                image = '../models/assets/textures/steel-scratched.png'
                texture = visii.texture.get(image)

                texture = visii.texture.create_from_file(name = 'peg2_texture',
                                                          path = image)

                pos = [float(x) for x in self.mujoco_arena.peg2_body.get('pos').split(' ')]
                geom = self.mujoco_arena.peg1_body.find("./geom[@group='1']")
                size = [float(x) for x in geom.get('size').split(' ')]
                name = f'peg2'

                obj_mesh = visii.mesh.create_box(name = name,
                                                 size = visii.vec3(size[0],
                                                                   size[1],
                                                                   size[2]))

                obj_entity = visii.entity.create(
                    name = name,
                    mesh = obj_mesh,
                    transform = visii.transform.create(name),
                    material = visii.material.create(name)
                )

                obj_entity.get_transform().set_position(visii.vec3(pos[0],
                                                                   pos[1],
                                                                   pos[2]))

                obj_entity.get_material().set_base_color_texture(texture)

                self.set_entity_material(obj_entity)

        elif isinstance(self.mujoco_arena, BinsArena):

            base_pos = self.mujoco_arena.bin1_body.get('pos').split(' ')

            wall_count = 1
            image = '../models/assets/textures/light-wood.png'
            texture_light_table = visii.texture.create_from_file(name = 'light-wood_table_texture',
                                                                 path = image)
            for wall in self.mujoco_arena.bin1_body.findall("./geom[@material='light-wood']"):

                size = wall.get('size').split(' ')

                table_size_x = float(size[0])
                table_size_y = float(size[1])
                table_size_z = float(size[2])

                table = visii.entity.create(
                            name = f'wall_light_{wall_count}',
                            mesh = visii.mesh.create_box(name = f'wall_light_{wall_count}',
                                                         size = visii.vec3(table_size_x,
                                                                           table_size_y,
                                                                           table_size_z)),
                            transform = visii.transform.create(f'wall_light_{wall_count}'),
                            material = visii.material.create(f'wall_light_{wall_count}')
                        )

                table.get_material().set_base_color_texture(texture_light_table)

                self.set_entity_material(table)

                relative_pos = wall.get('pos').split(' ')
                table_pos = []
                for i in range(len(relative_pos)):
                    table_pos.append(float(base_pos[i]) + float(relative_pos[i]))

                table.get_transform().set_position(visii.vec3(table_pos[0],
                                                              table_pos[1],
                                                              table_pos[2]))

                wall_count+=1

            # iterate through the legs of the table
            leg_count = 1
            image = '../models/assets/textures/steel-brushed.png'
            texture_legs = visii.texture.create_from_file(name = 'steel_legs_texture',
                                                      path = image)

            for leg in self.mujoco_arena.bin1_body.findall("./geom[@material='table_legs_metal']"):

                size = leg.get('size').split(' ')
                pos  = leg.get('pos').split(' ')

                leg_entity = visii.entity.create(
                        name = f"table_light_leg_{leg_count}",
                        mesh = visii.mesh.create_cylinder(name   = f"table_light_leg_{leg_count}",
                                                          radius = float(size[0]),
                                                          size   = (float(size[1]))),
                        transform = visii.transform.create(f"table_light_leg_{leg_count}"),
                        material = visii.material.create(f"table_light_leg_{leg_count}")
                    )

                leg_entity.get_material().set_base_color_texture(texture_legs)

                self.set_entity_material(leg_entity)

                leg_pos = []
                for i in range(len(pos)):
                    leg_pos.append(float(base_pos[i]) + float(pos[i]))

                leg_entity.get_transform().set_position(visii.vec3(leg_pos[0], 
                                                                   leg_pos[1], 
                                                                   leg_pos[2]))

                leg_count += 1

            base_pos = self.mujoco_arena.bin2_body.get('pos').split(' ')

            wall_count = 1
            image = '../models/assets/textures/dark-wood.png'
            texture_dark_table = visii.texture.create_from_file(name = 'dark-wood_table_texture',
                                                                path = image)
            for wall in self.mujoco_arena.bin2_body.findall("./geom[@material='dark-wood']"):

                size = wall.get('size').split(' ')

                table_size_x = float(size[0])
                table_size_y = float(size[1])
                table_size_z = float(size[2])

                table = visii.entity.create(
                            name = f'wall_dark_{wall_count}',
                            mesh = visii.mesh.create_box(name = f'wall_dark_{wall_count}',
                                                         size = visii.vec3(table_size_x,
                                                                           table_size_y,
                                                                           table_size_z)),
                            transform = visii.transform.create(f'wall_dark_{wall_count}'),
                            material = visii.material.create(f'wall_dark_{wall_count}')
                        )

                table.get_material().set_base_color_texture(texture_dark_table)

                relative_pos = wall.get('pos').split(' ')
                table_pos = []
                for i in range(len(relative_pos)):
                    table_pos.append(float(base_pos[i]) + float(relative_pos[i]))

                table.get_transform().set_position(visii.vec3(table_pos[0],
                                                              table_pos[1],
                                                              table_pos[2]))

                self.set_entity_material(table)

                wall_count+=1

            # iterate through the legs of the table
            leg_count = 1
            image = '../models/assets/textures/steel-brushed.png'

            for leg in self.mujoco_arena.bin2_body.findall("./geom[@material='table_legs_metal']"):

                size = leg.get('size').split(' ')
                pos  = leg.get('pos').split(' ')

                leg_entity = visii.entity.create(
                        name = f"table_dark_leg_{leg_count}",
                        mesh = visii.mesh.create_cylinder(name   = f"table_dark_leg_{leg_count}",
                                                          radius = float(size[0]),
                                                          size   = (float(size[1]) * 2)),
                        transform = visii.transform.create(f"table_dark_leg_{leg_count}"),
                        material = visii.material.create(f"table_dark_leg_{leg_count}")
                    )

                leg_entity.get_material().set_base_color_texture(texture_legs)

                leg_pos = []
                for i in range(len(pos)):
                    leg_pos.append(float(base_pos[i]) + float(pos[i]))

                leg_entity.get_transform().set_position(visii.vec3(float(leg_pos[0]), 
                                                                   float(leg_pos[1]), 
                                                                   0))

                leg_entity.get_material().set_metallic(0)
                leg_entity.get_material().set_transmission(0)
                leg_entity.get_material().set_roughness(0.3)

                self.set_entity_material(leg_entity)

                leg_count += 1

    def init_static_objects(self):

        self.mujoco_objects = self.env.model.mujoco_objects

        # initializing the objects
        for static_object in self.mujoco_objects:

            static_object_name = static_object.name

            # first check if the environment is in single object mode
            single_obj_env = False

            if isinstance(self.env, PickPlace) and self.env.single_object_mode == 2:
                single_obj_env = True
                obj_id = self.env.object_id
                for key, value in self.env.object_to_id.items():
                    if obj_id == value:
                        obj = key
            
                static_object_name = obj[0].upper() + obj[1:] + '0'
                static_object = self.mujoco_objects[static_object_name]

            tuple_obj = True
            obj_entities = None # for the most part, obj_entities is only going to be a singluar mesh

            if isinstance(static_object, BoxObject):

                obj_mesh = visii.mesh.create_box(name = static_object_name,
                                                 size = visii.vec3(static_object.size[0],
                                                                   static_object.size[1],
                                                                   static_object.size[2]))

                obj_entity = self.create_entity(static_object_name, obj_mesh)

                image = static_object.material.tex_attrib["file"]
                texture = visii.texture.get(static_object.material.tex_attrib["name"])

                if texture == None:
                    texture = visii.texture.create_from_file(name = static_object.material.tex_attrib["name"],
                                                              path = image)

                obj_entity.get_material().set_base_color_texture(texture)

                self.set_entity_material(obj_entity)

                self.static_object_entities[static_object_name] = [obj_entity]

            elif isinstance(static_object, CylinderObject):
                obj_mesh = visii.mesh.create_cylinder(name   = static_object_name,
                                                      radius = static_object.size[0],
                                                      size   = static_object.size[1])

                obj_entity = self.create_entity(static_object_name, obj_mesh)

                image = static_object.material.tex_attrib["file"]
                texture = visii.texture.get(static_object.material.tex_attrib["name"])

                if texture == None:
                    texture = visii.texture.create_from_file(name = static_object.material.tex_attrib["name"],
                                                              path = image)

                obj_entity.get_material().set_base_color_texture(texture)

                self.set_entity_material(obj_entity)

                self.static_object_entities[static_object_name] = [obj_entity]

            elif isinstance(static_object, BallObject):
                obj_mesh = visii.mesh.create_sphere(name   = static_object_name,
                                                    radius = static_object.size[0])

                obj_entity = self.create_entity(static_object_name, obj_mesh)

                image = static_object.material.tex_attrib["file"]
                texture = visii.texture.get(static_object.material.tex_attrib["name"])

                if texture == None:
                    texture = visii.texture.create_from_file(name = static_object.material.tex_attrib["name"],
                                                              path = image)

                obj_entity.get_material().set_base_color_texture(texture)

                self.set_entity_material(obj_entity)

                self.static_object_entities[static_object_name] = [obj_entity]

            elif isinstance(static_object, CapsuleObject):
                obj_mesh = visii.mesh.create_capsule(name   = static_object_name,
                                                     radius = static_object.size[0],
                                                     size   = static_object.size[1])

                obj_entity = self.create_entity(static_object_name, obj_mesh)

                image = static_object.material.tex_attrib["file"]
                texture = visii.texture.get(static_object.material.tex_attrib["name"])

                if texture == None:
                    texture = visii.texture.create_from_file(name = static_object.material.tex_attrib["name"],
                                                              path = image)

                obj_entity.get_material().set_base_color_texture(texture)

                self.static_object_entities[static_object_name] = [obj_entity]

            elif isinstance(static_object, MilkObject):
                obj_file = '../models/assets/objects/meshes/milk.obj'

                obj_entity = visii.import_obj(
                            static_object_name,
                            obj_file,
                            '../models/assets/objects/meshes/'
                        )

                self.static_object_entities[static_object_name] = [obj_entity]

            elif isinstance(static_object, CanObject):
                obj_file = '../models/assets/objects/meshes/can.obj'

                obj_entity = visii.import_obj(
                            static_object_name,
                            obj_file,
                            '../models/assets/objects/meshes/'
                        )

                self.static_object_entities[static_object_name] = [obj_entity]

            elif isinstance(static_object, BreadObject):
                obj_file = '../models/assets/objects/meshes/bread.obj'

                obj_entity = visii.import_obj(
                            static_object_name,
                            obj_file,
                            '../models/assets/objects/meshes/'
                        )

                self.static_object_entities[static_object_name] = [obj_entity]

            elif isinstance(static_object, CerealObject):
                obj_file = '../models/assets/objects/meshes/cereal.obj'

                obj_entity = visii.import_obj(
                            static_object_name,
                            obj_file,
                            '../models/assets/objects/meshes/'
                        )

                self.static_object_entities[static_object_name] = [obj_entity]

            elif isinstance(static_object, DoorObject):

                static_object_list = []

                frame = static_object.worldbody.find("./body/body/body[@name='Door_frame']")

                door = frame.find("./body[@name='Door_door']")  
                door_geom = door.find("./geom[@name='Door_panel']")

                door_size = [float(x) for x in door_geom.get('size').split(' ')]

                obj_mesh = visii.mesh.create_box(name = door_geom.get('name'),
                                                 size = visii.vec3(door_size[0],
                                                                   door_size[1],
                                                                   door_size[2]))

                obj_entity = self.create_entity(door_geom.get('name'), obj_mesh)

                static_object_list.append((door_geom.get('name'), obj_entity))

                image = '../models/assets/textures/dark-wood.png'
                texture = visii.texture.get(image)

                if texture == None:
                    texture = visii.texture.create_from_file(name = 'door_texture',
                                                              path = image)

                obj_entity.get_material().set_base_color_texture(texture)

                # first get the left and right frames
                for f in frame.findall("./geom[@type='cylinder']"):
                    
                    frame_size = [float(x) for x in f.get('size').split(' ')]

                    obj_mesh = visii.mesh.create_cylinder(name   = f.get('name'),
                                                          radius = frame_size[0],
                                                          size   = frame_size[1])

                    obj_entity = self.create_entity(f.get('name'), obj_mesh)

                    static_object_list.append((f.get('name'), obj_entity))

                latch = frame.find("./body/body[@name='Door_latch']")

                image = '../models/assets/textures/brass-ambra.png'
                texture = visii.texture.get(image)

                if texture == None:
                    texture = visii.texture.create_from_file(name = 'handle_texture',
                                                              path = image)

                for f in latch.findall("geom"):
                    
                    size = [float(x) for x in f.get('size').split(' ')]

                    obj_mesh = None

                    if f.get('type') == 'cylinder':

                        radius = size[0]
                        length = None
                        if len(size) == 1:
                            fromto = [float(x) for x in f.get('fromto').split(' ')]
                            length = (fromto[1] - fromto[4]) / 2

                        else:
                            length = size[1]

                        obj_mesh = visii.mesh.create_cylinder(name   = f.get('name'),
                                                              radius = radius,
                                                              size   = length)

                    else:

                        obj_mesh = visii.mesh.create_box(name = f.get('name'),
                                                         size = visii.vec3(size[0],
                                                                           size[1],
                                                                           size[2]))

                    obj_entity = self.create_entity(f.get('name'), obj_mesh)

                    obj_entity.get_material().set_base_color_texture(texture)

                    static_object_list.append((f.get('name'), obj_entity))

                self.static_object_entities[static_object_name] = static_object_list

            elif isinstance(static_object, SquareNutObject):

                collision = static_object.worldbody.find("./body/body")

                static_object_list = []

                geom_count = 0

                image = '../models/assets/textures/brass-ambra.png'
                texture = visii.texture.get(image)

                texture = visii.texture.create_from_file(name = 'square_nut_texture',
                                                          path = image)

                for geom in collision.findall('geom'):

                    size = [float(x) for x in geom.get('size').split(' ')]
                    name = f'SquareNut_g{geom_count}'

                    obj_mesh = visii.mesh.create_box(name = name,
                                                     size = visii.vec3(size[0],
                                                                       size[1],
                                                                       size[2]))

                    obj_entity = self.create_entity(name, obj_mesh)

                    obj_entity.get_material().set_base_color_texture(texture)

                    geom_count+=1
                    static_object_list.append((name, obj_entity))

                self.static_object_entities[static_object_name] = static_object_list

            elif isinstance(static_object, RoundNutObject):

                collision = static_object.worldbody.find("./body/body")
                static_object_list = []

                geom_count = 0

                image = '../models/assets/textures/steel-scratched.png'
                texture = visii.texture.get(image)

                texture = visii.texture.create_from_file(name = 'round_nut_texture',
                                                          path = image)

                for geom in collision.findall('geom'):

                    size = [float(x) for x in geom.get('size').split(' ')]
                    name = f'RoundNut_g{geom_count}'

                    obj_mesh = visii.mesh.create_box(name = name,
                                                     size = visii.vec3(size[0],
                                                                       size[1],
                                                                       size[2]))

                    obj_entity = self.create_entity(name, obj_mesh)

                    obj_entity.get_material().set_base_color_texture(texture)

                    geom_count+=1
                    static_object_list.append((name, obj_entity))

                self.static_object_entities[static_object_name] = static_object_list

            elif isinstance(static_object, PotWithHandlesObject) or isinstance(static_object, HammerObject):

                obj_dict = static_object._get_geom_attrs()
                static_object_list = []

                # getting all the geoms for the main body
                geom_count = 0
            
                num_objects = len(obj_dict['geom_names'])

                for i in range(num_objects):

                    size = obj_dict['geom_sizes'][i]
                    name = static_object_name + '_' + obj_dict['geom_names'][i]
                    geom_type = obj_dict['geom_types'][i]
                    geom_mat = obj_dict['geom_materials'][i]

                    if geom_mat == 'pot_mat':
                        image = '../models/assets/textures/red-wood.png'
                    elif geom_mat == 'handle0_mat':
                        image = '../models/assets/textures/green-wood.png'
                    elif geom_mat == 'handle1_mat':
                        image = '../models/assets/textures/blue-wood.png'
                    elif geom_mat == 'metal_mat':
                        image = '../models/assets/textures/steel-scratched.png'
                    elif geom_mat == 'wood_mat':
                        image = '../models/assets/textures/light-wood.png'

                    texture_name = geom_mat + '_texture'

                    texture = visii.texture.get(texture_name)

                    if texture == None:
                        texture = visii.texture.create_from_file(name = texture_name,
                                                                  path = image)

                    if geom_type == 'box':

                        obj_mesh = visii.mesh.create_box(name = name,
                                                         size = visii.vec3(size[0],
                                                                           size[1],
                                                                           size[2]))

                        obj_entity = self.create_entity(name, obj_mesh)

                        obj_entity.get_material().set_base_color_texture(texture)

                    elif geom_type == 'cylinder':

                        obj_mesh = visii.mesh.create_cylinder(name = name,
                                                              radius = size[0],
                                                              size = size[1])

                        obj_entity = self.create_entity(name, obj_mesh)

                        obj_entity.get_material().set_base_color_texture(texture)

                    static_object_list.append((static_object_name, obj_entity, 'geom'))
                    geom_count += 1

                self.static_object_entities[static_object_name] = static_object_list
            
            else:
                print(f'Not an object: {type(static_object)}')
                continue

            if single_obj_env:
                break

    def init_robot_parts(self):

        for i in range(len(self.env.robots)):

            link_entity = visii.import_obj(
                                f'pedestal_{i}',
                                f'../models/assets/mounts/meshes/rethink_mount/pedestal.obj',
                                f'../models/assets/mounts/meshes/rethink_mount/',
                            )
            self.set_entity_material(link_entity)

            for link_idx in range(len(link_entity)):

                entity = link_entity[link_idx]

                pos = np.array(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(f'mount{i}_base')])
                R = self.env.sim.data.body_xmat[self.env.sim.model.body_name2id(f'mount{i}_base')].reshape(3, 3)

                quat_xyzw = self.quaternion_from_matrix3(R)
                quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                entity.get_transform().set_position(visii.vec3(pos[0],
                                                               pos[1],
                                                               pos[2]))
                
                entity.get_transform().set_rotation(visii.quat(quat[0],
                                                               quat[1],
                                                               quat[2],
                                                               quat[3]))
                self.set_entity_material(entity)

        mesh_count = 1
        current_robot_num = 0
        for i in range(len(self.meshes)):

            if mesh_count > self.num_meshes_per_robot[current_robot_num]:
                mesh_count = 1
                current_robot_num+=1

            current_robot_name = self.robot_names[current_robot_num]

            link_entity = None
            part_mesh = self.meshes[i]
            part_name = self.geom_names[part_mesh]

            if 'vis' not in part_mesh and 'vis' not in part_name and part_mesh != 'pedestal':
                self.entities.append(None)
                mesh_count += 1
                continue

            mesh = None

            link_name = part_mesh

            if(part_mesh == 'pedestal'):
                mesh = visii.mesh.create_from_obj(name = part_mesh, path = f'../models/assets/mounts/meshes/{part_mesh}.obj') # change
                link_entity = visii.entity.create(
                    name      = link_name,
                    mesh      = mesh,
                    transform = visii.transform.create(link_name),
                    material  = visii.material.create(link_name)
                )

            else:
                obj_file = f'../models/assets/robots/{current_robot_name.lower()}/meshes/{part_mesh}.obj'
                mtl_file = obj_file.replace('obj', 'mtl')
                
                if os.path.exists(mtl_file):
                    entity_imported = visii.import_obj(
                        part_mesh,
                        obj_file,
                        '/'.join(obj_file.split('/')[:-1]) + '/',
                    )
                    link_entity = entity_imported
                
                else:
                    
                    mesh = visii.mesh.create_from_obj(name = part_mesh, path = f'../models/assets/robots/{current_robot_name}/meshes/{part_mesh}.obj') # change

                    link_entity = visii.entity.create(
                        name      = link_name,
                        mesh      = mesh,
                        transform = visii.transform.create(link_name),
                        material  = visii.material.create(link_name)
                    )

            self.set_entity_material(link_entity)

            self.entities.append(link_entity)
            mesh_count += 1

    def init_gripper_parts(self):

        for key in self.gripper_parts.keys():

            gripper_entity = None

            gripper_mesh_arr = self.gripper_parts[key][0]
            gripper_mesh_quat = self.gripper_parts[key][1]

            if key == 'wiping_gripper':

                xml_file = f'../models/assets/grippers/{key}.xml'

                wipe_xml = ET.parse(xml_file)
                wipe_xml_root = wipe_xml.getroot()

                for wipe_geom in wipe_xml_root.find('worldbody').find('body').findall('geom'):

                    size = [float(x) for x in wipe_geom.get('size').split(' ')]

                    if wipe_geom.get('type') == 'box':
                        obj_mesh = visii.mesh.create_box(name = wipe_geom.get('name'),
                                                         size = visii.vec3(size[0],
                                                                           size[1],
                                                                           size[2]))
                    elif wipe_geom.get('type') == 'sphere':
                        obj_mesh = visii.mesh.create_sphere(name = wipe_geom.get('name'),
                                                            radius = size[0])

                    obj_entity = visii.entity.create(
                        name=wipe_geom.get('name'),
                        mesh = obj_mesh,
                        transform = visii.transform.create(wipe_geom.get('name')),
                        material = visii.material.create(wipe_geom.get('name'))
                    )

                    obj_entity.get_material().set_base_color(visii.vec3(0.25,0.25,0.25))

                    self.gripper_entities[wipe_geom.get('name')] = obj_entity

            for (mesh, mesh_quat) in zip(gripper_mesh_arr, gripper_mesh_quat):

                gripper_entity = None

                stl_file = f'../models/assets/grippers/{self.gripper_mesh_files[mesh]}'
                obj_file = stl_file.replace('stl', 'obj')
                mtl_file = stl_file.replace('stl', 'mtl')

                mesh_name = f'{key}_{mesh}_mesh'

                if os.path.exists(mtl_file):
                    
                    gripper_entity = visii.import_obj(
                                        mesh_name,
                                        obj_file,
                                        '/'.join(obj_file.split('/')[:-1]) + '/',
                                    )

                else:

                    mesh_gripper = o3d.io.read_triangle_mesh(stl_file)

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

        if isinstance(self.env, TwoArmPegInHole):

            hole_xml = self.env.hole.worldbody
            object_tag = hole_xml.find("./body/body[@name='hole_object']")

            geom_count = 0
            image = '../models/assets/textures/red-wood.png'
            texture_name = 'hole_texture'
            texture = visii.texture.get(texture_name)

            if texture == None:
                texture = visii.texture.create_from_file(name = texture_name,
                                                          path = image)
            for geom in object_tag.findall('geom'):

                size = [float(x) for x in geom.get('size').split(' ')]
                name = f'hole_g{geom_count}_visual'

                obj_mesh = visii.mesh.create_box(name = name,
                                                 size = visii.vec3(size[0],
                                                                   size[1],
                                                                   size[2]))

                obj_entity = visii.entity.create(
                    name=name,
                    mesh = obj_mesh,
                    transform = visii.transform.create(name),
                    material = visii.material.create(name)
                )

                obj_entity.get_material().set_base_color_texture(texture)
                geom_name = f'hole_g{geom_count}_visual'
                self.extra_entities[geom_name] = obj_entity

                geom_count+=1

            peg_name = self.env.peg.name + '_g0_vis'
            peg_size = self.env.peg.size

            obj_mesh = visii.mesh.create_cylinder(name   = peg_name,
                                                  radius = peg_size[0],
                                                  size   = peg_size[1])

            obj_entity = visii.entity.create(
                name=peg_name,
                mesh = obj_mesh,
                transform = visii.transform.create(peg_name),
                material = visii.material.create(peg_name)
            )

            texture_file = self.env.peg.material.tex_attrib["file"]
            image = texture_file
            texture = visii.texture.get(texture_file)

            if texture == None:
                texture = visii.texture.create_from_file(name = self.env.peg.material.tex_attrib["name"],
                                                          path = image)

            obj_entity.get_material().set_base_color_texture(texture)

            self.extra_entities[peg_name] = obj_entity

    def create_entity(self, name, mesh):
        return visii.entity.create(
                        name=name,
                        mesh = mesh,
                        transform = visii.transform.create(name),
                        material = visii.material.create(name)
                    )

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
                                            parts = self.mesh_parts,
                                            robot_count = 0)
        self.quats = self.get_quats(part_type = 'robot')

        self.gripper_positions = self.get_positions(part_type = 'gripper', 
                                                    parts = self.gripper_parts.keys(),
                                                    robot_count = 0)
        self.gripper_quats = self.get_quats(part_type = 'gripper')

    def render(self, render_type = 'png'):

        """Renders the scene
        Arg:
            render_type: tells the method whether to save to png or save to hdr
        """
        self.image_counter += 1
            
        # iterate through all the mujoco objects
        
        for static_object in self.mujoco_objects:

            static_object_name = static_object.name

            if static_object_name in self.static_object_entities:
                obj_entities = self.static_object_entities[static_object_name]
            else:
                continue 

            for i in range(len(obj_entities)):

                curr_obj_entity = obj_entities[i]

                if isinstance(curr_obj_entity, tuple) and not isinstance(static_object, DoorObject) and len(curr_obj_entity) == 2:

                    for link_idx in range(len(curr_obj_entity)):

                        if len(curr_obj_entity) == 1:
                            name = static_object_name
                            entity = curr_obj_entity[0]
                        else:
                            name = curr_obj_entity[0]
                            entity = curr_obj_entity[1]

                        pos = np.array(self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(name)])
                        R = self.env.sim.data.geom_xmat[self.env.sim.model.geom_name2id(name)].reshape(3, 3)

                        quat_xyzw = self.quaternion_from_matrix3(R)
                        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                        entity.get_transform().set_position(visii.vec3(pos[0],
                                                                       pos[1],
                                                                       pos[2]))
                        
                        entity.get_transform().set_rotation(visii.quat(quat[0],
                                                                       quat[1],
                                                                       quat[2],
                                                                       quat[3]))

                        self.set_entity_material(entity)

                elif isinstance(curr_obj_entity, tuple) and len(curr_obj_entity) == 3:

                    entity = curr_obj_entity[1]
                    name = entity.get_name()

                    pos = np.array(self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(name)])

                    R = self.env.sim.data.geom_xmat[self.env.sim.model.geom_name2id(name)].reshape(3, 3)
                    quat_xyzw = self.quaternion_from_matrix3(R)
                    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                    entity.get_transform().set_position(visii.vec3(pos[0],
                                                                   pos[1],
                                                                   pos[2]))
                    entity.get_transform().set_rotation(visii.quat(quat[0],
                                                                   quat[1],
                                                                   quat[2],
                                                                   quat[3]))

                    self.set_entity_material(entity)

                elif isinstance(static_object, DoorObject):

                    name = curr_obj_entity[0]
                    entity = curr_obj_entity[1]

                    pos = np.array(self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(name)])
                    R = self.env.sim.data.geom_xmat[self.env.sim.model.geom_name2id(name)].reshape(3, 3)
                    quat_xyzw = self.quaternion_from_matrix3(R)
                    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                    entity.get_transform().set_position(visii.vec3(pos[0],
                                                                   pos[1],
                                                                   pos[2]))
                    entity.get_transform().set_rotation(visii.quat(quat[0],
                                                                   quat[1],
                                                                   quat[2],
                                                                   quat[3]))

                    self.set_entity_material(entity)

                else:
                    name = static_object_name + '_main'
                    
                    entity = None
                    if isinstance(curr_obj_entity, tuple):
                        entity = curr_obj_entity[0]
                    else:
                        entity = curr_obj_entity

                    pos = np.array(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(name)])
                    R = self.env.sim.data.body_xmat[self.env.sim.model.body_name2id(name)].reshape(3, 3)
                    quat_xyzw = self.quaternion_from_matrix3(R)
                    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                    entity.get_transform().set_position(visii.vec3(pos[0],
                                                                   pos[1],
                                                                   pos[2]))
                    entity.get_transform().set_rotation(visii.quat(quat[0],
                                                                   quat[1],
                                                                   quat[2],
                                                                   quat[3]))

                    self.set_entity_material(entity)

        # For now we are only rendering it as a png
        # stl file extension
        mesh_color_arr = None

        for i in range(len(self.meshes)):

            part_mesh = self.meshes[i]
            part_name = self.geom_names[part_mesh]
            link_entity = self.entities[i]

            part_position = self.positions[i]
            part_quaternion = self.quats[i]

            if link_entity == None:
                continue

            if isinstance(link_entity, tuple):

                for link_idx in range(len(link_entity)):
                    link_entity[link_idx].get_transform().set_position(visii.vec3(part_position[0], part_position[1], part_position[2]))

                    link_entity[link_idx].get_transform().set_rotation(visii.quat(part_quaternion[0],
                                                                                  part_quaternion[1],
                                                                                  part_quaternion[2],
                                                                                  part_quaternion[3]))

                    self.set_entity_material(link_entity[link_idx])

            else:
            
                link_entity.get_transform().set_position(visii.vec3(part_position[0], part_position[1], part_position[2]))

                link_entity.get_transform().set_rotation(visii.quat(part_quaternion[0],
                                                                    part_quaternion[1],
                                                                    part_quaternion[2],
                                                                    part_quaternion[3]))

                if part_mesh in self.mesh_colors and self.mesh_colors[part_mesh] != None:
                    mesh_color_arr = self.mesh_colors[part_mesh].split(' ')
                    link_entity.get_material().set_base_color(visii.vec3(float(mesh_color_arr[0]),
                                                                         float(mesh_color_arr[1]),
                                                                         float(mesh_color_arr[2])))

                self.set_entity_material(link_entity)
            
        for key in self.gripper_parts.keys():
            
            gripper_entity = None

            gripper_mesh_arr = self.gripper_parts[key][0]
            gripper_mesh_quat = self.gripper_parts[key][1]

            if key == 'wiping_gripper':
                xml_file = f'../models/assets/grippers/{key}.xml'

                wipe_xml = ET.parse(xml_file)
                wipe_xml_root = wipe_xml.getroot()
                    
                for wipe_geom in wipe_xml_root.find('worldbody').find('body').findall('geom'):

                    obj_entity = self.gripper_entities[wipe_geom.get('name')]
                
                    name = 'gripper0_' + wipe_geom.get('name')
                    gripper_pos = np.array(self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(name)])
                    R = self.env.sim.data.geom_xmat[self.env.sim.model.geom_name2id(name)].reshape(3, 3)
                    quat_xyzw = self.quaternion_from_matrix3(R)
                    gripper_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                    
                    obj_entity.get_transform().set_position(visii.vec3(gripper_pos[0],
                                                                       gripper_pos[1],
                                                                       gripper_pos[2]))

                    obj_entity.get_transform().set_rotation(visii.quat(gripper_quat[0],
                                                                       gripper_quat[1],
                                                                       gripper_quat[2],
                                                                       gripper_quat[3]))

                    self.set_entity_material(obj_entity)

            for (mesh, mesh_quat) in zip(gripper_mesh_arr, gripper_mesh_quat):

                gripper_entity = self.gripper_entities[f'{key}_{mesh}']
                part_position = self.gripper_positions[key]
                part_quaternion = self.gripper_quats[key]
                
                if isinstance(gripper_entity, tuple):

                    visii_part_quat = visii.quat(*part_quaternion) * visii.quat(*mesh_quat)

                    for link_idx in range(len(gripper_entity)):
                        gripper_entity[link_idx].get_transform().set_position(visii.vec3(part_position[0], part_position[1], part_position[2]))

                        gripper_entity[link_idx].get_transform().set_rotation(visii_part_quat)

                        self.set_entity_material(gripper_entity[link_idx])

                else:

                    gripper_entity.get_transform().set_position(visii.vec3(part_position[0], part_position[1], part_position[2]))

                    visii_part_quat = visii.quat(*part_quaternion) * visii.quat(*mesh_quat)
                    gripper_entity.get_transform().set_rotation(visii_part_quat)

                    if mesh == 'finger_vis':
                        gripper_entity.get_material().set_base_color(visii.vec3(0.5, 0.5, 0.5))
                    
                    self.set_entity_material(gripper_entity)

        for key in self.extra_entities.keys():

            extra_entity = self.extra_entities[key]
            name = key

            pos = np.array(self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(name)])
            R = self.env.sim.data.geom_xmat[self.env.sim.model.geom_name2id(name)].reshape(3, 3)
            quat_xyzw = self.quaternion_from_matrix3(R)
            quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            extra_entity.get_transform().set_position(visii.vec3(pos[0],
                                                                 pos[1],
                                                                 pos[2]))
            extra_entity.get_transform().set_rotation(visii.quat(quat[0],
                                                                 quat[1],
                                                                 quat[2],
                                                                 quat[3]))

            self.set_entity_material(extra_entity)

        visii.render_to_file(
            width             = self.width,
            height            = self.height, 
            samples_per_pixel = self.samples_per_pixel,   
            file_path        = f'images/image_{self.image_counter}.png'
        )

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

    def set_entity_material(self, entity):

        if isinstance(entity, tuple):
            for link_idx in range(len(entity)):
                entity[link_idx].get_material().set_metallic(0)
                entity[link_idx].get_material().set_transmission(0)
                entity[link_idx].get_material().set_roughness(0.3)

        else:
            entity.get_material().set_metallic(0)
            entity.get_material().set_transmission(0)
            entity.get_material().set_roughness(0.3)

    def printState(self): # For testing purposes
        print(self.obs_dict)

    def close(self):
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
