import os
import numpy as np
import visii
import visii_utils as vutils

from robosuite.models.objects import MujocoXMLObject, PrimitiveObject, CompositeObject
from robosuite.models.objects import BoxObject, CylinderObject, BallObject, CapsuleObject
from robosuite.models.objects import DoorObject, SquareNutObject, RoundNutObject
from robosuite.models.objects import PotWithHandlesObject, HammerObject
from robosuite.models.arenas import TableArena, BinsArena, PegsArena, EmptyArena, WipeArena

import xml.etree.ElementTree as ET

def init_arena_visii(env):

    mujoco_arena = env.model.mujoco_arena

    if isinstance(mujoco_arena, TableArena):
        init_table_arena(mujoco_arena)
        
        if isinstance(mujoco_arena, WipeArena):
            init_wipe_arena(env, mujoco_arena)
        elif isinstance(mujoco_arena, PegsArena):
            init_pegs_arena(mujoco_arena)
    
    elif isinstance(mujoco_arena, BinsArena):
        init_bins_arena(mujoco_arena)

def init_table_arena(mujoco_arena):
    """
    def create_entity(entity_type, entity_name, size, position, texture_name, texture_file)
    """

    vutils.create_entity(entity_type='box', 
                         entity_name='table', 
                         size=mujoco_arena.table_half_size, 
                         pos=_split_pos(mujoco_arena.table_body), 
                         texture_name='ceramic_table_texture', 
                         texture_file='../models/assets/textures/ceramic.png')

    leg_count = 1
    for leg in mujoco_arena.table_legs_visual:

        leg_name = f"table_leg_{leg_count}"

        rel_pos = _split_pos(leg)
        pos = _split_pos(mujoco_arena.table_body)

        for i in range(len(pos)):
            pos[i] = pos[i] + rel_pos[i]

        vutils.create_entity(entity_type='cylinder', 
                             entity_name=leg_name, 
                             size=_split_size(leg), 
                             pos=pos, 
                             texture_name='steel_legs_texture', 
                             texture_file='../models/assets/textures/steel-brushed.png')

        leg_count+=1

def init_wipe_arena(env, mujoco_arena):

    for sensor in range(mujoco_arena.num_markers):

        sensor_name = f'contact{sensor}_g0_vis'
        pos = np.array(env.sim.data.geom_xpos[env.sim.model.geom_name2id(sensor_name)])
        radius = mujoco_arena.line_width / 2
        half_length = 0.001
        size = [radius, half_length]

        vutils.create_entity(entity_type='cylinder', 
                             entity_name=sensor_name, 
                             size=size, 
                             pos=pos, 
                             texture_name='dirt_texture', 
                             texture_file='../models/assets/textures/dirt.png')

def init_pegs_arena(mujoco_arena):

    vutils.create_entity(entity_type='box', 
                         entity_name='peg1', 
                         size=_split_size(mujoco_arena.peg1_body.find("./geom[@group='1']")),
                         pos=_split_pos(mujoco_arena.peg1_body), 
                         texture_name='peg1_texture', 
                         texture_file='../models/assets/textures/brass-ambra.png')

    vutils.create_entity(entity_type='cylinder', 
                         entity_name='peg2', 
                         size=_split_size(mujoco_arena.peg2_body.find("./geom[@group='1']")),
                         pos=_split_pos(mujoco_arena.peg2_body), 
                         texture_name='peg2_texture', 
                         texture_file='../models/assets/textures/steel-scratched.png')

def init_bins_arena(mujoco_arena):

    base_pos = _split_pos(mujoco_arena.bin1_body)

    wall_count = 1
    for wall in mujoco_arena.bin1_body.findall("./geom[@material='light-wood']"):

        name = f'wall_light_{wall_count}'

        rel_pos = _split_pos(wall)
        pos = []
        for i in range(len(base_pos)):
            pos.append(base_pos[i] + rel_pos[i])

        vutils.create_entity(entity_type='box', 
                             entity_name=name, 
                             size=_split_size(wall),
                             pos=pos, 
                             texture_name='light-wood_table_texture', 
                             texture_file='../models/assets/textures/light-wood.png')
        wall_count+=1

    leg_count = 1
    for leg in mujoco_arena.bin1_body.findall("./geom[@material='table_legs_metal']"):

        leg_name = f'table_light_leg_{leg_count}'

        rel_pos = _split_pos(leg)
        pos = []
        for i in range(len(base_pos)):
            pos.append(base_pos[i] + rel_pos[i])

        vutils.create_entity(entity_type='cylinder', 
                             entity_name=leg_name, 
                             size=_split_size(leg), 
                             pos=pos, 
                             texture_name='steel_legs_texture', 
                             texture_file='../models/assets/textures/steel-brushed.png')

        leg_count+=1

    base_pos = _split_pos(mujoco_arena.bin2_body)

    wall_count = 1
    for wall in mujoco_arena.bin2_body.findall("./geom[@material='dark-wood']"):

        name = f'wall_dark_{wall_count}'

        rel_pos = _split_pos(wall)
        pos = []
        for i in range(len(base_pos)):
            pos.append(base_pos[i] + rel_pos[i])

        vutils.create_entity(entity_type='box', 
                             entity_name=name, 
                             size=_split_size(wall),
                             pos=pos, 
                             texture_name='dark-wood_table_texture', 
                             texture_file='../models/assets/textures/dark-wood.png')
        wall_count+=1

    leg_count = 1
    for leg in mujoco_arena.bin2_body.findall("./geom[@material='table_legs_metal']"):

        leg_name = f'table_dark_leg_{leg_count}'

        rel_pos = _split_pos(leg)
        pos = []
        for i in range(len(base_pos)):
            pos.append(base_pos[i] + rel_pos[i])

        vutils.create_entity(entity_type='cylinder', 
                             entity_name=leg_name, 
                             size=_split_size(leg), 
                             pos=pos, 
                             texture_name='steel_legs_texture', 
                             texture_file='../models/assets/textures/steel-brushed.png')

        leg_count+=1

def _split(arr):
    return [float(x) for x in arr]

def _split_pos(body):
    return _split(body.get('pos').split(' ')) 

def _split_size(body):
    return _split(body.get('size').split(' ')) 

def init_objects_visii(env):

    mujoco_objects = env.model.mujoco_objects

    obj_entity_dict = {}

    for static_object in mujoco_objects:

        name = f'{static_object.name}_main'
        reg_name = name

        entity = None
        bq_type = 'body'

        if 'Visual' in name:
            continue

        if isinstance(static_object, MujocoXMLObject):

            xml_file = static_object.file
            curr_dir = xml_file[:xml_file.rfind('/')]

            mesh_tag = ET.parse(xml_file).getroot().find('./asset/mesh')
            if mesh_tag != None:
                stl_file = mesh_tag.get('file')

                if stl_file != None:
                    obj_file = os.path.join(curr_dir, _stl_to_obj(stl_file))

                    entity = vutils.import_obj(env, name, obj_file)
                    obj_entity_dict[reg_name] = [entity, bq_type]

            else:

                xml_file = static_object.file
                base_path = xml_file[:xml_file.rfind('/')] + '/'

                tree = ET.parse(xml_file)
                root = tree.getroot()

                textures = {}
                materials = {}

                for asset in root.findall('asset'):
                    for texture in asset.findall('texture'):
                        texture_name = static_object.name + '_' + texture.get('name')
                        texture_file = texture.get('file')
                        textures[texture_name] = base_path + texture_file

                    for material in asset.findall('material'):
                        material_name = static_object.name + '_' + material.get('name')
                        material_texture = static_object.name + '_' + material.get('texture')
                        materials[material_name] = material_texture

                obj_name = f'{static_object.name}_object'
                obj_body = static_object.worldbody.find("./body/body") # [@name='object']

                geom_count = 0 
                for geom in obj_body.iter('geom'):

                    entity_name = geom.get('name')

                    if isinstance(static_object, SquareNutObject) or isinstance(static_object, RoundNutObject):
                        entity_name = name.replace('_main', f'_g{geom_count}')
                    
                    reg_name = entity_name

                    size = _split_size(geom)
                    if len(size) == 1:
                        fromto = _split(geom.get('fromto').split(' '))
                        length = (fromto[1] - fromto[4]) / 2
                        size.append(length)

                    pos = vutils.get_position_geom(env, entity_name)
                    quat = vutils.get_quaternion_geom(env, entity_name)
                    entity_type = geom.get('type')
                    bq_type = 'geom'

                    material = geom.get('material')

                    if material != None:
                        texture_name = materials[material]
                        texture_file = textures[texture_name]

                        entity = vutils.create_entity(entity_type=entity_type, 
                                                      entity_name=entity_name, 
                                                      size=size, 
                                                      pos=pos,
                                                      texture_name=texture_name, 
                                                      texture_file=texture_file)

                    else:
                        rgba = _split(geom.get('rgba').split(' '))
                        entity = vutils.create_entity(entity_type=entity_type, 
                                                      entity_name=entity_name, 
                                                      size=size, 
                                                      pos=pos,
                                                      texture_name=None, 
                                                      texture_file=None,
                                                      rgba=rgba)

                    vutils.set_entity_rotation_geom(env, entity_name, entity)
                    obj_entity_dict[reg_name] = [entity, bq_type]

                    geom_count += 1

        elif isinstance(static_object, PrimitiveObject):

            texture_name = static_object.material.tex_attrib["name"]
            texture_file = static_object.material.tex_attrib["file"]

            entity_type = None
            if isinstance(static_object, BoxObject): entity_type = 'box'
            elif isinstance(static_object, CylinderObject): entity_type = 'cylinder'
            elif isinstance(static_object, CapsuleObject): entity_type = 'capsule'
            elif isinstance(static_object, BallObject): entity_type = 'ball' 

            pos = vutils.get_position_body(env, name)

            entity = vutils.create_entity(entity_type=entity_type, 
                                          entity_name=name, 
                                          size=static_object.size,
                                          pos=pos,
                                          texture_name=texture_name, 
                                          texture_file=texture_file)
            obj_entity_dict[reg_name] = [entity, bq_type]

        elif isinstance(static_object, CompositeObject):

            obj_dict = static_object._get_geom_attrs()
            num_objects = len(obj_dict['geom_names'])

            textures = {}
            materials = {}

            for texture in static_object.asset.iter('texture'):
                name = texture.get('name')
                file = texture.get('file')
                textures[name] = file

            for material in static_object.asset.iter('material'):
                name = material.get('name')
                texture = material.get('texture')
                materials[name] = texture

            for i in range(num_objects):

                size = obj_dict['geom_sizes'][i]
                name = static_object.name + '_' + obj_dict['geom_names'][i]
                geom_type = obj_dict['geom_types'][i]
                geom_mat = obj_dict['geom_materials'][i]
                pos = vutils.get_position_geom(env, name)
                bq_type = 'geom'
                reg_name = name

                if isinstance(static_object, PotWithHandlesObject) and 'pot_' not in geom_mat:
                    geom_mat = 'pot_' + geom_mat
                elif isinstance(static_object, HammerObject) and 'hammer_' not in geom_mat:
                    geom_mat = 'hammer_' + geom_mat

                texture_name = materials[geom_mat]
                texture_file = textures[texture_name]

                entity = vutils.create_entity(entity_type=geom_type, 
                                              entity_name=name, 
                                              size=size,
                                              pos=pos,
                                              texture_name=texture_name, 
                                              texture_file=texture_file)

                vutils.set_entity_rotation_geom(env, name, entity)
                obj_entity_dict[reg_name] = [entity, bq_type]

        #obj_entity_dict[reg_name] = [entity, bq_type]

    return obj_entity_dict

def _stl_to_obj(stl_file):
    return stl_file.replace('.stl', '.obj')

def init_pedestals(env):

    obj_file = f'../models/assets/mounts/meshes/rethink_mount/pedestal.obj'

    robot_count = 0
    for robot in range(len(env.robots)):

        name = f'mount{robot_count}_base'

        entity = visii.import_obj(name,
                                  obj_file,
                                  obj_file[:obj_file.rfind('/')] + '/')


        pos = vutils.get_position_body(env, name)
        quat = vutils.get_quaternion_body(env, name)

        for link_idx in range(len(entity)):

            entity[link_idx].get_transform().set_position(visii.vec3(pos[0],
                                                                     pos[1],
                                                                     pos[2]))
            
            entity[link_idx].get_transform().set_rotation(visii.quat(quat[0],
                                                                     quat[1],
                                                                     quat[2],
                                                                     quat[3]))

        robot_count += 1