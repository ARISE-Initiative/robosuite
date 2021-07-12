import os
import numpy as np
import nvisii_rendering_utils as vutils
import nvisii
import open3d as o3d

from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole

path = str(os.path.realpath(__file__))
file_path = path[:path.rfind('/') + 1]

def dynamic_robot_init(env, root, robot_num, robot_name):

    meshes = {}

    count = 0
    for body in root.iter('body'):

        for geom in body.findall('geom'):

            body_name  = body.get('name')
            geom_mesh  = geom.get('mesh')

            if geom_mesh != None:
                
                if body_name not in meshes:
                    meshes[body_name] = geom_mesh

                    count += 1

    positions = vutils.get_positions(env, 'robot', meshes, robot_num)
    quats = vutils.get_quaternions(env, 'robot', meshes.keys(), robot_num)

    robot_entities = {}

    for mesh in meshes:
        
        file_name = meshes[mesh]
        # if 'vis' in file_name:
        #     file_name = file_name[:-4]
        
        obj_file = file_path + f'../../models/assets/robots/{robot_name.lower()}/meshes/{file_name}.obj'

        key = f'robot{robot_num}_{mesh}'

        entity = vutils.import_obj(env,
                                   key,
                                   obj_file)

        
        robot_entities[key] = entity

    return meshes, positions, quats, count, robot_entities

def dynamic_gripper_init(env, root, robot_num, gripper_name):

    gripper_entities = {}
    meshes = {}
    geom_quats = {}
    positions = {}
    quats = {}

    count = 0

    if 'robotiq' in gripper_name:
        new_gripper_name = 'robotiq_' + gripper_name[gripper_name.rfind('_') + 1:] + '_gripper'
        gripper_name = new_gripper_name

    if isinstance(env, TwoArmPegInHole):

        if robot_num == 0:

            name = env.peg.name + '_g0_vis'
            size = env.peg.size

            texture_name = "peg_texture"
            texture_file = env.peg.material.tex_attrib["file"]
            
            pos = vutils.get_position_geom(env, name)

            positions[name] = pos

            entity = vutils.create_entity(entity_type='cylinder',
                                          entity_name=name,
                                          size=size,
                                          pos=pos,
                                          texture_name=texture_name,
                                          texture_file=texture_file)

            vutils.set_entity_rotation_geom(env, name, entity)

            gripper_entities[name] = entity

            return meshes, positions, quats, geom_quats, count, gripper_entities

        elif robot_num == 1:

            hole_xml = env.hole.worldbody
            object_tag = hole_xml.find("./body/body[@name='hole_object']")
            # create_entity(entity_type, entity_name, size, pos, texture_name, texture_file, rgba=None):

            geom_count = 0
            texture_name = 'hole_texture'
            texture_file = file_path + '../../models/assets/textures/red-wood.png'

            for geom in object_tag.findall('geom'):

                name = f'hole_g{geom_count}_visual'
                size = [float(x) for x in geom.get('size').split(' ')]
                pos = vutils.get_position_geom(env, name)
                geom_type = geom.get('type')

                positions[name] = pos

                entity = vutils.create_entity(entity_type=geom_type,
                                              entity_name=name,
                                              size=size,
                                              pos=pos,
                                              texture_name=texture_name,
                                              texture_file=texture_file)

                vutils.set_entity_rotation_geom(env, name, entity)

                gripper_entities[name] = entity

                geom_count+=1

            return meshes, positions, quats, geom_quats, count, gripper_entities

    if gripper_name == 'wiping_gripper':

        for geom in root.find('worldbody').find('body').findall('geom'):

            name = 'gripper0_' + geom.get('name')
            size = [float(x) for x in geom.get('size').split(' ')]
            pos = vutils.get_position_geom(env, name)
            geom_type = geom.get('type')

            positions[name] = pos

            entity = vutils.create_entity(entity_type=geom_type,
                                          entity_name=name,
                                          size=size,
                                          pos=pos,
                                          texture_name=None,
                                          texture_file=None,
                                          rgba=[0.25,0.25,0.25])

            vutils.set_entity_rotation_geom(env, name, entity)

            gripper_entities[name] = entity

        return meshes, positions, quats, geom_quats, count, gripper_entities

    for body in root.iter('body'):

        body_name  = body.get('name')

        for geom in body.findall('geom'):

            geom_mesh  = geom.get('mesh')
            geom_quat = geom.get('quat')

            if "collision" in geom.get('name'):
                continue

            if geom_mesh != None:
                if body_name in meshes:
                    meshes[body_name].append(geom_mesh)
                else:
                    meshes[body_name] = [geom_mesh]

                if geom_quat is None:
                    geom_quat = [1, 0, 0, 0]
                else:
                    geom_split_values = geom_quat.split(' ')
                    geom_split_values_no_whitespace = []
                    for i in geom_split_values:
                        if i != "":
                            geom_split_values_no_whitespace.append(i)

                    # print(f'geom_quat: {geom_split_values_no_whitespace}')
                    geom_quat = [float(element) for element in geom_split_values_no_whitespace]
                
                geom_quats[f'{body_name}-{geom_mesh}'] = geom_quat

                count += 1

    positions = vutils.get_positions(env, 'gripper', meshes.keys(), robot_num)
    quats = vutils.get_quaternions(env, 'gripper', meshes.keys(), robot_num)

    for key in meshes:

        for mesh_n in meshes[key]:

            mesh_name = f'{key}-{mesh_n}'

            object_name = f'gripper{robot_num}_{key}'
            key_g = f'gripper{robot_num}_{mesh_name}'

            obj_file = file_path + f'../../models/assets/grippers/meshes/{gripper_name}/{mesh_n}.obj'

            entity = None

            if os.path.exists(obj_file):

                # print('obj_file EXISTS!')
                entity = vutils.import_obj(env,
                                           object_name,
                                           obj_file,
                                           part_type='body')

            else:
                
                # print('obj_file DOES NOT EXIST!')
                stl_file = obj_file.replace('obj', 'stl')
                mesh_gripper = o3d.io.read_triangle_mesh(stl_file)

                normals  = np.array(mesh_gripper.vertex_normals).flatten().tolist()
                vertices = np.array(mesh_gripper.vertices).flatten().tolist()

                mesh_name += f'_{robot_num}'

                # print(mesh_name)
                mesh = nvisii.mesh.create_from_data(mesh_name, positions=vertices, normals=normals)
                entity = nvisii.entity.create(
                    name      = mesh_name,
                    mesh      = mesh,
                    transform = nvisii.transform.create(mesh_name),
                    material  = nvisii.material.create(mesh_name)
                )

            # if mesh_n == 'finger_vis':

            #     if isinstance(entity, tuple):
                    
            #         for link_idx in range(len(entity)):
            #             entity[link_idx].get_material().set_base_color(nvisii.vec3(0.5, 0.5, 0.5))

            #     else:
            #         entity.get_material().set_base_color(nvisii.vec3(0.5, 0.5, 0.5))

            gripper_entities[key_g] = entity

    return meshes, positions, quats, geom_quats, count, gripper_entities