import os
import numpy as np
import math
import nvisii

def load_object(geom,
                geom_name,
                geom_type,
                geom_quat,
                geom_pos,
                geom_size,
                geom_scale,
                geom_rgba,
                geom_tex_name,
                geom_tex_file,
                meshes):
    """
    Function that initializes the meshes in the memory.

    Args:
        geom (XML element): Object in XML file to load

        geom_name (str): Name for the object.

        geom_type (str): Type of the object. Types include "box", "cylinder", or "mesh".

        geom_quat (array): Quaternion (wxyz) of the object.

        geom_pos (array): Position of the object.

        geom_size (array): Size of the object.

        geom_scale (array): Scale of the object.

        geom_rgba (array): Color of the object. This is only used if the geom type is not
                           a mesh and there is no specified material.

        geom_tex_name (str): Name of the texture for the object

        geom_tex_file (str): File of the texture for the object

        meshes (dict): meshes for the object
    """

    primitive_types = ['box', 'cylinder']
    component = None

    if geom_type == 'box':
        component = nvisii.entity.create(
                    name = geom_name,
                    mesh = nvisii.mesh.create_box(name = geom_name,
                                                  size = nvisii.vec3(geom_size[0],
                                                                     geom_size[1],
                                                                     geom_size[2])),
                    transform = nvisii.transform.create(geom_name),
                    material = nvisii.material.create(geom_name)
                )

    elif geom_type == 'cylinder':
        component = nvisii.entity.create(
                    name = geom_name,
                    mesh = nvisii.mesh.create_capped_cylinder(name   = geom_name,
                                                              radius = geom_size[0],
                                                              size   = geom_size[1]),
                    transform = nvisii.transform.create(geom_name),
                    material = nvisii.material.create(geom_name)
                )

    elif geom_type == 'mesh':
        filename = meshes[geom.attrib['mesh']]['file']
        filename = os.path.splitext(filename)[0] + '.obj'

        component = nvisii.import_scene(
                    file_path=filename,
                    position=nvisii.vec3(geom_pos[0],
                                         geom_pos[1],
                                         geom_pos[2]),
                    scale=(geom_scale[0], geom_scale[1], geom_scale[2]),
                    rotation=nvisii.quat(geom_quat[0],
                                         geom_quat[1],
                                         geom_quat[2],
                                         geom_quat[3])
                )

    if geom_type in primitive_types:
        component.get_transform().set_position(nvisii.vec3(float(geom_pos[0]),
                                                           float(geom_pos[1]),
                                                           float(geom_pos[2])))

    if geom_tex_file is not None and geom_tex_name is not None and geom_type != 'mesh':

        texture = nvisii.texture.get(geom_tex_name)

        if texture is None:
            texture = nvisii.texture.create_from_file(name = geom_tex_name,
                                                      path = geom_tex_file)

        component.get_material().set_base_color_texture(texture)
    else:
        if 'gripper' in geom_name:
            if geom_rgba is not None:
                if isinstance(component, nvisii.scene):
                    for entity in component.entities:
                        entity.get_material().set_base_color(nvisii.vec3(geom_rgba[0], geom_rgba[1], geom_rgba[2]))
                else:
                    component.get_material().set_base_color(nvisii.vec3(geom_rgba[0], geom_rgba[1], geom_rgba[2]))
            elif 'hand_visual' in geom_name:
                for entity in component.entities:
                        entity.get_material().set_base_color(nvisii.vec3(0.05, 0.05, 0.05))

    return component