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
                instance_id,
                visual_objects,
                meshes):
    """
    Function that initializes the meshes in the memory.
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

        # print(geom_name)
        # if 's_visual' in geom_name:
        #     geom_scale = (3, 3, 3)

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

def quaternion_from_matrix3(matrix3):
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