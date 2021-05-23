import numpy as np
import math
import nvisii

def create_entity(entity_type, entity_name, size, pos, texture_name, texture_file, rgba=None):

    entity = None

    if entity_type == 'box':

        entity = nvisii.entity.create(
                    name = entity_name,
                    mesh = nvisii.mesh.create_box(name = entity_name,
                                                 size = nvisii.vec3(size[0],
                                                                   size[1],
                                                                   size[2])),
                    transform = nvisii.transform.create(entity_name),
                    material = nvisii.material.create(entity_name)
                )

    elif entity_type == 'cylinder':

        entity = nvisii.entity.create(
                    name = entity_name,
                    mesh = nvisii.mesh.create_capped_cylinder(name   = entity_name,
                                                             radius = float(size[0]),
                                                             size   = float(size[1])),
                    transform = nvisii.transform.create(entity_name),
                    material = nvisii.material.create(entity_name)
                )

    elif entity_type == 'sphere':

        entity = nvisii.entity.create(
                    name = entity_name,
                    mesh = nvisii.mesh.create_sphere(name = entity_name,
                                                    radius = float(size[0])),
                    transform = nvisii.transform.create(entity_name),
                    material = nvisii.material.create(entity_name)
                )

    if texture_name != None:
        texture = nvisii.texture.get(texture_name)

        if texture == None:
            texture = nvisii.texture.create_from_file(name = texture_name,
                                                      path = texture_file)

            # texture = nvisii.texture.create_hsv(name = texture_name + '_darker',
            #                                     tex = texture,
            #                                     hue = 0,
            #                                     saturation = 0,
            #                                     value = -0.5)

        entity.get_material().set_base_color_texture(texture)

    else:
        entity.get_material().set_base_color(nvisii.vec3(rgba[0], rgba[1], rgba[2]))


    entity.get_transform().set_position(nvisii.vec3(float(pos[0]),
                                                    float(pos[1]),
                                                    float(pos[2])))

    return entity

def set_entity_rotation_geom(env, entity_name, entity):

    quat = get_quaternion_geom(env, entity_name)

    entity.get_transform().set_rotation(nvisii.quat(quat[0],
                                                    quat[1],
                                                    quat[2],
                                                    quat[3]))

def set_entity_rotation_body(env, entity_name, entity):

    quat = get_quaternion_body(env, entity_name)

    entity.get_transform().set_rotation(nvisii.quat(quat[0],
                                                    quat[1],
                                                    quat[2],
                                                    quat[3]))

def import_obj(env, name, obj_file, part_type='body'):

    pos = None
    quat = None

    if part_type == 'body':
        pos = get_position_body(env, name)
        quat = get_quaternion_body(env, name)
    elif part_type == 'geom':
        pos = get_position_geom(env, name)
        quat = get_quaternion_geom(env, name)

    scene = nvisii.import_scene(
                    file_path=obj_file,
                    position=nvisii.vec3(pos[0],
                                         pos[1],
                                         pos[2]),
                    scale=(1.0, 1.0, 1.0),
                    rotation=nvisii.quat(quat[0],
                                         quat[1],
                                         quat[2],
                                         quat[3])
                )

    return scene

def get_positions(env, part_type, parts, robot_num):

    positions = {}

    for part in parts:

        part_name = f'{part_type}{robot_num}_{part}'

        positions[part_name] = get_position_body(env, part_name)

    return positions

def get_quaternions(env, part_type, parts, robot_num):

    quats = {}

    for part in parts:

        part_name = f'{part_type}{robot_num}_{part}'
        quats[part_name] = get_quaternion_body(env, part_name)

    return quats

def get_position_body(env, name):
    return env.sim.data.body_xpos[env.sim.model.body_name2id(name)]

def get_quaternion_body(env, name):
    R = env.sim.data.body_xmat[env.sim.model.body_name2id(name)].reshape(3, 3)
    quat_xyzw = _quaternion_from_matrix3(R)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return quat_wxyz

def get_position_geom(env, name):
    return env.sim.data.geom_xpos[env.sim.model.geom_name2id(name)]

def get_quaternion_geom(env, name):
    R = env.sim.data.geom_xmat[env.sim.model.geom_name2id(name)].reshape(3, 3)
    quat_xyzw = _quaternion_from_matrix3(R)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return quat_wxyz

def _quaternion_from_matrix3(matrix3):
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