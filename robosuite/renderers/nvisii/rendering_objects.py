import os
import numpy as np
import nvisii
import nvisii_rendering_utils as vutils

from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite.environments.manipulation.wipe import Wipe

def render_robots(env, robot_info):

    for robot in robot_info.keys():

        robot_entities = robot_info[robot][4]
        robot_positions = robot_info[robot][1]
        robot_quats = robot_info[robot][2]

        for key in robot_entities.keys():

            entity = robot_entities[key]
            entity_pos = robot_positions[key]
            entity_quat = robot_quats[key]
            
            set_position_rotation(entity, entity_pos, entity_quat)

def render_grippers(env, gripper_info):

    for gripper in gripper_info.keys():

        gripper_entities = gripper_info[gripper][4]
        gripper_positions = gripper_info[gripper][1]
        gripper_quats = gripper_info[gripper][2]
        geom_quats = gripper_info[gripper][5]

        for key in gripper_entities.keys():

            entity = gripper_entities[key]

            if isinstance(env, TwoArmPegInHole) or isinstance(env, Wipe):

                entity_pos = vutils.get_position_geom(env, key)
                entity_quat = vutils.get_quaternion_geom(env, key)

                set_position_rotation(entity, entity_pos, entity_quat, False)

                continue

            if '-' in key:
                pq_key = key[:key.find('-')]
            else:
                pq_key = key
            mesh = key[key.find('-')+1:]
            gq_key = key[key.find('_')+1:]

            if pq_key in gripper_positions:
                entity_pos = gripper_positions[pq_key]
            else:
                continue

            entity_quat = gripper_quats[pq_key]
            geom_quat = geom_quats[gq_key]

            nvisii_quat = nvisii.quat(*entity_quat) * nvisii.quat(*geom_quat)

            set_position_rotation(entity, entity_pos, nvisii_quat, True)

def render_objects(env, obj_entities):

    mujoco_objects = obj_entities

    for entity_name in mujoco_objects:

        if isinstance(env, Wipe):
            if entity_name in env.wiped_markers:
                nvisii.remove(entity_name)
                continue

        entity = obj_entities[entity_name][0]
        bq_type = obj_entities[entity_name][1]

        if bq_type == 'body':
            pos = vutils.get_position_body(env, entity_name)
            quat = vutils.get_quaternion_body(env, entity_name)
        else:
            pos = vutils.get_position_geom(env, entity_name)
            quat = vutils.get_quaternion_geom(env, entity_name)

        set_position_rotation(entity, pos, quat)
        
def set_position_rotation(entity, entity_pos, entity_quat, nvisii_quat = False, ):

    if isinstance(entity, tuple):

        for link_idx in range(len(entity)):

            entity[link_idx].get_transform().set_position(nvisii.vec3(entity_pos[0],
                                                                     entity_pos[1],
                                                                     entity_pos[2]))

            if not nvisii_quat:

                entity[link_idx].get_transform().set_rotation(nvisii.quat(entity_quat[0],
                                                                         entity_quat[1],
                                                                         entity_quat[2],
                                                                         entity_quat[3]))

            else:
                entity[link_idx].get_transform().set_rotation(entity_quat)

    elif isinstance(entity, nvisii.scene):
        
        entity.transforms[0].set_position(nvisii.vec3(entity_pos[0],
                                                     entity_pos[1],
                                                     entity_pos[2]))

        if not nvisii_quat:
            entity.transforms[0].set_rotation(nvisii.quat(entity_quat[0],
                                                           entity_quat[1],
                                                           entity_quat[2],
                                                           entity_quat[3]))
        
        else:
            entity.transforms[0].set_rotation(entity_quat)

    else:

        entity.get_transform().set_position(nvisii.vec3(entity_pos[0],
                                                       entity_pos[1],
                                                       entity_pos[2]))

        if not nvisii_quat:
            entity.get_transform().set_rotation(nvisii.quat(entity_quat[0],
                                                           entity_quat[1],
                                                           entity_quat[2],
                                                           entity_quat[3]))
        
        else:
            entity.get_transform().set_rotation(entity_quat)