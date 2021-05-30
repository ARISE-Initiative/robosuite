import os
import random
import string
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import gibson2
from transforms3d import quaternions
import robosuite.utils.transform_utils as T
import robosuite

def load_object(renderer,
                geom,
                geom_name,
                geom_type,
                geom_orn,
                geom_pos,
                geom_rgba,
                geom_size,
                geom_scale,
                geom_material,
                parent_body,
                instance_id,
                visual_objects,
                meshes):


        primitive_shapes_path = {
            'box': os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj'),
            'cylinder': os.path.join(robosuite.models.assets_root, 'objects/meshes/cylinder.obj'),
            'sphere': os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj'),
            'plane': os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
        }
        
        # if not in primitive shapes, get path to mesh
        filename = primitive_shapes_path.get(geom_type)
        if filename is None:
            filename = meshes[geom.attrib['mesh']]['file']
            filename = os.path.splitext(filename)[0] + '.obj'
        
        parent_body_name = parent_body.get('name', 'worldbody')
        load_texture = True if geom_material is None else False

        # place holder meshes for which we do not textures loaded
        if geom_name in ['VisualBread_g0', 'VisualCan_g0', 'VisualCereal_g0', 'VisualMilk_g0']:
            load_texture = False

        if geom_name in ['Milk_g0_visual', 'Can_g0_visual', 'Cereal_g0_visual', 'Bread_g0_visual']:
            load_texture = True
    
        if geom_type == 'mesh':
            scale = geom_scale
        elif geom_type in ['box', 'sphere']:
            scale = geom_size * 2
        elif geom_type == 'cylinder':
            scale = [geom_size[0], geom_size[0], geom_size[1]]
        elif geom_type == 'plane':
            scale = [geom_size[0]*2 , geom_size[1]*2, 0.01]

        material = None if (geom_type == 'mesh' and geom_material._is_set_by_parser) \
                   else geom_material

        if "rethink_mount/pedestal.obj" in filename:
            filename = filename[:-4]
            filename += '_ig.obj'

        renderer.load_object(filename,
                             scale=scale,
                             transform_orn=geom_orn,
                             transform_pos=geom_pos,
                             input_kd=geom_rgba,
                             load_texture=load_texture,
                             input_material=material,  
                             geom_type=geom_type)

        if geom_type == 'mesh':
            visual_objects[filename] = len(renderer.visual_objects) - 1

        renderer.add_instance(len(renderer.visual_objects) - 1,
                              pybullet_uuid=0,
                              class_id=instance_id,
                              dynamic=True,
                              parent_body=parent_body_name)



def random_string():
    res = ''.join(random.choices(string.ascii_letters +
                        string.digits, k=10))
    return res

def get_id(intensity, name, self, resolution=20):
    #TODO: Fix the directory creation in optimized and non optimized case
    if isinstance(intensity, np.ndarray):
        # import pdb; pdb.set_trace();
        im = intensity
        im = np.tile(im, (resolution, resolution, 1))
    else:
        # im = np.array([intensity] * 3).reshape(1,1,3) * 255
        # im = im.astype(np.uint8)
        im = np.array([intensity] * (resolution ** 2)).reshape(resolution, resolution)
        
    tmpdirname = os.path.join(tempfile.gettempdir(), f'igibson_{random_string()}')
    os.makedirs(tmpdirname, exist_ok=True)
    fname = os.path.join(tmpdirname, f'{name}.png')
    plt.imsave(fname, im)
    print(fname)
    return self.renderer.load_texture_file(str(fname))


class MujocoRobot(object):
    def __init__(self):
        self.cameras = []

class MujocoCamera(object):
    """
    Camera class to define camera locations and its activation state (to render from them or not)
    """
    def __init__(self, 
                 camera_link_name, 
                 offset_pos = np.array([0,0,0]), 
                 offset_ori = np.array([0,0,0,1]), #xyzw -> Pybullet convention (to be consistent)
                 fov=45,
                 active=True, 
                 modes=None, 
                 camera_name=None,
                 mujoco_env=None,
                 ):
        """
        :param link_name: string, name of the link the camera is attached to
        :param offset_pos: vector 3d, position offset to the reference frame of the link
        :param offset_ori: vector 4d, orientation offset (quaternion: x, y, z, w) to the reference frame of the link
        :param active: boolean, whether the camera is active and we render virtual images from it
        :param modes: string, modalities rendered by this camera, a subset of ('rgb', 'normal', 'seg', '3d'). If None, we use the default of the renderer
        """
        self.camera_link_name = camera_link_name
        self.offset_pos = np.array(offset_pos)
        self.offset_ori = np.array(offset_ori)
        self.active = active
        self.modes = modes
        self.camera_name = [camera_name, camera_link_name + '_cam'][camera_name is None]
        self.mujoco_env = mujoco_env
        self.fov = fov

    def is_active(self):
        return self.active

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def switch(self):
        self.active = [True, False][self.active]

    def get_pose(self):
        offset_mat = np.eye(4)
        q_wxyz = np.concatenate((self.offset_ori[3:], self.offset_ori[:3]))
        offset_mat[:3, :3] = quaternions.quat2mat(q_wxyz)
        offset_mat[:3, -1] = self.offset_pos

        if self.camera_link_name != 'worldbody':

            pos_body_in_world = self.mujoco_env.sim.data.get_body_xpos(self.camera_link_name)
            rot_body_in_world = self.mujoco_env.sim.data.get_body_xmat(self.camera_link_name).reshape((3, 3))
            pose_body_in_world = T.make_pose(pos_body_in_world, rot_body_in_world) 

            total_pose = np.array(pose_body_in_world).dot(np.array(offset_mat))

            position = total_pose[:3, -1]

            rot = total_pose[:3, :3]
            wxyz = quaternions.mat2quat(rot)
            xyzw = np.concatenate((wxyz[1:], wxyz[:1]))

        else:
            position = np.array(self.offset_pos)
            xyzw = self.offset_ori

        return np.concatenate((position, xyzw))