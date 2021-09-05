import os
import random
from robosuite.utils.observables import Observable
import string
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import igibson
from transforms3d import quaternions
import robosuite.utils.transform_utils as T
import robosuite

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# these robots have material defined in mtl files.
# Update this when mtl files are defined for other robots.
ROBOTS_WITH_MATERIALS_DEFINED_IN_MTL = {'panda', 'sawyer'}

# place holder meshes for which we do not textures loaded
# This list should be extended if one sees placeholder meshes
# having textures. Place holder meshes
MESHES_FOR_NO_LOAD_TEXTURE = {'VisualBread_g0', 'VisualCan_g0', 'VisualCereal_g0', 'VisualMilk_g0'}

# Special meshes for which we have to load the textures
MESHES_FOR_LOAD_TEXTURE = {'Milk_g0_visual', 'Can_g0_visual', 'Cereal_g0_visual', 'Bread_g0_visual'}


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
    """
    Function that initializes the meshes in the memeory with appropriate materials.
    """
    
    primitive_shapes_path = {
        'box': os.path.join(igibson.assets_path, 'models/mjcf_primitives/cube.obj'),
        'cylinder': os.path.join(robosuite.models.assets_root, 'objects/meshes/cylinder.obj'),
        'sphere': os.path.join(igibson.assets_path, 'models/mjcf_primitives/sphere8.obj'),
        'plane': os.path.join(igibson.assets_path, 'models/mjcf_primitives/cube.obj')
    }
    
    # if not in primitive shapes, get path to mesh
    filename = primitive_shapes_path.get(geom_type)
    if filename is None:
        filename = meshes[geom.attrib['mesh']]['file']
        filename = os.path.splitext(filename)[0] + '.obj'
    
    mesh_file_name = Path(filename).parents[1].name
    parent_body_name = parent_body.get('name', 'worldbody')

    load_texture = geom_material is None or geom_material._is_set_by_parser

    if geom_name in MESHES_FOR_NO_LOAD_TEXTURE:
        load_texture = False

    if geom_name in MESHES_FOR_LOAD_TEXTURE:
        load_texture = True

    if geom_type == 'mesh':
        scale = geom_scale
    elif geom_type in ['box', 'sphere']:
        scale = geom_size * 2
    elif geom_type == 'cylinder':
        scale = [geom_size[0], geom_size[0], geom_size[1]]
    elif geom_type == 'plane':
        scale = [geom_size[0]*2 , geom_size[1]*2, 0.01]

    # If only color of the robot mesh is defined we add some metallic and specular by default which makes it look a bit nicer.
    material = None if (geom_type == 'mesh' and geom_material._is_set_by_parser and mesh_file_name in ROBOTS_WITH_MATERIALS_DEFINED_IN_MTL) \
                else geom_material

    renderer.load_object(filename,
                        scale=scale,
                        transform_orn=geom_orn,
                        transform_pos=geom_pos,
                        input_kd=geom_rgba,
                        load_texture=load_texture,
                        overwrite_material=material)

    if geom_type == 'mesh':
        visual_objects[filename] = len(renderer.visual_objects)
    
    # do not use pbr if robots have already defined materials.
    use_pbr_mapping = mesh_file_name not in ROBOTS_WITH_MATERIALS_DEFINED_IN_MTL

    renderer.add_instance(len(renderer.visual_objects) - 1,
                            pybullet_uuid=0,
                            class_id=instance_id,
                            dynamic=True,
                            use_pbr_mapping=use_pbr_mapping,
                            parent_body=parent_body_name)



def random_string():
    """
    Generate a random string.
    """
    res = ''.join(random.choices(string.ascii_letters +
                        string.digits, k=10))
    return res

def adjust_convention(img, convention):
    """
    Inverts (could) the image according to the given convention

    Args:
        img (np.ndarray): Image numpy array
        convention (int): -1 or 1 depending on macros.IMAGE_CONVENTION

    Returns:
        np.ndarray: Inverted or non inverted (vertically) image.
    """
    img = img[::-convention]
    return img

def get_texture_id(intensity, name, self, resolution=1):
    """
    Create dummy png or size resolution from intensity values and load the texture in renderer.

    Args:
        intensity (float, np.ndarray): Could be any intensity value
        name (string): name which will be used to save the file
        resolution (int, optional): Resolution at which dummy texture file is written.
                                    Defaults to 1.

    Returns:
        int: texture_id of iG renderer.
    """
    if isinstance(intensity, np.ndarray):
        im = intensity
        im = np.tile(im, (resolution, resolution, 1))
    else:
        im = np.array([intensity] * (resolution ** 2)).reshape(resolution, resolution)
    
    if not self.renderer.rendering_settings.optimized:
        tmpdirname = tempfile.TemporaryDirectory().name
    else:
        # if optimized the file created should stay on the disk as it is required later
        # by the optimized renderer. In non-optimized case the file written could be deleted.
        tmpdirname = os.path.join(tempfile.gettempdir(), f'igibson_{random_string()}')
    os.makedirs(tmpdirname, exist_ok=True)
    fname = os.path.join(tmpdirname, f'{name}.png')
    plt.imsave(fname, im)
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
        self.active = not self.active

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


class TensorObservable(Observable):
    """
    Extends observable class to handle torch tensors.

    """

    def update(self, timestep, obs_cache, force):
        """
        Updates internal values for this observable, if enabled.

        Args:
            timestep (float): Amount of simulation time (in sec) that has passed since last call.
            obs_cache (dict): Observation cache mapping observable names to pre-computed values to pass to sensor. This
                will be updated in-place during this call.
            force (bool): If True, will force the observable to update its internal value to the newest value.
        """
        if self._enabled:
            # Increment internal time counter
            self._time_since_last_sample += timestep

            # If the delayed sampling time has been passed and we haven't sampled yet for this sampling period,
            # we should grab a new measurement
            if (not self._sampled and self._sampling_timestep - self._current_delay >= self._time_since_last_sample) or\
                    force:
                obs = self._sensor(obs_cache)
                torch_tensor = False
                if self.modality == 'image':
                    if HAS_TORCH and isinstance(obs, torch.Tensor):
                        torch_tensor = True                
                # Get newest raw value, corrupt it, filter it, and set it as our current observed value
                obs = self._filter(self._corrupter(obs))
                if not torch_tensor:
                    obs = np.array(obs)
                self._current_observed_value = obs[0] if len(obs.shape) == 1 and obs.shape[0] == 1 else obs
                # Update cache entry as well
                if torch_tensor:
                    obs_cache[self.name] = self._current_observed_value
                else:
                    obs_cache[self.name] = np.array(self._current_observed_value)
                # Toggle sampled and re-sample next time delay
                self._sampled = True
                self._current_delay = self._delayer()

            # If our total time since last sample has surpassed our sampling timestep,
            # then we reset our timer and sampled flag
            if self._time_since_last_sample >= self._sampling_timestep:
                if not self._sampled:
                    # If we still haven't sampled yet, sample immediately and warn user that sampling rate is too low
                    print(f"Warning: sampling rate for observable {self.name} is either too low or delay is too high. "
                          f"Please adjust one (or both)")
                    # Get newest raw value, corrupt it, filter it, and set it as our current observed value
                    obs = np.array(self._filter(self._corrupter(self._sensor(obs_cache))))
                    self._current_observed_value = obs[0] if len(obs.shape) == 1 and obs.shape[0] == 1 else obs
                    # Update cache entry as well
                    obs_cache[self.name] = np.array(self._current_observed_value)
                    # Re-sample next time delay
                    self._current_delay = self._delayer()
                self._time_since_last_sample %= self._sampling_timestep
                self._sampled = False

    def _check_sensor_validity(self):
        """
        Internal function that checks the validity of this observable's sensor. It does the following:

            - Asserts that the inputted sensor has its __modality__ attribute defined from the sensor decorator
            - Asserts that the inputted sensor can handle the empty dict {} arg case
            - Updates the corresponding name, and data-types for this sensor
        """
        try:
            _ = self.modality
            img = self._sensor({})
            if isinstance(img, (np.ndarray, list, int, float)):
                self._data_shape = np.array(img).shape
                self._is_number = len(self._data_shape) == 1 and self._data_shape[0] == 1
            else:
                # torch tensor.shape returns torch.Size object, hence casted to tuple
                self._data_shape = tuple(img.shape)
            self._is_number = len(self._data_shape) == 1 and self._data_shape[0] == 1
        except Exception as e:
            raise ValueError("Current sensor for observable {} is invalid.".format(self.name))