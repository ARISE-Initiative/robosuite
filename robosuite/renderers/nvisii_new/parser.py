import numpy as np
import xml.etree.ElementTree as ET

from robosuite.utils.mjcf_utils import string_to_array
from nvisii_utils import load_object

class Parser():
    def __init__(self, env):
        """
        Parse the mujoco xml and initialize NViSII renderer objects.
        Args:
            renderer: iGibson renderer
            env : Mujoco env
        """

        self.env = env
        self.xml_root = ET.fromstring(self.env.mjpy_model.get_xml())      
        self.parent_map = {c:p for p in self.xml_root.iter() for c in p}
        self.visual_objects = {}
        self.components = {}

    def parse_meshes(self):
        """
        Create mapping of meshes.
        """
        self.meshes = {}
        for mesh in self.xml_root.iter('mesh'):
            self.meshes[mesh.get('name')] = mesh.attrib

    def parse_geometries(self):
        """
        Iterate through each goemetry and load it in the NViSII renderer.
        """
        self.parse_meshes()
        instance_id = 0
        for geom in self.xml_root.iter('geom'):
            geom_name = geom.get('name', 'NONAME')
            geom_type = geom.get('type')

            if 'floor' in geom_name or 'wall' in geom_name:
                continue

            if (geom.get('group') != '1' and geom_type != 'plane') or ('collision' in geom_name):
                continue
            
            parent_body = self.parent_map.get(geom)
            parent_body_name = parent_body.get('name', 'worldbody')
            
            geom_quat = string_to_array(geom.get('quat', '1 0 0 0'))
            geom_quat = [geom_quat[0], geom_quat[1], geom_quat[2], geom_quat[3]]
            geom_pos = string_to_array(geom.get('pos', "0 0 0"))

            if geom_type == 'mesh':
                geom_scale = string_to_array(self.meshes[geom.get('mesh')].get('scale', '1 1 1') )
            else:
                geom_scale = [1, 1, 1]
            geom_size = string_to_array(geom.get('size', "1 1 1"))

            # load obj into nvisii
            component = load_object(geom=geom,
                                    geom_name=geom_name,
                                    geom_type=geom_type,
                                    geom_quat=geom_quat,
                                    geom_pos=geom_pos,
                                    geom_size=geom_size,
                                    geom_scale=geom_scale,
                                    instance_id=instance_id,
                                    visual_objects=self.visual_objects,
                                    meshes=self.meshes
                                    )

            self.components[geom_name] = (component, parent_body_name, geom_quat)
