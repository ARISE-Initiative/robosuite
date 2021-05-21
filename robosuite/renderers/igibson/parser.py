import xml.etree.ElementTree as ET

from igibson_utils import get_id, MujocoCamera, MujocoRobot, load_object
from robosuite.utils.mjcf_utils import string_to_array
import numpy as np

from gibson2.render.mesh_renderer.mesh_renderer_cpu import Material
from gibson2.utils.mesh_util import xyzw2wxyz #xyz2mat, quat2rotmat,

class Parser():

    def __init__(self, renderer,  env):

        self.renderer = renderer
        self.env = env
        self.xml_root = ET.fromstring(self.env.mjpy_model.get_xml())
        self.parent_map = {c:p for p in self.xml_root.iter() for c in p}
        self.visual_objects = {}
    
    def parse_textures(self):

        self.texture_attributes = {}
        self.texture_id_mapping = {}

        for texture in self.xml_root.iter('texture'):
            texture_type = texture.get('type')
            texture_name = texture.get('name')
            texture_file = texture.get('file')
            texture_rgb = texture.get('rgb1')

            if texture_file is not None:
                self.texture_attributes[texture_name] = texture.attrib
                self.texture_id_mapping[texture_name] = (self.renderer.load_texture_file(texture_file), texture_type)
            else:
                color = np.array(string_to_array(texture_rgb))
                self.texture_id_mapping[texture_name] = (color, texture_type)

    def parse_materials(self):

        self.material_attributes = {}
        self.material_mapping = {}
        self.normal_id = get_id(np.array([127, 127, 255]).reshape(1,1,3).astype(np.uint8), 'normal', self)
        for material in self.xml_root.iter('material'):
            material_name = material.get('name')
            texture_name = material.get('texture')
            rgba = string_to_array(material.get('rgba', '0.5 0.5 0.5'))
            self.material_attributes[material_name] = material.attrib

            if texture_name is not None:
                texture_id, texture_type = self.texture_id_mapping[texture_name]
                specular = material.get('specular')
                shininess = material.get('shininess')
                roughness_id = -1 if specular is None else get_id(1 - float(specular), 'roughness', self)
                metallic_id = -1 if shininess is None else get_id(float(shininess), 'metallic', self)

                if isinstance(texture_id, int):
                    repeat = string_to_array(material.get('texrepeat', "1 1"))
                    texuniform = material.get('texuniform', 'false') == 'true'
                    self.material_mapping[material_name] = Material('texture',
                                                                    texture_id=texture_id,
                                                                    repeat_x=repeat[0], 
                                                                    repeat_y=repeat[1], 
                                                                    metallic_texture_id=metallic_id,
                                                                    roughness_texture_id=roughness_id,
                                                                    normal_texture_id=self.normal_id,
                                                                    texuniform=texuniform, 
                                                                    texture_type=texture_type)
                else:
                    # texture id in this case will be a numpy array of rgb values
                    self.material_mapping[material_name] = Material('color',
                                                                    kd=texture_id)
            else:
                # color can either come from texture, or could be defined in the material itself.
                self.material_mapping[material_name] = Material('color',
                                                                kd=rgba[:3])

    def parse_cameras(self):

        #TODO: Fix the logic here and figure out the need for MuJoCo camera class
        #TODO: Fix camera modes
        robot = MujocoRobot()
        for cam in self.xml_root.iter('camera'):
            camera_name = cam.get('name')
            # get parent body name.
            parent_body_name = self.parent_map[cam].get('name', 'worldbody')
            camera = MujocoCamera(parent_body_name, 
                                  string_to_array(cam.get('pos', "0 0 0")),
                                  xyzw2wxyz(string_to_array(cam.get('quat', "1 0 0 0"))),
                                #   modes='seg',
                                  active=False, #True if self.camera_name == camm.get("name") else False, , 
                                  mujoco_env=self.env, 
                                  camera_name=camera_name)
            robot.cameras.append(camera)    

        self.renderer.add_robot([],
                                [],
                                [],
                                [],
                                [],
                                0,
                                dynamic=False,
                                robot=robot)                       

    def parse_meshes(self):
        self.meshes = {}
        for mesh in self.xml_root.iter('mesh'):
            self.meshes[mesh.get('name')] = mesh.attrib

    def parse_geometries(self):
        self.parse_meshes()
        for instance_id, geom in enumerate(self.xml_root.iter('geom')):
            geom_name = geom.get('name')
            geom_type = geom.get('type')

            if geom.get('group') != '1': #  and geom_type != 'plane':
                continue

            parent_body = self.parent_map.get(geom)
            # [1, 0, 0, 0] is wxyz, we convert it back to xyzw.
            # The imported function performs the same function
            geom_orn = xyzw2wxyz(string_to_array(geom.get('quat', '1 0 0 0')))
            geom_pos = string_to_array(geom.get('pos', "0 0 0"))
            geom_scale = string_to_array(geom.get('scale', "1 1 1"))
            geom_size = string_to_array(geom.get('size', "1 1 1"))
            geom_rgba = string_to_array(geom.get('rgba', "1 1 1 1"))
            geom_material = self.material_mapping.get(geom.get('material'))
            if geom_material is None:
                color = geom_rgba[:3].reshape(1,1,3)
                #TODO: check converting the below texture to color material
                dummy_texture_id = get_id(color, 'texture', self)
                geom_material = Material('texture',
                                         texture_id=dummy_texture_id,
                                         metallic_texture_id=-1,
                                         roughness_texture_id=-1,
                                         normal_texture_id=self.normal_id,
                                         texuniform=False)
            
            load_object(renderer=self.renderer,
                        geom=geom,
                        geom_name=geom_name,
                        geom_type=geom_type,
                        geom_orn=geom_orn,
                        geom_pos=geom_pos,
                        geom_rgba=geom_rgba,
                        geom_size=geom_size,
                        geom_scale=geom_scale,
                        geom_material=geom_material,
                        parent_body=parent_body,
                        instance_id=instance_id,
                        visual_objects=self.visual_objects,
                        meshes=self.meshes
                        )

            

            

        