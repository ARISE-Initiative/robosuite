import xml.etree.ElementTree as ET

import numpy as np
from igibson.render.mesh_renderer.mesh_renderer_cpu import Material

from robosuite.renderers.base_parser import BaseParser
from robosuite.renderers.igibson.igibson_utils import MujocoCamera, MujocoRobot, get_texture_id, load_object
from robosuite.utils.mjcf_utils import string_to_array


class Parser(BaseParser):
    def __init__(self, renderer, env, segmentation_type):
        """
        Parse the mujoco xml and initialize iG renderer objects.

        Args:
            renderer: iGibson renderer
            env : Mujoco env
        """

        super().__init__(renderer, env)
        self.segmentation_type = segmentation_type
        self.create_class_mapping()

    def parse_textures(self):
        """
        Parse and load all textures and store them
        """

        self.texture_attributes = {}
        self.texture_id_mapping = {}

        for texture in self.xml_root.iter("texture"):
            texture_type = texture.get("type")
            texture_name = texture.get("name")
            texture_file = texture.get("file")
            texture_rgb = texture.get("rgb1")

            if texture_file is not None:
                self.texture_attributes[texture_name] = texture.attrib
                self.texture_id_mapping[texture_name] = (self.renderer.load_texture_file(texture_file, 1), texture_type)
            else:
                color = np.array(string_to_array(texture_rgb))
                self.texture_id_mapping[texture_name] = (color, texture_type)

    def parse_materials(self):
        """
        Parse all materials and use texture mapping to initialize materials
        """

        self.material_attributes = {}
        self.material_mapping = {}
        self.normal_id = get_texture_id(np.array([127, 127, 255]).reshape(1, 1, 3).astype(np.uint8), "normal", self)
        for material in self.xml_root.iter("material"):
            material_name = material.get("name")
            texture_name = material.get("texture")
            rgba = string_to_array(material.get("rgba", "0.5 0.5 0.5"))
            self.material_attributes[material_name] = material.attrib

            if texture_name is not None:
                texture_id, _ = self.texture_id_mapping[texture_name]
                specular = material.get("specular")
                shininess = material.get("shininess")
                roughness_id = -1 if specular is None else get_texture_id(1 - float(specular), "roughness", self)
                metallic_id = -1 if shininess is None else get_texture_id(float(shininess), "metallic", self)

                if isinstance(texture_id, int):
                    repeat = string_to_array(material.get("texrepeat", "1 1"))
                    self.material_mapping[material_name] = Material(
                        "texture",
                        texture_id=texture_id,
                        transform_param=[repeat[0], repeat[1], 0],
                        metallic_texture_id=metallic_id,
                        roughness_texture_id=roughness_id,
                        normal_texture_id=self.normal_id,
                    )
                else:
                    # texture id in this case will be a numpy array of rgb values
                    # If color are present both in material and texture, prioritize material color.
                    if material.get("rgba") is None:
                        self.material_mapping[material_name] = Material("color", kd=texture_id)
                    else:
                        self.material_mapping[material_name] = Material("color", kd=rgba[:3])

            else:
                # color can either come from texture, or could be defined in the material itself.
                self.material_mapping[material_name] = Material("color", kd=rgba[:3])

    def create_class_mapping(self):
        """
        Create class name to index mapping for both semantic and instance
        segmentation.
        """
        self.class2index = {}
        for i, c in enumerate(self.env.model._classes_to_ids.keys()):
            self.class2index[c] = i
        self.class2index[None] = i + 1
        self.max_classes = len(self.class2index)

        self.instance2index = {}
        for i, instance_class in enumerate(self.env.model._instances_to_ids.keys()):
            self.instance2index[instance_class] = i
        self.instance2index[None] = i + 1
        self.max_instances = len(self.instance2index)

    def get_class_id(self, geom_index, element_id):
        """
        Given index of the geom object get the class id based on
        self.segmentation type.
        """
        if self.segmentation_type == "class":
            class_id = self.class2index[self.env.model._geom_ids_to_classes.get(geom_index)]
        elif self.segmentation_type == "instance":
            class_id = self.instance2index[self.env.model._geom_ids_to_instances.get(geom_index)]
        else:
            class_id = element_id

        return class_id

    def parse_cameras(self):
        """
        Parse cameras and initialize the cameras.
        """

        robot = MujocoRobot()
        for cam in self.xml_root.iter("camera"):
            camera_name = cam.get("name")
            # get parent body name to find out where the camera is attached.
            parent_body_name = self.parent_map[cam].get("name", "worldbody")
            pos = string_to_array(cam.get("pos", "0 0 0"))
            quat = string_to_array(cam.get("quat", "1 0 0 0"))
            fov = float(cam.get("fovy", "45"))
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])
            camera = MujocoCamera(
                parent_body_name, pos, quat, active=False, mujoco_env=self.env, camera_name=camera_name, fov=fov
            )
            robot.cameras.append(camera)

        #self.renderer.add_robot([], [], [], [], None, 0, dynamic=False, robot=robot)
        self.renderer.add_instance_group([], ig_object=robot)

    def parse_meshes(self):
        """
        Create mapping of meshes.
        """
        self.meshes = {}
        for mesh in self.xml_root.iter("mesh"):
            self.meshes[mesh.get("name")] = mesh.attrib

    def parse_geometries(self):
        """
        Iterate through each goemetry and load it in the iGibson renderer.
        """
        self.parse_meshes()
        element_id = 0
        for geom_index, geom in enumerate(self.xml_root.iter("geom")):
            geom_name = geom.get("name", "NONAME")
            geom_type = geom.get("type", "sphere")

            if (geom.get("group") != "1" and geom_type != "plane") or ("collision" in geom_name):
                continue

            parent_body = self.parent_map.get(geom)
            # [1, 0, 0, 0] is wxyz, we convert it back to xyzw.
            geom_orn = string_to_array(geom.get("quat", "1 0 0 0"))
            geom_orn = [geom_orn[1], geom_orn[2], geom_orn[3], geom_orn[0]]
            geom_pos = string_to_array(geom.get("pos", "0 0 0"))
            if geom_type == "mesh":
                geom_scale = string_to_array(self.meshes[geom.get("mesh")].get("scale", "1 1 1"))
            else:
                geom_scale = [1, 1, 1]
            geom_size = string_to_array(geom.get("size", "1 1 1"))
            geom_rgba = string_to_array(geom.get("rgba", "1 1 1 1"))
            geom_material = self.material_mapping.get(geom.get("material"))
            if geom_material is None:
                color = geom_rgba[:3].reshape(1, 1, 3)
                # TODO: check converting the below texture to color material
                dummy_texture_id = get_texture_id(color, "texture", self)
                geom_material = Material(
                    "texture",
                    texture_id=dummy_texture_id,
                    metallic_texture_id=get_texture_id(1, "metallic", self),
                    roughness_texture_id=get_texture_id(1, "roughness", self),
                    normal_texture_id=self.normal_id,
                )
                # Flag to check if default material is used
                geom_material._is_set_by_parser = True
            else:
                geom_material._is_set_by_parser = False

            # saving original params because transform param will be overwritten
            if not hasattr(geom_material, "_orig_transform_param"):
                geom_material._orig_transform_param = geom_material.transform_param

            # setting material_ids so that randomized material works
            geom_material.material_ids = 0

            class_id = self.get_class_id(geom_index, element_id)

            load_object(
                renderer=self.renderer,
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
                class_id=class_id,
                visual_objects=self.visual_objects,
                meshes=self.meshes,
            )

            element_id += 1

        self.max_elements = element_id
