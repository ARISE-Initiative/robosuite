import os
import numpy as np
import tempfile
import random
import string
import xml.etree.ElementTree as ET

import robosuite

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, string_to_array

# import robosuite_model_zoo


def postprocess_model_xml(xml_str):
    """
    New version of postprocess model xml that only replaces robosuite file paths if necessary (otherwise
    there is an error with the "max" operation), and also replaces robosuite-model-zoo file paths
    if necessary.
    """

    path = os.path.split(robosuite.__file__)[0]
    path_split = path.split("/")

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue

        old_path_split = old_path.split("/")
        # maybe replace all paths to robosuite assets
        check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite"]
        if len(check_lst) > 0:
            ind = max(check_lst)  # last occurrence index
            new_path_split = path_split + old_path_split[ind + 1:]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)

        # maybe replace all paths to robosuite model zoo assets
        # check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite-model-zoo-dev"]
        # if len(check_lst) > 0:
        #     ind = max(check_lst)  # last occurrence index
        #     new_path_split = os.path.split(robosuite_model_zoo.__file__)[0].split("/")[:-1] + old_path_split[ind + 1:]
        #     new_path = "/".join(new_path_split)
        #     elem.set("file", new_path)

    return ET.tostring(root, encoding="utf8").decode("utf8")


class MJCFObject(MujocoXMLObject):
    """
    Blender object with support for changing the scaling
    """

    def __init__(
            self,
            name,
            mjcf_path,
            scale=1.0,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            friction=(0.95, 0.3, 0.1),
            margin=None,
            rgba=None,
    ):
        # get scale in x, y, z
        if isinstance(scale, float):
            scale = [scale, scale, scale]
        elif isinstance(scale, tuple) or isinstance(scale, list):
            assert len(scale) == 3
            scale = tuple(scale)
        else:
            raise Exception("got invalid scale: {}".format(scale))
        scale = np.array(scale)

        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.margin = margin

        self.rgba = rgba

        # read default xml
        xml_path = mjcf_path
        folder = os.path.dirname(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # modify mesh scales
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        for mesh in meshes:
            # if a scale already exists, multiply the scales
            scale_to_set = scale
            existing_scale = mesh.get("scale")
            if existing_scale is not None:
                scale_to_set = string_to_array(existing_scale) * scale
            mesh.set("scale", array_to_string(scale_to_set))

        # modify sites for collision (assumes we can just scale up the locations - may or may not work)
        for n in ["bottom_site", "top_site", "horizontal_radius_site"]:
            site = root.find("worldbody/body/site[@name='{}']".format(n))
            pos = string_to_array(site.get("pos"))
            pos = scale * pos
            site.set("pos", array_to_string(pos))

        # write modified xml (and make sure to postprocess any paths just in case)
        xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
        xml_str = postprocess_model_xml(xml_str)
        random_code = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        new_xml_path = os.path.join(folder, "{}.xml".format(random_code))
        f = open(new_xml_path, "w")
        f.write(xml_str)
        f.close()
        # print(f"Write to {new_xml_path}")

        # initialize object with new xml we wrote
        super().__init__(
            # xml_path_completion("objects/{}.xml".format(obj_name)),
            fname=new_xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

        # clean up xml - we don't need it anymore
        os.remove(new_xml_path)

    def _get_geoms(self, root, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a geom element

        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call

        Returns:
            list: array of (parent, child) tuples where the child element is a geom type
        """
        geom_pairs = super(MJCFObject, self)._get_geoms(root=root, _parent=_parent)

        # modify geoms according to the attributes
        for i, (parent, element) in enumerate(geom_pairs):
            element.set("solref", array_to_string(self.solref))
            element.set("solimp", array_to_string(self.solimp))
            element.set("density", str(self.density))
            element.set("friction", array_to_string(self.friction))
            if self.margin is not None:
                element.set("margin", str(self.margin))

            if (self.rgba is not None) and (element.get("group") == "1"):
                element.set("rgba", array_to_string(self.rgba))

        return geom_pairs