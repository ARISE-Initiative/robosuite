# utility functions for manipulating MJCF XML models

import xml.etree.ElementTree as ET
import os
import numpy as np
from collections.abc import Iterable
from PIL import Image

import robosuite

RED = [1, 0, 0, 1]
GREEN = [0, 1, 0, 1]
BLUE = [0, 0, 1, 1]

TEXTURES = {
    "WoodRed": "red-wood.png",
    "WoodGreen": "green-wood.png",
    "WoodBlue": "blue-wood.png",
    "WoodLight": "light-wood.png",
    "WoodDark": "dark-wood.png",
    "WoodTiles": "wood-tiles.png",
    "WoodPanels": "wood-varnished-panels.png",
    "WoodgrainGray": "gray-woodgrain.png",
    "Metal": "metal.png",
    "SteelBrushed": "steel-brushed.png",
    "SteelScratched": "steel-scratched.png",
    "Brass": "brass-ambra.png",
    "Bread": "bread.png",
    "Can": "can.png",
    "Ceramic": "ceramic.png",
    "Cereal": "cereal.png",
    "Clay": "clay.png",
    "Glass": "glass.png",
    "FeltGray": "gray-felt.png",
    "Lemon": "lemon.png",
}

ALL_TEXTURES = TEXTURES.keys()


def xml_path_completion(xml_path):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        full_path = os.path.join(robosuite.models.assets_root, xml_path)
    return full_path


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"
    """
    return " ".join(["{}".format(x) for x in array])


def string_to_array(string):
    """
    Converts a array string in mujoco xml to np.array.

    Examples:
        "0 1 2" => [0, 1, 2]
    """
    return np.array([float(x) for x in string.split(" ")])


def set_alpha(node, alpha=0.1):
    """
    Sets all a(lpha) field of the rgba attribute to be @alpha
    for @node and all subnodes
    used for managing display
    """
    for child_node in node.findall(".//*[@rgba]"):
        rgba_orig = string_to_array(child_node.get("rgba"))
        child_node.set("rgba", array_to_string(list(rgba_orig[0:3]) + [alpha]))


def new_joint(**kwargs):
    """
    Creates a joint tag with attributes specified by @**kwargs.
    """

    element = ET.Element("joint", attrib=kwargs)
    return element


def new_actuator(joint, act_type="actuator", **kwargs):
    """
    Creates an actuator tag with attributes specified by @**kwargs.

    Args:
        joint: type of actuator transmission.
            see all types here: http://mujoco.org/book/modeling.html#actuator
        act_type (str): actuator type. Defaults to "actuator"

    """
    element = ET.Element(act_type, attrib=kwargs)
    element.set("joint", joint)
    return element


def new_site(name, rgba=RED, pos=(0, 0, 0), size=(0.005,), **kwargs):
    """
    Creates a site element with attributes specified by @**kwargs.

    Args:
        name (str): site name.
        rgba: color and transparency. Defaults to solid red.
        pos: 3d position of the site.
        size ([float]): site size (sites are spherical by default).

    NOTE: With the exception of @name, @pos, and @size, if any arg is set to
        None, the value will automatically be popped before passing the values
        to create the appropriate XML
    """
    kwargs["name"] = name
    kwargs["pos"] = array_to_string(pos)
    kwargs["size"] = array_to_string(size)
    kwargs["rgba"] = array_to_string(rgba) if rgba is not None else None
    # Loop through all remaining attributes and pop any that are None
    for k, v in kwargs.copy().items():
        if v is None:
            kwargs.pop(k)
    element = ET.Element("site", attrib=kwargs)
    return element


def new_geom(geom_type, size, pos=(0, 0, 0), rgba=RED, group=0, **kwargs):
    """
    Creates a geom element with attributes specified by @**kwargs.

    Args:
        geom_type (str): type of the geom.
            see all types here: http://mujoco.org/book/modeling.html#geom
        size: geom size parameters.
        pos: 3d position of the geom frame.
        rgba: color and transparency. Defaults to solid red.
        group: the integrer group that the geom belongs to. useful for
            separating visual and physical elements.

    NOTE: With the exception of @geom_type, @size, and @pos, if any arg is set to
        None, the value will automatically be popped before passing the values
        to create the appropriate XML
    """
    kwargs["type"] = str(geom_type)
    kwargs["size"] = array_to_string(size)
    kwargs["pos"] = array_to_string(pos)
    kwargs["rgba"] = array_to_string(rgba) if rgba is not None else None
    kwargs["group"] = str(group) if group is not None else None
    # Loop through all remaining attributes and pop any that are None
    for k, v in kwargs.copy().items():
        if v is None:
            kwargs.pop(k)
    element = ET.Element("geom", attrib=kwargs)
    return element


def new_body(name=None, pos=None, **kwargs):
    """
    Creates a body element with attributes specified by @**kwargs.

    Args:
        name (str): body name.
        pos: 3d position of the body frame.
    """
    if name is not None:
        kwargs["name"] = name
    if pos is not None:
        kwargs["pos"] = array_to_string(pos)
    element = ET.Element("body", attrib=kwargs)
    return element


def new_inertial(name=None, pos=(0, 0, 0), mass=None, **kwargs):
    """
    Creates a inertial element with attributes specified by @**kwargs.

    Args:
        mass: The mass of inertial
    """
    if mass is not None:
        kwargs["mass"] = str(mass)
    kwargs["pos"] = array_to_string(pos)
    element = ET.Element("inertial", attrib=kwargs)
    return element


def postprocess_model_xml(xml_str):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.
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
        ind = max(
            loc for loc, val in enumerate(old_path_split) if val == "robosuite"
        )  # last occurrence index
        new_path_split = path_split + old_path_split[ind + 1 :]
        new_path = "/".join(new_path_split)
        elem.set("file", new_path)

    return ET.tostring(root, encoding="utf8").decode("utf8")


class CustomMaterial(object):
    """
    Simple class to instantiate the necessary parameters to define an appropriate texture / material combo
    """

    def __init__(
            self,
            texture,
            tex_name,
            mat_name,
            tex_attrib=None,
            mat_attrib=None,
    ):
        """
        Instantiates a nested dict holding necessary components for procedurally generating a texture / material combo

        Args:
            texture (str or 4-array): Name of texture file to be imported. If a string, should be part of ALL_TEXTURES
                If texture is a 4-array, then this argument will be interpreted as an rgba tuple value and a template
                    png will be procedurally generated during object instantiation, with any additional
                    texture / material attributes specified.
                    Note the RGBA values are expected to be floats between 0 and 1
            tex_name (str): Name to reference the imported texture
            mat_name (str): Name to reference the imported material
            tex_attrib (dict): Any other optional mujoco texture specifications.
            mat_attrib (dict): Any other optional mujoco material specifications.

            Please see http://www.mujoco.org/book/XMLreference.html#asset for specific details on
                attributes expected for Mujoco texture / material tags, respectively

            Note that the values in @tex_attrib and @mat_attrib can be in string or array / numerical form.
        """
        # Check if the desired texture is an rgba value
        if type(texture) is str:
            default = False
            # Verify that requested texture is valid
            assert texture in ALL_TEXTURES, "Error: Requested invalid texture. Got {}. Valid options are:\n{}".format(
                texture, ALL_TEXTURES)
        else:
            default = True
            # This is an rgba value and a default texture is desired; make sure length of rgba array is 4
            assert len(texture) == 4, "Error: Requested default texture. Got array of length {}. Expected rgba array " \
                                      "of length 4.".format(len(texture))

        # Setup the texture and material attributes
        self.tex_attrib = {} if tex_attrib is None else tex_attrib.copy()
        self.mat_attrib = {} if mat_attrib is None else mat_attrib.copy()

        # Add in name values
        self.tex_attrib["name"] = tex_name
        self.mat_attrib["name"] = mat_name
        self.mat_attrib["texture"] = tex_name

        # Loop through all attributes and convert all non-string values into strings
        for attrib in (self.tex_attrib, self.mat_attrib):
            for k, v in attrib.items():
                if type(v) is not str:
                    if isinstance(v, Iterable):
                        attrib[k] = array_to_string(v)
                    else:
                        attrib[k] = str(v)

        # Handle default and non-default cases separately for linking texture patch file locations
        if not default:
            # Add in the filepath to texture patch
            self.tex_attrib["file"] = xml_path_completion("textures/" + TEXTURES[texture])
        else:
            # Create a texture patch
            tex = Image.new('RGBA', (100, 100), tuple((np.array(texture)*255).astype('int')))
            # Save this texture patch to the temp directory on disk (MacOS / Linux)
            fpath = "/tmp/{}.png".format(tex_name)
            tex.save(fpath, "PNG")
            # Link this texture file to the default texture dict
            self.tex_attrib["file"] = fpath
