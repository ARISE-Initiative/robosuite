# utility functions for manipulating MJCF XML models

import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

import robosuite

RED = [1, 0, 0, 1]
GREEN = [0, 1, 0, 1]
BLUE = [0, 0, 1, 1]
CYAN = [0, 1, 1, 1]
ROBOT_COLLISION_COLOR = [0, 0.5, 0, 1]
MOUNT_COLLISION_COLOR = [0.5, 0.5, 0, 1]
GRIPPER_COLLISION_COLOR = [0, 0, 0.5, 1]
OBJECT_COLLISION_COLOR = [0.5, 0, 0, 1]
ENVIRONMENT_COLLISION_COLOR = [0.5, 0.5, 0, 1]
SENSOR_TYPES = {
    "touch",
    "accelerometer",
    "velocimeter",
    "gyro",
    "force",
    "torque",
    "magnetometer",
    "rangefinder",
    "jointpos",
    "jointvel",
    "tendonpos",
    "tendonvel",
    "actuatorpos",
    "actuatorvel",
    "actuatorfrc",
    "ballangvel",
    "jointlimitpos",
    "jointlimitvel",
    "jointlimitfrc",
    "tendonlimitpos",
    "tendonlimitvel",
    "tendonlimitfrc",
    "framepos",
    "framequat",
    "framexaxis",
    "frameyaxis",
    "framezaxis",
    "framelinvel",
    "frameangvel",
    "framelinacc",
    "frameangacc",
    "subtreecom",
    "subtreelinvel",
    "subtreeangmom",
    "user",
}

MUJOCO_NAMED_ATTRIBUTES = {
    "class",
    "childclass",
    "name",
    "objname",
    "material",
    "texture",
    "joint",
    "joint1",
    "joint2",
    "jointinparent",
    "geom",
    "geom1",
    "geom2",
    "mesh",
    "fixed",
    "actuator",
    "objname",
    "tendon",
    "tendon1",
    "tendon2",
    "slidesite",
    "cranksite",
    "body",
    "body1",
    "body2",
    "hfield",
    "target",
    "prefix",
    "site",
}

IMAGE_CONVENTION_MAPPING = {
    "opengl": 1,
    "opencv": -1,
}

TEXTURE_FILES = {
    "WoodRed": "red-wood.png",
    "WoodGreen": "green-wood.png",
    "WoodBlue": "blue-wood.png",
    "WoodLight": "light-wood.png",
    "WoodDark": "dark-wood.png",
    "WoodTiles": "wood-tiles.png",
    "WoodPanels": "wood-varnished-panels.png",
    "WoodgrainGray": "gray-woodgrain.png",
    "PlasterCream": "cream-plaster.png",
    "PlasterPink": "pink-plaster.png",
    "PlasterYellow": "yellow-plaster.png",
    "PlasterGray": "gray-plaster.png",
    "PlasterWhite": "white-plaster.png",
    "BricksWhite": "white-bricks.png",
    "Metal": "metal.png",
    "SteelBrushed": "steel-brushed.png",
    "SteelScratched": "steel-scratched.png",
    "Brass": "brass-ambra.png",
    "Bread": "bread.png",
    "Can": "can.png",
    "Ceramic": "ceramic.png",
    "Cereal": "cereal.png",
    "Clay": "clay.png",
    "Dirt": "dirt.png",
    "Glass": "glass.png",
    "FeltGray": "gray-felt.png",
    "Lemon": "lemon.png",
}

TEXTURES = {
    texture_name: os.path.join("textures", texture_file) for (texture_name, texture_file) in TEXTURE_FILES.items()
}

ALL_TEXTURES = TEXTURES.keys()


class CustomMaterial(object):
    """
    Simple class to instantiate the necessary parameters to define an appropriate texture / material combo

    Instantiates a nested dict holding necessary components for procedurally generating a texture / material combo

    Please see http://www.mujoco.org/book/XMLreference.html#asset for specific details on
        attributes expected for Mujoco texture / material tags, respectively

    Note that the values in @tex_attrib and @mat_attrib can be in string or array / numerical form.

    Args:
        texture (None or str or 4-array): Name of texture file to be imported. If a string, should be part of
            ALL_TEXTURES. If texture is a 4-array, then this argument will be interpreted as an rgba tuple value and
            a template png will be procedurally generated during object instantiation, with any additional
            texture / material attributes specified. If None, no file will be linked and no rgba value will be set
            Note, if specified, the RGBA values are expected to be floats between 0 and 1

        tex_name (str): Name to reference the imported texture

        mat_name (str): Name to reference the imported material

        tex_attrib (dict): Any other optional mujoco texture specifications.

        mat_attrib (dict): Any other optional mujoco material specifications.

        shared (bool): If True, this material should not have any naming prefixes added to all names

    Raises:
        AssertionError: [Invalid texture]
    """

    def __init__(
        self,
        texture,
        tex_name,
        mat_name,
        tex_attrib=None,
        mat_attrib=None,
        shared=False,
    ):
        # Check if the desired texture is an rgba value
        if type(texture) is str:
            default = False
            # Verify that requested texture is valid
            texture_is_path = "/" in texture
            if not texture_is_path:
                assert (
                    texture in ALL_TEXTURES
                ), "Error: Requested invalid texture. Got {}. Valid options are:\n{}".format(texture, ALL_TEXTURES)
        else:
            default = True
            # If specified, this is an rgba value and a default texture is desired; make sure length of rgba array is 4
            if texture is not None:
                assert len(texture) == 4, (
                    "Error: Requested default texture. Got array of length {}."
                    "Expected rgba array of length 4.".format(len(texture))
                )

        # Setup the texture and material attributes
        self.tex_attrib = {} if tex_attrib is None else tex_attrib.copy()
        self.mat_attrib = {} if mat_attrib is None else mat_attrib.copy()

        # Add in name values
        self.name = mat_name
        self.shared = shared
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
            texture_is_path = "/" in texture
            if texture_is_path:
                self.tex_attrib["file"] = xml_path_completion(texture)
            else:
                self.tex_attrib["file"] = xml_path_completion(TEXTURES[texture])
        else:
            if texture is not None:
                # Create a texture patch
                tex = Image.new("RGBA", (100, 100), tuple((np.array(texture) * 255).astype("int")))
                # Create temp directory if it does not exist
                save_dir = "/tmp/robosuite_temp_tex"
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                # Save this texture patch to the temp directory on disk (MacOS / Linux)
                fpath = save_dir + "/{}.png".format(tex_name)
                tex.save(fpath, "PNG")
                # Link this texture file to the default texture dict
                self.tex_attrib["file"] = fpath


def xml_path_completion(xml_path, root=None):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package

    Args:
        xml_path (str): local xml path
        root (str): root folder for xml path. If not specified defaults to robosuite.models.assets_root

    Returns:
        str: Full (absolute) xml path
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        if root is None:
            root = robosuite.models.assets_root
        full_path = os.path.join(root, xml_path)
    return full_path


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"

    Args:
        array (n-array): Array to convert to a string

    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array])


def string_to_array(string):
    """
    Converts a array string in mujoco xml to np.array.

    Examples:
        "0 1 2" => [0, 1, 2]

    Args:
        string (str): String to convert to an array

    Returns:
        np.array: Numerical array equivalent of @string
    """
    return np.array([float(x) if x != "None" else None for x in string.strip().split(" ")])


def convert_to_string(inp):
    """
    Converts any type of {bool, int, float, list, tuple, array, string, np.str_} into an mujoco-xml compatible string.
        Note that an input string / np.str_ results in a no-op action.

    Args:
        inp: Input to convert to string

    Returns:
        str: String equivalent of @inp
    """
    if type(inp) in {list, tuple, np.ndarray}:
        return array_to_string(inp)
    elif type(inp) in {int, float, bool}:
        return str(inp).lower()
    elif type(inp) in {str, np.str_}:
        return inp
    else:
        raise ValueError("Unsupported type received: got {}".format(type(inp)))


def set_alpha(node, alpha=0.1):
    """
    Sets all a(lpha) field of the rgba attribute to be @alpha
    for @node and all subnodes
    used for managing display

    Args:
        node (ET.Element): Specific node element within XML tree
        alpha (float): Value to set alpha value of rgba tuple
    """
    for child_node in node.findall(".//*[@rgba]"):
        rgba_orig = string_to_array(child_node.get("rgba"))
        child_node.set("rgba", array_to_string(list(rgba_orig[0:3]) + [alpha]))


def new_element(tag, name, **kwargs):
    """
    Creates a new @tag element with attributes specified by @**kwargs.

    Args:
        tag (str): Type of element to create
        name (None or str): Name for this element. Should only be None for elements that do not have an explicit
            name attribute (e.g.: inertial elements)
        **kwargs: Specified attributes for the new joint

    Returns:
        ET.Element: new specified xml element
    """
    # Name will be set if it's not None
    if name is not None:
        kwargs["name"] = name
    # Loop through all attributes and pop any that are None, otherwise convert them to strings
    for k, v in kwargs.copy().items():
        if v is None:
            kwargs.pop(k)
        else:
            kwargs[k] = convert_to_string(v)
    element = ET.Element(tag, attrib=kwargs)
    return element


def new_joint(name, **kwargs):
    """
    Creates a joint tag with attributes specified by @**kwargs.

    Args:
        name (str): Name for this joint
        **kwargs: Specified attributes for the new joint

    Returns:
        ET.Element: new joint xml element
    """
    return new_element(tag="joint", name=name, **kwargs)


def new_actuator(name, joint, act_type="actuator", **kwargs):
    """
    Creates an actuator tag with attributes specified by @**kwargs.

    Args:
        name (str): Name for this actuator
        joint (str): type of actuator transmission.
            see all types here: http://mujoco.org/book/modeling.html#actuator
        act_type (str): actuator type. Defaults to "actuator"
        **kwargs: Any additional specified attributes for the new joint

    Returns:
        ET.Element: new actuator xml element
    """
    element = new_element(tag=act_type, name=name, **kwargs)
    element.set("joint", joint)
    return element


def new_site(name, rgba=RED, pos=(0, 0, 0), size=(0.005,), **kwargs):
    """
    Creates a site element with attributes specified by @**kwargs.

    NOTE: With the exception of @name, @pos, and @size, if any arg is set to
        None, the value will automatically be popped before passing the values
        to create the appropriate XML

    Args:
        name (str): Name for this site
        rgba (4-array): (r,g,b,a) color and transparency. Defaults to solid red.
        pos (3-array): (x,y,z) 3d position of the site.
        size (n-array of float): site size (sites are spherical by default).
        **kwargs: Any additional specified attributes for the new site

    Returns:
        ET.Element: new site xml element
    """
    kwargs["pos"] = pos
    kwargs["size"] = size
    kwargs["rgba"] = rgba if rgba is not None else None
    return new_element(tag="site", name=name, **kwargs)


def new_geom(name, type, size, pos=(0, 0, 0), group=0, **kwargs):
    """
    Creates a geom element with attributes specified by @**kwargs.

    NOTE: With the exception of @geom_type, @size, and @pos, if any arg is set to
        None, the value will automatically be popped before passing the values
        to create the appropriate XML

    Args:
        name (str): Name for this geom
        type (str): type of the geom.
            see all types here: http://mujoco.org/book/modeling.html#geom
        size (n-array of float): geom size parameters.
        pos (3-array): (x,y,z) 3d position of the site.
        group (int): the integrer group that the geom belongs to. useful for
            separating visual and physical elements.
        **kwargs: Any additional specified attributes for the new geom

    Returns:
        ET.Element: new geom xml element
    """
    kwargs["type"] = type
    kwargs["size"] = size
    kwargs["pos"] = pos
    kwargs["group"] = group if group is not None else None
    return new_element(tag="geom", name=name, **kwargs)


def new_body(name, pos=(0, 0, 0), **kwargs):
    """
    Creates a body element with attributes specified by @**kwargs.

    Args:
        name (str): Name for this body
        pos (3-array): (x,y,z) 3d position of the body frame.
        **kwargs: Any additional specified attributes for the new body

    Returns:
        ET.Element: new body xml element
    """
    kwargs["pos"] = pos
    return new_element(tag="body", name=name, **kwargs)


def new_inertial(pos=(0, 0, 0), mass=None, **kwargs):
    """
    Creates a inertial element with attributes specified by @**kwargs.

    Args:
        pos (3-array): (x,y,z) 3d position of the inertial frame.
        mass (float): The mass of inertial
        **kwargs: Any additional specified attributes for the new inertial element

    Returns:
        ET.Element: new inertial xml element
    """
    kwargs["mass"] = mass if mass is not None else None
    kwargs["pos"] = pos
    return new_element(tag="inertial", name=None, **kwargs)


def get_size(size, size_max, size_min, default_max, default_min, rng=None):
    """
    Helper method for providing a size, or a range to randomize from

    Args:
        size (n-array): Array of numbers that explicitly define the size
        size_max (n-array): Array of numbers that define the custom max size from which to randomly sample
        size_min (n-array): Array of numbers that define the custom min size from which to randomly sample
        default_max (n-array): Array of numbers that define the default max size from which to randomly sample
        default_min (n-array): Array of numbers that define the default min size from which to randomly sample
        rng (None or np.random.RandomState): Random number generator to use. If None, will use np.random.default_rng()

    Returns:
        np.array: size generated

    Raises:
        ValueError: [Inconsistent array sizes]
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(default_max) != len(default_min):
        raise ValueError(
            "default_max = {} and default_min = {}".format(str(default_max), str(default_min))
            + " have different lengths"
        )
    if size is not None:
        if (size_max is not None) or (size_min is not None):
            raise ValueError("size = {} overrides size_max = {}, size_min = {}".format(size, size_max, size_min))
    else:
        if size_max is None:
            size_max = default_max
        if size_min is None:
            size_min = default_min
        size = np.array([rng.uniform(size_min[i], size_max[i]) for i in range(len(default_max))])
    return np.array(size)


def add_to_dict(dic, fill_in_defaults=True, default_value=None, **kwargs):
    """
    Helper function to add key-values to dictionary @dic where each entry is its own array (list).
    Args:
        dic (dict): Dictionary to which new key / value pairs will be added. If the key already exists,
            will append the value to that key entry
        fill_in_defaults (bool): If True, will automatically add @default_value to all dictionary entries that are
            not explicitly specified in @kwargs
        default_value (any): Default value to fill (None by default)

    Returns:
        dict: Modified dictionary
    """
    # Get keys and length of array for a given entry in dic
    keys = set(dic.keys())
    n = len(list(keys)[0]) if keys else 0
    for k, v in kwargs.items():
        if k in dic:
            dic[k].append(v)
            keys.remove(k)
        else:
            dic[k] = [default_value] * n + [v] if fill_in_defaults else [v]
    # If filling in defaults, fill in remaining default values
    if fill_in_defaults:
        for k in keys:
            dic[k].append(default_value)
    return dic


def add_prefix(
    root,
    prefix,
    tags="default",
    attribs="default",
    exclude=None,
):
    """
    Find all element(s) matching the requested @tag, and appends @prefix to all @attributes if they exist.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through.
        prefix (str): Prefix to add to all specified attributes
        tags (str or list of str or set): Tag(s) to search for in this ElementTree. "Default" corresponds to all tags
        attribs (str or list of str or set): Element attribute(s) to append prefix to. "Default" corresponds
            to all attributes that reference names
        exclude (None or function): Filtering function that should take in an ET.Element or a string (attribute) and
            return True if we should exclude the given element / attribute from having any prefixes added
    """
    # Standardize tags and attributes to be a set
    if tags != "default":
        tags = {tags} if type(tags) is str else set(tags)
    if attribs == "default":
        attribs = MUJOCO_NAMED_ATTRIBUTES
    attribs = {attribs} if type(attribs) is str else set(attribs)

    # Check the current element for matching conditions
    if (tags == "default" or root.tag in tags) and (exclude is None or not exclude(root)):
        for attrib in attribs:
            v = root.get(attrib, None)
            # Only add prefix if the attribute exist, the current attribute doesn't already begin with prefix,
            # and the @exclude filter is either None or returns False
            if v is not None and not v.startswith(prefix) and (exclude is None or not exclude(v)):
                root.set(attrib, prefix + v)
    # Continue recursively searching through the element tree
    for r in root:
        add_prefix(root=r, prefix=prefix, tags=tags, attribs=attribs, exclude=exclude)


def add_material(root, naming_prefix="", custom_material=None):
    """
    Iterates through all element(s) in @root recursively and adds a material / texture to all visual geoms that don't
    already have a material specified.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through.
        naming_prefix (str): Adds this prefix to all material and texture names
        custom_material (None or CustomMaterial): If specified, will add this material to all visual geoms.
            Else, will add a default "no-change" material.

    Returns:
        4-tuple: (ET.Element, ET.Element, CustomMaterial, bool) (tex_element, mat_element, material, used)
            corresponding to the added material and whether the material was actually used or not.
    """
    # Initialize used as False
    used = False
    # First, make sure material is specified
    if custom_material is None:
        custom_material = CustomMaterial(
            texture=None,
            tex_name="default_tex",
            mat_name="default_mat",
            tex_attrib={
                "type": "cube",
                "builtin": "flat",
                "width": 100,
                "height": 100,
                "rgb1": np.ones(3),
                "rgb2": np.ones(3),
            },
        )
    # Else, check to make sure the custom material begins with the specified prefix and that it's unique
    if not custom_material.name.startswith(naming_prefix) and not custom_material.shared:
        custom_material.name = naming_prefix + custom_material.name
        custom_material.tex_attrib["name"] = naming_prefix + custom_material.tex_attrib["name"]
        custom_material.mat_attrib["name"] = naming_prefix + custom_material.mat_attrib["name"]
        custom_material.mat_attrib["texture"] = naming_prefix + custom_material.mat_attrib["texture"]

    # Check the current element for matching conditions
    if root.tag == "geom" and root.get("group", None) == "1" and root.get("material", None) is None:
        # Add a new material attribute to this geom
        root.set("material", custom_material.name)
        # Set used to True
        used = True
    # Continue recursively searching through the element tree
    for r in root:
        _, _, _, _used = add_material(root=r, naming_prefix=naming_prefix, custom_material=custom_material)
        # Update used
        used = used or _used
    # Lastly, return the new texture and material elements
    tex_element = new_element(tag="texture", **custom_material.tex_attrib)
    mat_element = new_element(tag="material", **custom_material.mat_attrib)
    return tex_element, mat_element, custom_material, used


def recolor_collision_geoms(root, rgba, exclude=None):
    """
    Iteratively searches through all elements starting with @root to find all geoms belonging to group 0 and set
    the corresponding rgba value to the specified @rgba argument. Note: also removes any material values for these
    elements.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through
        rgba (4-array): (R, G, B, A) values to assign to all geoms with this group.
        exclude (None or function): Filtering function that should take in an ET.Element and
            return True if we should exclude the given element / attribute from having its collision geom impacted.
    """
    # Check this body
    if root.tag == "geom" and root.get("group") in {None, "0"} and (exclude is None or not exclude(root)):
        root.set("rgba", array_to_string(rgba))
        root.attrib.pop("material", None)

    # Iterate through all children elements
    for r in root:
        recolor_collision_geoms(root=r, rgba=rgba, exclude=exclude)


def _element_filter(element, parent):
    """
    Default element filter to be used in sort_elements. This will filter for the following groups:

        :`'root_body'`: Top-level body element
        :`'bodies'`: Any body elements
        :`'joints'`: Any joint elements
        :`'actuators'`: Any actuator elements
        :`'sites'`: Any site elements
        :`'sensors'`: Any sensor elements
        :`'contact_geoms'`: Any geoms used for collision (as specified by group 0 (default group) geoms)
        :`'visual_geoms'`: Any geoms used for visual rendering (as specified by group 1 geoms)

    Args:
        element (ET.Element): Current XML element that we are filtering
        parent (ET.Element): Parent XML element for the current element

    Returns:
        str or None: Assigned filter key for this element. None if no matching filter is found.
    """
    # Check for actuator first since this is dependent on the parent element
    if parent is not None and parent.tag == "actuator":
        return "actuators"
    elif parent is not None and parent.tag == "composite":
        return "composite_geoms"
    elif element.tag == "joint":
        # Make sure this is not a tendon (this should not have a "joint", "joint1", or "joint2" attribute specified)
        if element.get("joint") is None and element.get("joint1") is None:
            return "joints"
    elif element.tag == "body":
        # If the parent of this does not have a tag "body", then this is the top-level body element
        if parent is None or parent.tag != "body":
            return "root_body"
        return "bodies"
    elif element.tag == "site":
        return "sites"
    elif element.tag in SENSOR_TYPES:
        return "sensors"
    elif element.tag == "geom":
        # Only get collision and visual geoms (group 0 / None, or 1, respectively)
        group = element.get("group")
        if group in {None, "0", "1"}:
            return "visual_geoms" if group == "1" else "contact_geoms"
    else:
        # If no condition met, return None
        return None


def sort_elements(root, parent=None, element_filter=None, _elements_dict=None):
    """
    Utility method to iteratively sort all elements based on @tags. This XML ElementTree will be parsed such that
    all elements with the same key as returned by @element_filter will be grouped as a list entry in the returned
    dictionary.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through
        parent (ET.Element): Parent of the root node. Default is None (no parent node initially)
        element_filter (None or function): Function used to filter the incoming elements. Should take in two
            ET.Elements (current_element, parent_element) and return a string filter_key if the element
            should be added to the list of values sorted by filter_key, and return None if no value should be added.
            If no element_filter is specified, defaults to self._element_filter.
        _elements_dict (dict): Dictionary that gets passed to recursive calls. Should not be modified externally by
            top-level call.

    Returns:
        dict: Filtered key-specific lists of the corresponding elements
    """
    # Initialize dictionary and element filter if None is set
    if _elements_dict is None:
        _elements_dict = {}
    if element_filter is None:
        element_filter = _element_filter

    # Parse this element
    key = element_filter(root, parent)
    if key is not None:
        # Initialize new entry in the dict if this is the first time encountering this value, otherwise append
        if key not in _elements_dict:
            _elements_dict[key] = [root]
        else:
            _elements_dict[key].append(root)

    # Loop through all possible subtrees for this XML recurisvely
    for r in root:
        _elements_dict = sort_elements(
            root=r, parent=root, element_filter=element_filter, _elements_dict=_elements_dict
        )

    return _elements_dict


def find_parent(root, child):
    """
    Find the parent element of the specified @child node, recurisvely searching through @root.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through.
        child (ET.Element): Child element whose parent is to be found

    Returns:
        None or ET.Element: Matching parent if found, else None
    """
    # Iterate through children (DFS), if the correct child element is found, then return the current root as the parent
    for r in root:
        if r == child:
            return root
        parent = find_parent(root=r, child=child)
        if parent is not None:
            return parent
    # If we get here, we didn't find anything ):
    return None


def find_elements(root, tags, attribs=None, return_first=True):
    """
    Find all element(s) matching the requested @tag and @attributes. If @return_first is True, then will return the
    first element found matching the criteria specified. Otherwise, will return a list of elements that match the
    criteria.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through.
        tags (str or list of str or set): Tag(s) to search for in this ElementTree.
        attribs (None or dict of str): Element attribute(s) to check against for a filtered element. A match is
            considered found only if all attributes match. Each attribute key should have a corresponding value with
            which to compare against.
        return_first (bool): Whether to immediately return once the first matching element is found.

    Returns:
        None or ET.Element or list of ET.Element: Matching element(s) found. Returns None if there was no match.
    """
    # Initialize return value
    elements = None if return_first else []

    # Make sure tags is list
    tags = [tags] if type(tags) is str else tags

    # Check the current element for matching conditions
    if root.tag in tags:
        matching = True
        if attribs is not None:
            for k, v in attribs.items():
                if root.get(k) != v:
                    matching = False
                    break
        # If all criteria were matched, add this to the solution (or return immediately if specified)
        if matching:
            if return_first:
                return root
            else:
                elements.append(root)
    # Continue recursively searching through the element tree
    for r in root:
        if return_first:
            elements = find_elements(tags=tags, attribs=attribs, root=r, return_first=return_first)
            if elements is not None:
                return elements
        else:
            found_elements = find_elements(tags=tags, attribs=attribs, root=r, return_first=return_first)
            pre_elements = deepcopy(elements)
            if found_elements:
                elements += found_elements if type(found_elements) is list else [found_elements]

    return elements if elements else None


def find_elements_by_substring(root, tags, substrings, attribs=None, return_first=False):
    """
    Find all element(s) matching the requested @substrings and @attributes. If @return_first is True, then will return the
    first element found matching the criteria specified. Otherwise, will return a list of elements that match the
    criteria.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through.
        tags (str or list of str or set): Tag(s) to search for in this ElementTree.
        substrings (str or list of str or set): Substring(s) to search for in this ElementTree.
        attribs (None or dict of str): Element attribute(s) to check against for a filtered element. A match is
            considered found only if all attributes match. Each attribute key should have a corresponding value with
            which to compare against.
        return_first (bool): Whether to immediately return once the first matching element is found.

    Returns:
        None or ET.Element or list of ET.Element: Matching element(s) found. Returns None if there was no match.
    """
    # Initialize return value
    elements = None if return_first else []

    # Make sure substrings is list
    substrings = [substrings] if type(substrings) is str else substrings
    elements = find_elements(root, tags, attribs=attribs, return_first=return_first)

    new_elements = []
    if elements is not None:
        for element in elements:
            for substring in substrings:
                if substring in element.get("name"):
                    new_elements.append(element)
                    break
    return new_elements if len(new_elements) > 0 else None


def find_parent(element, target):
    """
    Find the parent element of the target.
    """
    for child in element:
        if child == target:
            return element  # Found the parent
        parent = find_parent(child, target)
        if parent is not None:
            return parent
    return None


def save_sim_model(sim, fname):
    """
    Saves the current model xml from @sim at file location @fname.

    Args:
        sim (MjSim): XML file to save, in string form
        fname (str): Absolute filepath to the location to save the file
    """
    with open(fname, "w") as f:
        sim.save(file=f, format="xml")


def get_ids(sim, elements, element_type="geom", inplace=False):
    """
    Grabs the ids corresponding to @elements. If the inputted elements are already a list of ids, immediately
    returns that list.

    Args:
        sim (MjSim): Mujoco sim reference
        elements (str or list or int): Object(s) to grab ids for. Can be a string (name), a list of strings (names),
            or an int / list of ints (ids). Also supported are lists of mixed types
        element_type (str): Type of element to grab ids for.
            Options are {'body', 'geom', 'site', 'joint', 'actuator', 'sensor', 'tendon', 'camera', 'light'}
        inplace (bool): If True, will replace the inputted @elements list in-place

    Returns:
        list: id(s) corresponding to @elements
    """
    if type(elements) is not list:
        elements = [elements]
    if not inplace:
        elements = list(elements)

    assert element_type in [
        "geom",
        "body",
        "joint",
        "site",
        "actuator",
        "sensor",
        "tendon",
        "camera",
        "light",
    ], "Invalid element type"
    # Iterate through all elements and grab their corresponding IDs
    for i, element in enumerate(elements):
        if type(element) is not int:
            element_func = sim.model.__getattribute__("{}_name2id".format(element_type))
            elements[i] = element_func(element)

    return elements


def normalize_scale_array(scale):
    """
    Normalizes a scale factor to be a 3-element numpy array.

    Args:
        scale (float or array-like): Scale factor (1 or 3 dims)

    Returns:
        np.array: 3-element scale array

    Raises:
        ValueError: If scale is not scalar or 3-element array
    """
    scale_array = np.array(scale).flatten()
    if scale_array.size == 1:
        scale_array = np.repeat(scale_array, 3)
    elif scale_array.size != 3:
        raise ValueError("Scale must be a scalar or a 3-element array.")
    return scale_array


def scale_geom_element(element, scale_array):
    """
    Scales a single geom element's position and size.

    Args:
        element (ET.Element): Geom element to scale
        scale_array (np.array): 3-element scale array
    """
    g_pos = element.get("pos")
    g_size = element.get("size")

    if g_pos is not None:
        g_pos = array_to_string(string_to_array(g_pos) * scale_array)
        element.set("pos", g_pos)

    if g_size is not None:
        g_size_np = string_to_array(g_size)
        # Handle cases where size is not 3-dimensional
        if len(g_size_np) == 3:
            g_size_np = g_size_np * scale_array
        elif len(g_size_np) == 2:
            # For 2D size, assume [radius, height] for cylinders
            g_size_np[0] *= np.mean(scale_array[:2])  # Average scaling in x and y
            g_size_np[1] *= scale_array[2]  # Scaling in z
        elif len(g_size_np) == 1:
            g_size_np *= np.mean(scale_array)
        else:
            raise ValueError("Unsupported geom size dimensions.")
        g_size = array_to_string(g_size_np)
        element.set("size", g_size)


def scale_mesh_element(element, scale_array):
    """
    Scales a single mesh element.

    Args:
        element (ET.Element): Mesh element to scale
        scale_array (np.array): 3-element scale array
    """
    m_scale = element.get("scale")
    if m_scale is not None:
        m_scale = string_to_array(m_scale)
    else:
        m_scale = np.ones(3)
    m_scale *= scale_array
    element.set("scale", array_to_string(m_scale))


def scale_body_element(element, scale_array):
    """
    Scales a single body element's position.

    Args:
        element (ET.Element): Body element to scale
        scale_array (np.array): 3-element scale array
    """
    b_pos = element.get("pos")
    if b_pos is not None:
        b_pos = string_to_array(b_pos) * scale_array
        element.set("pos", array_to_string(b_pos))


def scale_joint_element(element, scale_array, scale_slide_joints=True):
    """
    Scales a single joint element's position and optionally range for slide joints.

    Args:
        element (ET.Element): Joint element to scale
        scale_array (np.array): 3-element scale array
        scale_slide_joints (bool): Whether to scale ranges for slide joints
    """
    j_pos = element.get("pos")
    if j_pos is not None:
        j_pos = string_to_array(j_pos) * scale_array
        element.set("pos", array_to_string(j_pos))

    # Scale joint ranges for slide joints if requested
    if scale_slide_joints:
        j_type = element.get("type", "hinge")  # Default joint type is 'hinge' if not specified
        j_range = element.get("range")
        if j_range is not None and j_type == "slide":
            # Get joint axis
            j_axis = element.get("axis", "1 0 0")  # Default axis is [1, 0, 0]
            j_axis = string_to_array(j_axis)
            axis_norm = np.linalg.norm(j_axis)
            if axis_norm > 0:
                axis_unit = j_axis / axis_norm
            else:
                # Avoid division by zero
                axis_unit = np.array([1.0, 0.0, 0.0])
            # Compute scaling factor along the joint axis
            s = np.linalg.norm(axis_unit * scale_array)
            # Scale the range
            j_range_vals = string_to_array(j_range)
            j_range_vals = j_range_vals * s
            element.set("range", array_to_string(j_range_vals))


def scale_site_element(element, scale_array):
    """
    Scales a single site element's position and size.

    Args:
        element (ET.Element): Site element to scale
        scale_array (np.array): 3-element scale array
    """
    s_pos = element.get("pos")
    if s_pos is not None:
        s_pos = string_to_array(s_pos) * scale_array
        element.set("pos", array_to_string(s_pos))

    s_size = element.get("size")
    if s_size is not None:
        s_size_np = string_to_array(s_size)
        if len(s_size_np) == 3:
            s_size_np = s_size_np * scale_array
        elif len(s_size_np) == 2:
            s_size_np[0] *= np.mean(scale_array[:2])  # Average scaling in x and y
            s_size_np[1] *= scale_array[2]  # Scaling in z
        elif len(s_size_np) == 1:
            s_size_np *= np.mean(scale_array)
        else:
            raise ValueError("Unsupported site size dimensions.")
        s_size = array_to_string(s_size_np)
        element.set("size", s_size)


def scale_mjcf_model(obj, asset_root, scale, get_elements_func, worldbody=None, scale_slide_joints=True):
    """
    Scales all elements (geoms, meshes, bodies, joints, sites) in an MJCF model.

    Args:
        obj (ET.Element): Root object element to scale
        asset_root (ET.Element): Asset root element containing meshes
        scale (float or array-like): Scale factor (1 or 3 dims)
        get_elements_func (callable): Function to get elements of a specific type from obj
        scale_slide_joints (bool): Whether to scale ranges for slide joints

    Returns:
        np.array: The normalized 3-element scale array that was applied
    """
    # Normalize scale to 3-element array
    scale_array = normalize_scale_array(scale)

    # Scale geoms
    geom_pairs = get_elements_func(obj, "geom")
    for _, (_, element) in enumerate(geom_pairs):
        scale_geom_element(element, scale_array)

    # Scale meshes
    meshes = asset_root.findall("mesh")
    for elem in meshes:
        scale_mesh_element(elem, scale_array)

    # Scale bodies
    body_pairs = get_elements_func(obj, "body")
    for (_, elem) in body_pairs:
        scale_body_element(elem, scale_array)

    # Scale joints
    joint_pairs = get_elements_func(obj, "joint")
    for (_, elem) in joint_pairs:
        scale_joint_element(elem, scale_array, scale_slide_joints)

    # Scale sites
    if worldbody is not None:
        site_pairs = get_elements_func(worldbody, "site")
        for (_, elem) in site_pairs:
            scale_site_element(elem, scale_array)

    return scale_array


def get_elements(root, element_type, _parent=None):
    """
    Helper function to recursively search through element tree starting at @root and returns
    a list of (parent, child) tuples where the child is a specific type of element

    Args:
        root (ET.Element): Root of xml element tree to start recursively searching through
        element_type (str): Type of element to search for (e.g., "geom", "body", "joint", "site")
        _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
            during the recursive call

    Returns:
        list: array of (parent, child) tuples where the child element is of specified type
    """
    # Initialize return array
    elem_pairs = []
    # If the parent exists and this is a desired element, we add this current (parent, element) combo to the output
    if _parent is not None and root.tag == element_type:
        elem_pairs.append((_parent, root))
    # Loop through all children elements recursively and add to pairs
    for child in root:
        elem_pairs += get_elements(child, element_type, _parent=root)

    # Return all found pairs
    return elem_pairs
