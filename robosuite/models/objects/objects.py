import copy
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np

import robosuite.macros as macros
from robosuite.models.base import MujocoModel, MujocoXML
from robosuite.utils.mjcf_utils import (
    OBJECT_COLLISION_COLOR,
    CustomMaterial,
    add_material,
    add_prefix,
    array_to_string,
    find_elements,
    get_elements,
    new_joint,
    scale_mjcf_model,
    sort_elements,
    string_to_array,
)

# Dict mapping geom type string keywords to group number
GEOMTYPE2GROUP = {
    "collision": {0},  # If we want to use a geom for physics, but NOT visualize
    "visual": {1},  # If we want to use a geom for visualization, but NOT physics
    "all": {0, 1},  # If we want to use a geom for BOTH physics + visualization
}

GEOM_GROUPS = GEOMTYPE2GROUP.keys()


class MujocoObject(MujocoModel):
    """
    Base class for all objects.

    We use Mujoco Objects to implement all objects that:

        1) may appear for multiple times in a task
        2) can be swapped between different tasks

    Typical methods return copy so the caller can all joints/attributes as wanted

    Args:
        obj_type (str): Geom elements to generate / extract for this object. Must be one of:

            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy

    """

    def __init__(self, obj_type="all", duplicate_collision_geoms=True, scale=None):
        super().__init__()
        self.asset = ET.Element("asset")
        assert obj_type in GEOM_GROUPS, "object type must be one in {}, got: {} instead.".format(GEOM_GROUPS, obj_type)
        self.obj_type = obj_type
        self.duplicate_collision_geoms = duplicate_collision_geoms
        self._scale = scale
        # Attributes that should be filled in within the subclass
        self._name = None
        self._obj = None

        # Attributes that are auto-filled by _get_object_properties call
        self._root_body = None
        self._bodies = None
        self._joints = None
        self._actuators = None
        self._sites = None
        self._contact_geoms = None
        self._visual_geoms = None

        if self._scale is not None:
            self.set_scale(self._scale)

    def set_scale(self, scale, obj=None):
        """
        Scales each geom, mesh, site, and body.
        Called during initialization but can also be used externally

        Args:
            scale (float or list of floats): Scale factor (1 or 3 dims)
            obj (ET.Element) Root object to apply. Defaults to root object of model
        """
        if obj is None:
            obj = self._obj

        self._scale = scale

        # Use the centralized scaling utility function
        scale_mjcf_model(
            obj=obj,
            asset_root=self.asset,
            worldbody=None,  # because we don't have a worldbody in MujocoObject
            scale=scale,
            get_elements_func=get_elements,
            scale_slide_joints=False,  # MujocoObject doesn't handle slide joints
        )

    def merge_assets(self, other):
        """
        Merges @other's assets in a custom logic.

        Args:
            other (MujocoXML or MujocoObject): other xml file whose assets will be merged into this one
        """
        for asset in other.asset:
            if (
                find_elements(root=self.asset, tags=asset.tag, attribs={"name": asset.get("name")}, return_first=True)
                is None
            ):
                self.asset.append(asset)

    def get_obj(self):
        """
        Returns the generated / extracted object, in XML ElementTree form.

        Returns:
            ET.Element: Object in XML form.
        """
        assert self._obj is not None, "Object XML tree has not been generated yet!"
        return self._obj

    def exclude_from_prefixing(self, inp):
        """
        A function that should take in either an ET.Element or its attribute (str) and return either True or False,
        determining whether the corresponding name / str to @inp should have naming_prefix added to it.
        Must be defined by subclass.

        Args:
            inp (ET.Element or str): Element or its attribute to check for prefixing.

        Returns:
            bool: True if we should exclude the associated name(s) with @inp from being prefixed with naming_prefix
        """
        raise NotImplementedError

    def _get_object_subtree(self):

        """
        Returns a ET.Element
        It is a <body/> subtree that defines all collision and / or visualization related fields
        of this object.
        Return should be a copy.
        Must be defined by subclass.

        Returns:
            ET.Element: body
        """
        raise NotImplementedError

    def _get_object_properties(self):
        """
        Helper function to extract relevant object properties (bodies, joints, contact/visual geoms, etc...) from this
        object's XML tree. Assumes the self._obj attribute has already been filled.
        """
        # Parse element tree to get all relevant bodies, joints, actuators, and geom groups
        _elements = sort_elements(root=self.get_obj())
        assert (
            len(_elements["root_body"]) == 1
        ), "Invalid number of root bodies found for robot model. Expected 1," "got {}".format(
            len(_elements["root_body"])
        )
        _elements["root_body"] = _elements["root_body"][0]
        _elements["bodies"] = (
            [_elements["root_body"]] + _elements["bodies"] if "bodies" in _elements else [_elements["root_body"]]
        )
        self._root_body = _elements["root_body"].get("name")
        self._bodies = [e.get("name") for e in _elements.get("bodies", [])]
        self._joints = [e.get("name") for e in _elements.get("joints", [])]
        self._actuators = [e.get("name") for e in _elements.get("actuators", [])]
        self._sites = [e.get("name") for e in _elements.get("sites", [])]
        self._sensors = [e.get("name") for e in _elements.get("sensors", [])]
        self._contact_geoms = [e.get("name") for e in _elements.get("contact_geoms", [])]
        self._visual_geoms = [e.get("name") for e in _elements.get("visual_geoms", [])]

        # Add default materials if we're using domain randomization
        if macros.USING_INSTANCE_RANDOMIZATION:
            tex_element, mat_element, _, used = add_material(root=self.get_obj(), naming_prefix=self.naming_prefix)
            # Only add the material / texture if they were actually used
            if used:
                self.asset.append(tex_element)
                self.asset.append(mat_element)

        # Add prefix to all elements
        add_prefix(root=self.get_obj(), prefix=self.naming_prefix, exclude=self.exclude_from_prefixing)

    @property
    def name(self):
        return self._name

    @property
    def naming_prefix(self):
        return "{}_".format(self.name)

    @property
    def root_body(self):
        return self.correct_naming(self._root_body)

    @property
    def bodies(self):
        return self.correct_naming(self._bodies)

    @property
    def joints(self):
        return self.correct_naming(self._joints)

    @property
    def actuators(self):
        return self.correct_naming(self._actuators)

    @property
    def sites(self):
        return self.correct_naming(self._sites)

    @property
    def sensors(self):
        return self.correct_naming(self._sensors)

    @property
    def contact_geoms(self):
        return self.correct_naming(self._contact_geoms)

    @property
    def visual_geoms(self):
        return self.correct_naming(self._visual_geoms)

    @property
    def important_geoms(self):
        """
        Returns:
             dict: (Default is no important geoms; i.e.: empty dict)
        """
        return {}

    @property
    def important_sites(self):
        """
        Returns:
            dict:

                :`obj`: Object default site
        """
        return {"obj": self.naming_prefix + "default_site"}

    @property
    def important_sensors(self):
        """
        Returns:
            dict: (Default is no sensors; i.e.: empty dict)
        """
        return {}

    @property
    def bottom_offset(self):
        """
        Returns vector from model root body to model bottom.
        Useful for, e.g. placing models on a surface.
        Must be defined by subclass.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        raise NotImplementedError

    @property
    def top_offset(self):
        """
        Returns vector from model root body to model top.
        Useful for, e.g. placing models on a surface.
        Must be defined by subclass.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        raise NotImplementedError

    @property
    def horizontal_radius(self):
        """
        Returns maximum distance from model root body to any radial point of the model.

        Helps us put models programmatically without them flying away due to a huge initial contact force.
        Must be defined by subclass.

        Returns:
            float: radius
        """
        raise NotImplementedError

    @staticmethod
    def get_site_attrib_template():
        """
        Returns attribs of spherical site used to mark body origin

        Returns:
            dict: Dictionary of default site attributes
        """
        return {
            "pos": "0 0 0",
            "size": "0.002 0.002 0.002",
            "rgba": "1 0 0 1",
            "type": "sphere",
            "group": "0",
        }

    @staticmethod
    def get_joint_attrib_template():
        """
        Returns attribs of free joint

        Returns:
            dict: Dictionary of default joint attributes
        """
        return {
            "type": "free",
        }

    def get_bounding_box_half_size(self):
        raise NotImplementedError

    def get_bounding_box_size(self):
        """
        Returns numpy array with dimensions of a bounding box around this object.
        """
        return 2.0 * self.get_bounding_box_half_size()


class MujocoXMLObject(MujocoObject, MujocoXML):
    """
    MujocoObjects that are loaded from xml files (by default, inherit all properties (e.g.: name)
    from MujocoObject class first!)

    Args:
        fname (str): XML File path

        name (str): Name of this MujocoXMLObject

        joints (None or str or list of dict): each dictionary corresponds to a joint that will be created for this
            object. The dictionary should specify the joint attributes (type, pos, etc.) according to the MuJoCo xml
            specification. If "default", a single free-joint will be automatically generated. If None, no joints will
            be created.

        obj_type (str): Geom elements to generate / extract for this object. Must be one of:

            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy

        scale (float or list of floats): 3D scale factor
    """

    def __init__(self, fname, name, joints="default", obj_type="all", duplicate_collision_geoms=True, scale=None):
        MujocoXML.__init__(self, fname)
        # Set obj type and duplicate args
        assert obj_type in GEOM_GROUPS, "object type must be one in {}, got: {} instead.".format(GEOM_GROUPS, obj_type)
        self.obj_type = obj_type
        self.duplicate_collision_geoms = duplicate_collision_geoms

        # Set name
        self._name = name

        # set scale
        self._scale = scale

        # joints for this object
        if joints == "default":
            self.joint_specs = [self.get_joint_attrib_template()]  # default free joint
        elif joints is None:
            self.joint_specs = []
        else:
            self.joint_specs = joints

        # Make sure all joints have names!
        for i, joint_spec in enumerate(self.joint_specs):
            if "name" not in joint_spec:
                joint_spec["name"] = "joint{}".format(i)

        # Lastly, parse XML tree appropriately
        self._obj = self._get_object_subtree()

        # scale
        if self._scale is not None:
            self.set_scale(self._scale)

        # Extract the appropriate private attributes for this
        self._get_object_properties()

    def _get_object_subtree(self):
        # Parse object
        # this line used to be wrapped in deepcopy.
        # removed this deepcopy line, as it creates discrepancies between obj and self.worldbody!
        obj = self.worldbody.find("./body/body[@name='object']")
        # Rename this top level object body (will have self.naming_prefix added later)
        obj.attrib["name"] = "main"
        # Get all geom_pairs in this tree
        geom_pairs = get_elements(obj, "geom")

        # Define a temp function so we don't duplicate so much code
        obj_type = self.obj_type

        def _should_keep(el):
            return int(el.get("group")) in GEOMTYPE2GROUP[obj_type]

        # Loop through each of these pairs and modify them according to @elements arg
        for i, (parent, element) in enumerate(geom_pairs):
            # Delete non-relevant geoms and rename remaining ones
            if not _should_keep(element):
                parent.remove(element)
            else:
                g_name = element.get("name")
                g_name = g_name if g_name is not None else f"g{i}"
                element.set("name", g_name)
                # Also optionally duplicate collision geoms if requested (and this is a collision geom)
                if self.duplicate_collision_geoms and element.get("group") in {None, "0"}:
                    parent.append(self._duplicate_visual_from_collision(element))
                    # Also manually set the visual appearances to the original collision model
                    element.set("rgba", array_to_string(OBJECT_COLLISION_COLOR))
                    if element.get("material") is not None:
                        del element.attrib["material"]
        # add joint(s)
        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))
        # Lastly, add a site for this object
        template = self.get_site_attrib_template()
        template["rgba"] = "1 0 0 0"
        template["name"] = "default_site"
        obj.append(ET.Element("site", attrib=template))

        return obj

    def exclude_from_prefixing(self, inp):
        """
        By default, don't exclude any from being prefixed
        """
        return False

    def _get_object_properties(self):
        """
        Extends the base class method to also add prefixes to all bodies in this object
        """
        super()._get_object_properties()
        add_prefix(root=self.root, prefix=self.naming_prefix, exclude=self.exclude_from_prefixing)

    @staticmethod
    def _duplicate_visual_from_collision(element):
        """
        Helper function to duplicate a geom element to be a visual element. Namely, this corresponds to the
        following attribute requirements: group=1, conaffinity/contype=0, no mass, name appended with "_visual"

        Args:
            element (ET.Element): element to duplicate as a visual geom

        Returns:
            element (ET.Element): duplicated element
        """
        # Copy element
        vis_element = deepcopy(element)
        # Modify for visual-specific attributes (group=1, conaffinity/contype=0, no mass, update name)
        vis_element.set("group", "1")
        vis_element.set("conaffinity", "0")
        vis_element.set("contype", "0")
        vis_element.set("mass", "1e-8")
        vis_element.set("name", vis_element.get("name") + "_visual")
        return vis_element

    def set_pos(self, pos):
        """
        Set position of object position is defined as center of bounding box

        Args:
            pos (list of floats): 3D position to set object (should be 3 dims)
        """
        self._obj.set("pos", array_to_string(pos))

    def set_euler(self, euler):
        """
        Set Euler value object position

        Args:
            euler (list of floats): 3D Euler values (should be 3 dims)
        """
        self._obj.set("euler", array_to_string(euler))

    @property
    def rot(self):
        rot = string_to_array(self._obj.get("euler", "0.0 0.0 0.0"))
        return rot[2]

    def set_scale(self, scale, obj=None):
        """
        Scales each geom, mesh, site, and body.
        Called during initialization but can also be used externally

        Args:
            scale (float or list of floats): Scale factor (1 or 3 dims)
            obj (ET.Element) Root object to apply. Defaults to root object of model
        """
        if obj is None:
            obj = self._obj

        self._scale = scale

        # Use the centralized scaling utility function
        scale_mjcf_model(
            obj=obj,
            asset_root=self.asset,
            worldbody=self.worldbody,
            scale=scale,
            get_elements_func=get_elements,
            scale_slide_joints=False,  # MujocoXMLObject doesn't handle slide joints
        )

    @property
    def bottom_offset(self):
        bottom_site = self.worldbody.find("./body/site[@name='{}bottom_site']".format(self.naming_prefix))
        return string_to_array(bottom_site.get("pos"))

    @property
    def top_offset(self):
        top_site = self.worldbody.find("./body/site[@name='{}top_site']".format(self.naming_prefix))
        return string_to_array(top_site.get("pos"))

    @property
    def horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        return string_to_array(horizontal_radius_site.get("pos"))[0]

    def get_bounding_box_half_size(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        return string_to_array(horizontal_radius_site.get("pos")) - self.bottom_offset

    def _get_elements_by_name(self, geom_names, body_names=None, joint_names=None):
        """
        seaches for returns all geoms, bodies, and joints used for cabinet
        called by _get_cab_components, as implemented in subclasses

        for geoms, include both collision and visual geoms
        """

        # names of every geom
        geoms = {geom_name: list() for geom_name in geom_names}
        for geom_name in geoms.keys():
            for postfix in ["", "_visual"]:
                g = find_elements(
                    root=self._obj,
                    tags="geom",
                    attribs={"name": self.name + "_" + geom_name + postfix},
                    return_first=True,
                )
                geoms[geom_name].append(g)

        # get bodies
        bodies = dict()
        if body_names is not None:
            for body_name in body_names:
                bodies[body_name] = find_elements(
                    root=self._obj, tags="body", attribs={"name": self.name + "_" + body_name}, return_first=True
                )

        # get joints
        joints = dict()
        if joint_names is not None:
            for joint_name in joint_names:
                joints[joint_name] = find_elements(
                    root=self._obj, tags="joint", attribs={"name": self.name + "_" + joint_name}, return_first=True
                )
        return geoms, bodies, joints


class MujocoGeneratedObject(MujocoObject):
    """
    Base class for all procedurally generated objects.

    Args:
        obj_type (str): Geom elements to generate / extract for this object. Must be one of:

            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy
    """

    def __init__(self, obj_type="all", duplicate_collision_geoms=True):
        super().__init__(obj_type=obj_type, duplicate_collision_geoms=duplicate_collision_geoms)

        # Store common material names so we don't add prefixes to them
        self.shared_materials = set()
        self.shared_textures = set()

    def sanity_check(self):
        """
        Checks if data provided makes sense.
        Called in __init__()
        For subclasses to inherit from
        """
        pass

    @staticmethod
    def get_collision_attrib_template():
        """
        Generates template with collision attributes for a given geom

        Returns:
            dict: Initial template with `'pos'` and `'group'` already specified
        """
        return {"group": "0", "rgba": array_to_string(OBJECT_COLLISION_COLOR)}

    @staticmethod
    def get_visual_attrib_template():
        """
        Generates template with visual attributes for a given geom

        Returns:
            dict: Initial template with `'conaffinity'`, `'contype'`, and `'group'` already specified
        """
        return {"conaffinity": "0", "contype": "0", "mass": "1e-8", "group": "1"}

    def append_material(self, material):
        """
        Adds a new texture / material combination to the assets subtree of this XML
        Input is expected to be a CustomMaterial object

        See http://www.mujoco.org/book/XMLreference.html#asset for specific details on attributes expected for
        Mujoco texture / material tags, respectively

        Note that the "file" attribute for the "texture" tag should be specified relative to the textures directory
        located in robosuite/models/assets/textures/

        Args:
            material (CustomMaterial): Material to add to this object
        """
        # First check if asset attribute exists; if not, define the asset attribute
        if not hasattr(self, "asset"):
            self.asset = ET.Element("asset")
        # If the material name is not in shared materials, add this to our assets
        if material.name not in self.shared_materials:
            self.asset.append(ET.Element("texture", attrib=material.tex_attrib))
            self.asset.append(ET.Element("material", attrib=material.mat_attrib))
        # Add this material name to shared materials if it should be shared
        if material.shared:
            self.shared_materials.add(material.name)
            self.shared_textures.add(material.tex_attrib["name"])
        # Update prefix for assets
        add_prefix(root=self.asset, prefix=self.naming_prefix, exclude=self.exclude_from_prefixing)

    def exclude_from_prefixing(self, inp):
        """
        Exclude all shared materials and their associated names from being prefixed.

        Args:
            inp (ET.Element or str): Element or its attribute to check for prefixing.

        Returns:
            bool: True if we should exclude the associated name(s) with @inp from being prefixed with naming_prefix
        """
        # Automatically return False if this is not of type "str"
        if type(inp) is not str:
            return False
        # Only return True if the string matches the name of a common material
        return True if inp in self.shared_materials or inp in self.shared_textures else False

    # Methods that still need to be defined by subclass
    def _get_object_subtree(self):
        raise NotImplementedError

    def bottom_offset(self):
        raise NotImplementedError

    def top_offset(self):
        raise NotImplementedError

    def horizontal_radius(self):
        raise NotImplementedError

    def get_bounding_box_half_size(self):
        return np.array([self.horizontal_radius, self.horizontal_radius, 0.0]) - self.bottom_offset
