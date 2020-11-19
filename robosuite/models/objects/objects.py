import copy
from copy import deepcopy
import xml.etree.ElementTree as ET

from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import string_to_array, array_to_string, CustomMaterial, OBJECT_COLLISION_COLOR


# Dict mapping geom type string keywords to group number
GEOMTYPE2GROUP = {
    "collision": {0},                 # If we want to use a geom for physics, but NOT visualize
    "visual": {1},                    # If we want to use a geom for visualization, but NOT physics
    "all": {0, 1},                    # If we want to use a geom for BOTH physics + visualization
}

GEOM_GROUPS = GEOMTYPE2GROUP.keys()


class MujocoObject:
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

    def __init__(self, obj_type="all", duplicate_collision_geoms=True):
        self.asset = ET.Element("asset")
        assert obj_type in GEOM_GROUPS, "object type must be one in {}, got: {} instead.".format(GEOM_GROUPS, obj_type)
        self.obj_type = obj_type
        self.duplicate_collision_geoms = duplicate_collision_geoms

    def get_bottom_offset(self):
        """
        Returns vector from object center to object bottom.
        Helps us put objects on a surface.
        Must be defined by subclass.

        Returns:
            np.array: (dx, dy, dz) vector, eg. np.array([0, 0, -2])
        """
        raise NotImplementedError

    def get_top_offset(self):
        """
        Returns vector from object center to object top.
        Helps us put other objects on this object.
        Must be defined by subclass.

        Returns:
            np.array: (dx, dy, dz) vector, eg. np.array([0, 0, 2])
        """
        raise NotImplementedError

    def get_horizontal_radius(self):
        """
        Returns scalar
        If object a,b has horizontal distance d
        a.get_horizontal_radius() + b.get_horizontal_radius() < d
        should mean that a, b has no contact

        Helps us put objects programmatically without them flying away due to a huge initial contact force.
        Must be defined by subclass.

        Returns:
            float: radius
        """
        raise NotImplementedError

    def get_object_subtree(self, site=False):

        """
        Returns a ET.Element
        It is a <body/> subtree that defines all collision and / or visualization related fields
        of this object.
        Return should be a copy.
        Must be defined by subclass.

        Args:
            site (bool): If set, add a site (with name @name when applicable) to the returned body

        Returns:
            ET.Element: body
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


class MujocoXMLObject(MujocoXML, MujocoObject):
    """
    MujocoObjects that are loaded from xml files

    Args:
        fname (str): XML File path

        name (str): Name of this MujocoXMLObject

        joints (list of dict): each dictionary corresponds to a joint that will be created for this object. The
            dictionary should specify the joint attributes (type, pos, etc.) according to the MuJoCo xml specification.

        obj_type (str): Geom elements to generate / extract for this object. Must be one of:

            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy
    """

    def __init__(self, fname, name, joints=None, obj_type="all",  duplicate_collision_geoms=True):
        MujocoXML.__init__(self, fname)
        # Set obj type and duplicate args
        assert obj_type in GEOM_GROUPS, "object type must be one in {}, got: {} instead.".format(GEOM_GROUPS, obj_type)
        self.obj_type = obj_type
        self.duplicate_collision_geoms = duplicate_collision_geoms

        # Set name
        self.name = name

        # joints for this object
        if joints is None:
            self.joints = [{'type': 'free'}]  # default free joint
        else:
            self.joints = joints

    def get_bottom_offset(self):
        bottom_site = self.worldbody.find("./body/site[@name='bottom_site']")
        return string_to_array(bottom_site.get("pos"))

    def get_top_offset(self):
        top_site = self.worldbody.find("./body/site[@name='top_site']")
        return string_to_array(top_site.get("pos"))

    def get_horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='horizontal_radius_site']"
        )
        return string_to_array(horizontal_radius_site.get("pos"))[0]

    def get_object_subtree(self, site=False):
        # Parse object
        obj = copy.deepcopy(self.worldbody.find("./body/body[@name='object']"))
        # Rename this top level object body
        obj.attrib.pop("name")
        obj.attrib["name"] = self.name
        # Get all geom_pairs in this tree
        geom_pairs = self._get_geoms(obj)

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
                modded_name = f"{self.name}_{g_name}" if g_name is not None else f"{self.name}_{i}"
                element.set("name", modded_name)
                # Also optionally duplicate collision geoms if requested (and this is a collision geom)
                if self.duplicate_collision_geoms and int(element.get("group")) in {None, 0}:
                    parent.append(self._duplicate_visual_from_collision(element))
                    # Also manually set the visual appearances to the original collision model
                    element.set("rgba", array_to_string(OBJECT_COLLISION_COLOR))
                    if element.get("material") is not None:
                        del element.attrib["material"]
        # Lastly, add the site if requested
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            template["name"] = self.name
            obj.append(ET.Element("site", attrib=template))

        return obj

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

    def _get_geoms(self, root, parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a geom element

        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call

        Returns:
            list: array of (parent, child) tuples where the child element is a geom type
        """
        # Initialize return array
        geom_pairs = []
        # If the parent exists and this is a geom element, we add this current (parent, element) combo to the output
        if parent is not None and root.tag == "geom":
            geom_pairs.append((parent, root))
        # Loop through all children elements recursively and add to pairs
        for child in root:
            geom_pairs += self._get_geoms(child, parent=root)
        # Return all found pairs
        return geom_pairs


class MujocoGeneratedObject(MujocoObject):
    """
    Base class for all programmatically generated mujoco object
    i.e., every MujocoObject that does not have an corresponding xml file

    Args:
        name (str): (unique) name to identify this generated object

        size (n-tuple of float): relevant size parameters for the object, should be of size 1 - 3

        rgba (4-tuple of float): Color

        density (float): Density

        friction (3-tuple of float): (sliding friction, torsional friction, and rolling friction).
            A single float can also be specified, in order to set the sliding friction (the other values) will
            be set to the MuJoCo default. See http://www.mujoco.org/book/modeling.html#geom for details.

        solref (2-tuple of float): MuJoCo solver parameters that handle contact.
            See http://www.mujoco.org/book/XMLreference.html for more details.

        solimp (3-tuple of float): MuJoCo solver parameters that handle contact.
            See http://www.mujoco.org/book/XMLreference.html for more details.

        material (CustomMaterial or `'default'` or None): if "default", add a template material and texture for this
            object that is used to color the geom(s).
            Otherwise, input is expected to be a CustomMaterial object

            See http://www.mujoco.org/book/XMLreference.html#asset for specific details on attributes expected for
            Mujoco texture / material tags, respectively

            Note that specifying a custom texture in this way automatically overrides any rgba values set

        joints (None or str or list of dict): Joints for this object. If None, no joint will be created. If "default",
            a single joint will be crated. Else, should be a list of dict, where each dictionary corresponds to a joint
            that will be created for this object. The dictionary should specify the joint attributes (type, pos, etc.)
            according to the MuJoCo xml specification.

        obj_type (str): Geom elements to generate / extract for this object. Must be one of:

            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy
    """

    def __init__(
        self,
        name,
        size=None,
        rgba=None,
        density=None,
        friction=None,
        solref=None,
        solimp=None,
        material=None,
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=True,
    ):
        super().__init__(obj_type=obj_type, duplicate_collision_geoms=duplicate_collision_geoms)

        self.name = name

        if size is None:
            size = [0.05, 0.05, 0.05]
        self.size = list(size)

        if rgba is None:
            rgba = [1, 0, 0, 1]
        assert len(rgba) == 4, "rgba must be a length 4 array"
        self.rgba = list(rgba)

        if density is None:
            density = 1000  # water
        self.density = density

        if friction is None:
            friction = [1, 0.005, 0.0001]  # MuJoCo default
        elif isinstance(friction, float) or isinstance(friction, int):
            friction = [friction, 0.005, 0.0001]
        assert len(friction) == 3, "friction must be a length 3 array or a single number"
        self.friction = list(friction)

        if solref is None:
            self.solref = [0.02, 1.]  # MuJoCo default
        else:
            self.solref = solref

        if solimp is None:
            self.solimp = [0.9, 0.95, 0.001]  # MuJoCo default
        else:
            self.solimp = solimp

        self.material = material
        if material == "default":
            # add in default texture and material for this object (for domain randomization)
            default_tex = CustomMaterial(
                texture=self.rgba,
                tex_name="{}_tex".format(self.name),
                mat_name="{}_mat".format(self.name),
            )
            self.append_material(default_tex)
        elif material is not None:
            # add in custom texture and material
            self.append_material(material)

        # joints for this object
        if joints == "default":
            self.joints = [{'type': 'free'}]  # default free joint
        elif joints is None:
            self.joints = []
        else:
            self.joints = joints

        self.sanity_check()

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
        # Add texture and material inputs to asset
        self.asset.append(ET.Element("texture", attrib=material.tex_attrib))
        self.asset.append(ET.Element("material", attrib=material.mat_attrib))

    def get_object_subtree(self, site=False):
        raise NotImplementedError

    def _get_object_subtree(self, site=False, ob_type="box"):
        # Create element tree
        obj = ET.Element("body")
        obj.set("name", self.name)

        # Get base element attributes
        element_attr = {
            "name": self.name,
            "type": ob_type,
            "size": array_to_string(self.size)
        }

        # Add collision geom if necessary
        if self.obj_type in {"collision", "all"}:
            col_element_attr = deepcopy(element_attr)
            col_element_attr.update(self.get_collision_attrib_template())
            # col_element_attr["name"] += "_col"
            col_element_attr["density"] = str(self.density)
            col_element_attr["friction"] = array_to_string(self.friction)
            col_element_attr["solref"] = array_to_string(self.solref)
            col_element_attr["solimp"] = array_to_string(self.solimp)
            obj.append(ET.Element("geom", attrib=col_element_attr))
        # Add visual geom if necessary
        if self.obj_type in {"visual", "all"}:
            vis_element_attr = deepcopy(element_attr)
            vis_element_attr.update(self.get_visual_attrib_template())
            vis_element_attr["name"] += "_vis"
            if self.material == "default":
                vis_element_attr["rgba"] = "0.5 0.5 0.5 1"  # mujoco default
                vis_element_attr["material"] = "{}_mat".format(self.name)
            elif self.material is not None:
                vis_element_attr["material"] = self.material.mat_attrib["name"]
            else:
                vis_element_attr["rgba"] = array_to_string(self.rgba)
            obj.append(ET.Element("geom", attrib=vis_element_attr))
        # Lastly, add site as well if requested
        if site:
            # add a site as well
            site_element_attr = self.get_site_attrib_template()
            site_element_attr["name"] = self.name
            obj.append(ET.Element("site", attrib=site_element_attr))
        return obj
