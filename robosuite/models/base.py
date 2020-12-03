import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import io

from robosuite.utils import XMLError
from robosuite.utils.mjcf_utils import array_to_string, find_elements


class MujocoXML(object):
    """
    Base class of Mujoco xml file
    Wraps around ElementTree and provides additional functionality for merging different models.
    Specially, we keep track of <worldbody/>, <actuator/> and <asset/>

    When initialized, loads a mujoco xml from file.

    Args:
        fname (str): path to the MJCF xml file.
    """

    def __init__(self, fname):
        self.file = fname
        self.folder = os.path.dirname(fname)
        self.tree = ET.parse(fname)
        self.root = self.tree.getroot()
        self.name = self.root.get("model")
        self.worldbody = self.create_default_element("worldbody")
        self.actuator = self.create_default_element("actuator")
        self.sensor = self.create_default_element("sensor")
        self.asset = self.create_default_element("asset")
        self.tendon = self.create_default_element("tendon")
        self.equality = self.create_default_element("equality")
        self.contact = self.create_default_element("contact")

        # Parse any default classes and replace them inline
        default = self.create_default_element("default")
        default_classes = self._get_default_classes(default)
        self._replace_defaults_inline(default_dic=default_classes)

        self.resolve_asset_dependency()

    def resolve_asset_dependency(self):
        """
        Converts every file dependency into absolute path so when we merge we don't break things.
        """

        for node in self.asset.findall("./*[@file]"):
            file = node.get("file")
            abs_path = os.path.abspath(self.folder)
            abs_path = os.path.join(abs_path, file)
            node.set("file", abs_path)

    def create_default_element(self, name):
        """
        Creates a <@name/> tag under root if there is none.

        Args:
            name (str): Name to generate default element

        Returns:
            ET.Element: Node that was created
        """

        found = self.root.find(name)
        if found is not None:
            return found
        ele = ET.Element(name)
        self.root.append(ele)
        return ele

    def merge(self, others, merge_body="default"):
        """
        Default merge method.

        Args:
            others (MujocoXML or list of MujocoXML): other xmls to merge into this one
                raises XML error if @others is not a MujocoXML instance.
                merges <worldbody/>, <actuator/> and <asset/> of @others into @self
            merge_body (None or str): If set, will merge child bodies of @others. Default is "default", which
                corresponds to the root worldbody for this XML. Otherwise, should be an existing body name
                that exists in this XML.

        Raises:
            XMLError: [Invalid XML instance]
        """
        if type(others) is not list:
            others = [others]
        for idx, other in enumerate(others):
            if not isinstance(other, MujocoXML):
                raise XMLError("{} is not a MujocoXML instance.".format(type(other)))
            if merge_body is not None:
                root = self.worldbody if merge_body == "default" else \
                    find_elements(root=self.worldbody, tags="body", attribs={"name": merge_body}, return_first=True)
                for body in other.worldbody:
                    root.append(body)
            self.merge_asset(other)
            for one_actuator in other.actuator:
                self.actuator.append(one_actuator)
            for one_sensor in other.sensor:
                self.sensor.append(one_sensor)
            for one_tendon in other.tendon:
                self.tendon.append(one_tendon)
            for one_equality in other.equality:
                self.equality.append(one_equality)
            for one_contact in other.contact:
                self.contact.append(one_contact)

    def get_model(self, mode="mujoco_py"):
        """
        Generates a MjModel instance from the current xml tree.

        Args:
            mode (str): Mode with which to interpret xml tree

        Returns:
            MjModel: generated model from xml

        Raises:
            ValueError: [Invalid mode]
        """

        available_modes = ["mujoco_py"]
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            if mode == "mujoco_py":
                from mujoco_py import load_model_from_xml

                model = load_model_from_xml(string.getvalue())
                return model
            raise ValueError(
                "Unkown model mode: {}. Available options are: {}".format(
                    mode, ",".join(available_modes)
                )
            )

    def get_xml(self):
        """
        Reads a string of the MJCF XML file.

        Returns:
            str: XML tree read in from file
        """
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            return string.getvalue()

    def save_model(self, fname, pretty=False):
        """
        Saves the xml to file.

        Args:
            fname (str): output file location
            pretty (bool): If True, (attempts!! to) pretty print the output
        """
        with open(fname, "w") as f:
            xml_str = ET.tostring(self.root, encoding="unicode")
            if pretty:
                parsed_xml = xml.dom.minidom.parseString(xml_str)
                xml_str = parsed_xml.toprettyxml(newl="")
            f.write(xml_str)

    def merge_asset(self, other):
        """
        Merges other files in a custom logic.

        Args:
            other (MujocoXML or MujocoObject): other xml file whose assets will be merged into this one
        """
        for asset in other.asset:
            asset_name = asset.get("name")
            asset_type = asset.tag
            # Avoids duplication
            pattern = "./{}[@name='{}']".format(asset_type, asset_name)
            if self.asset.find(pattern) is None:
                self.asset.append(asset)

    def get_element_names(self, root, element_type):
        """
        Searches recursively through the @root and returns a list of names of the specified @element_type

        Args:
            root (ET.Element): Root of the xml element tree to start recursively searching through
                (e.g.: `self.worldbody`)
            element_type (str): Name of element to return names of. (e.g.: "site", "geom", etc.)

        Returns:
            list: names that correspond to the specified @element_type
        """
        names = []
        for child in root:
            if child.tag == element_type:
                names.append(child.get("name"))
            names += self.get_element_names(child, element_type)
        return names

    def add_prefix(self,
                   prefix,
                   tags=("body", "joint", "sensor", "site", "geom", "camera", "actuator", "tendon", "asset", "mesh", "texture", "material")):
        """
        Utility method to add prefix to all body names to prevent name clashes

        Args:
            prefix (str): Prefix to be appended to all requested elements in this XML
            tags (list or tuple or set): Tags to be searched in the XML. All elements with specified tags will have
                "prefix" prepended to it
        """
        # Define tags as a set
        tags = set(tags)

        # Define equalities set to pass at the end
        equalities = set(tags)

        # Add joints to equalities if necessary
        if "joint" in tags:
            equalities = equalities.union(["joint1", "joint2"])

        # Handle actuator elements
        if "actuator" in tags:
            tags.discard("actuator")
            for actuator in self.actuator:
                self._add_prefix_recursively(actuator, tags, prefix)

        # Handle sensor elements
        if "sensor" in tags:
            tags.discard("sensor")
            for sensor in self.sensor:
                self._add_prefix_recursively(sensor, tags, prefix)

        # Handle tendon elements
        if "tendon" in tags:
            tags.discard("tendon")
            for tendon in self.tendon:
                self._add_prefix_recursively(tendon, tags.union(["fixed"]), prefix)
            # Also take care of any tendons in equality constraints
            equalities = equalities.union(["tendon1", "tendon2"])

        # Handle asset elements
        if "asset" in tags:
            tags.discard("asset")
            for asset in self.asset:
                if asset.tag in tags:
                    self._add_prefix_recursively(asset, tags, prefix)

        # Handle contacts and equality names for body elements
        if "body" in tags:
            for contact in self.contact:
                if "body1" in contact.attrib:
                    contact.set("body1", prefix + contact.attrib["body1"])
                if "body2" in contact.attrib:
                    contact.set("body2", prefix + contact.attrib["body2"])
            # Also take care of any bodies in equality constraints
            equalities = equalities.union(["body1", "body2"])

        # Handle all equality elements
        for equality in self.equality:
            self._add_prefix_recursively(equality, equalities, prefix)

        # Handle all remaining bodies in the element tree
        for body in self.worldbody:
            if body.tag in tags:
                self._add_prefix_recursively(body, tags, prefix)

    def _add_prefix_recursively(self, root, tags, prefix):
        """
        Iteratively searches through all children nodes in "root" element to append "prefix" to any named subelements
        with a tag in "tags"

        Args:
            root (ET.Element): Root of the xml element tree to start recursively searching through
                (e.g.: `self.worldbody`)
            tags (list or tuple or set): Tags to be searched in the XML. All elements with specified tags will have
                "prefix" prepended to it
            prefix (str): Prefix to be appended to all requested elements in this XML
        """
        # First re-name this element
        if "name" in root.attrib:
            root.set("name", prefix + root.attrib["name"])

        # Then loop through all tags and rename any appropriately
        for tag in tags:
            if tag in root.attrib:
                root.set(tag, prefix + root.attrib[tag])

        # Recursively go through child elements
        for child in root:
            if child.tag in tags:
                self._add_prefix_recursively(child, tags, prefix)

    def recolor_collision_geoms(self, rgba):
        """
        Utility method to recolor all collision geoms (where collision geoms are defined as being part of group 0).

        Args:
            rgba (4-array): (R, G, B, A) values to assign to all geoms with this group.
        """
        for body in self.worldbody:
            self._recolor_collision_geoms_recursively(body, rgba)

    def _recolor_collision_geoms_recursively(self, root, rgba):
        """
        Iteratively searches through all children nodes in "root" element to find all geoms belonging to group 0 and set
        the corresponding rgba value to the specified @rgba argument. Note: also removes any material values for this
        model.

        Args:
            root (ET.Element): Root of the xml element tree to start recursively searching through
            rgba (4-array): (R, G, B, A) values to assign to all geoms with this group.
        """
        for child in root:
            if child.tag == "geom" and child.get("group") in {None, "0"}:
                child.set("rgba", array_to_string(rgba))
                child.attrib.pop("material", None)

            self._recolor_collision_geoms_recursively(child, rgba)

    @staticmethod
    def _get_default_classes(default):
        """
        Utility method to convert all default tags into a nested dictionary of values -- this will be used to replace
        all elements' class tags inline with the appropriate defaults if not specified.

        Args:
            default (ET.Element): Nested default tag XML root.

        Returns:
            dict: Nested dictionary, where each default class name is mapped to its own dict mapping element tag names
                (e.g.: geom, site, etc.) to the set of default attributes for that tag type
        """
        # Create nested dict to return
        default_dic = {}
        # Parse the default tag accordingly
        for cls in default:
            default_dic[cls.get("class")] = {child.tag: child for child in cls}
        return default_dic

    def _replace_defaults_inline(self, default_dic, root=None):
        """
        Utility method to replace all default class attributes recursively in the XML tree starting from @root
        with the corresponding defaults in @default_dic if they are not explicitly specified for ta given element.

        Args:
            root (ET.Element): Root of the xml element tree to start recursively replacing defaults. Only is used by
                recursive calls
            default_dic (dict): Nested dictionary, where each default class name is mapped to its own dict mapping
                element tag names (e.g.: geom, site, etc.) to the set of default attributes for that tag type
        """
        # If root is None, this is the top level call -- replace root with self.root
        if root is None:
            root = self.root
        # Check this current element if it contains any class elements
        cls_name = root.attrib.pop("class", None)
        if cls_name is not None:
            # If the tag for this element is contained in our default dic, we add any defaults that are not
            # explicitly specified in this
            tag_attrs = default_dic[cls_name].get(root.tag, None)
            if tag_attrs is not None:
                for k, v in tag_attrs.items():
                    if root.get(k, None) is None:
                        root.set(k, v)
        # Loop through all child elements
        for child in root:
            self._replace_defaults_inline(default_dic=default_dic, root=child)


class MujocoModel(object):
    """
    Base class for all simulation models used in mujoco.

    Standardizes core API for accessing models' relevant geoms, names, etc.
    """
    def correct_naming(self, names):
        """
        Corrects all strings in @names by adding the naming prefix to it and returns the name-corrected values

        Args:
            names (str, list, or dict): Name(s) to be corrected

        Raises:
            TypeError: [Invalid input type]
        """
        if type(names) is str:
            return self.naming_prefix + names
        elif type(names) is list:
            return [self.naming_prefix + name for name in names]
        elif type(names) is dict:
            names = names.copy()
            for key, val in names.items():
                names[key] = self.correct_naming(val)
            return names
        else:
            # Assumed to be type error
            raise TypeError("Error: type of 'names' must be str, list, or dict!")

    def set_sites_visibility(self, sim, visible):
        """
        Set all site visual states for this model.

        Args:
            sim (MjSim): Current active mujoco simulation instance
            visible (bool): If True, will visualize model sites. Else, will hide the sites.
        """
        # Loop through all visualization geoms and set their alpha values appropriately
        for vis_g in self.sites:
            vis_g_id = sim.model.site_name2id(vis_g)
            if (visible and sim.model.site_rgba[vis_g_id][3] < 0) or \
                    (not visible and sim.model.site_rgba[vis_g_id][3] > 0):
                # We toggle the alpha value
                sim.model.site_rgba[vis_g_id][3] = -sim.model.site_rgba[vis_g_id][3]

    @property
    def naming_prefix(self):
        """
        Generates a standardized prefix to prevent naming collisions

        Returns:
            str: Prefix unique to this model.
        """
        raise NotImplementedError

    @property
    def root_body(self):
        """
        Root body name for this model. This should correspond to the top-level body element in the equivalent mujoco xml
        tree for this object.
        """
        raise NotImplementedError

    @property
    def bodies(self):
        """
        Returns:
            list: Body names for this model
        """
        raise NotImplementedError

    @property
    def joints(self):
        """
        Returns:
            list: Joint names for this model
        """
        raise NotImplementedError

    @property
    def actuators(self):
        """
        Returns:
            list: Actuator names for this model
        """
        raise NotImplementedError

    @property
    def sites(self):
        """
        Returns:
             list: Site names for this model
        """
        raise NotImplementedError

    @property
    def contact_geoms(self):
        """
        List of names corresponding to the geoms used to determine contact with this model.

        Returns:
            list: relevant contact geoms for this model
        """
        raise NotImplementedError

    @property
    def visual_geoms(self):
        """
        List of names corresponding to the geoms used for visual rendering of this model.

        Returns:
            list: relevant visual geoms for this model
        """
        raise NotImplementedError

    @property
    def important_geoms(self):
        """
        Geoms corresponding to important components of this model. String keywords should be mapped to lists of geoms.

        Returns:
            dict of list: Important set of geoms, where each set of geoms are grouped as a list and are
            organized by keyword string entries into a dict
        """
        raise NotImplementedError

    @property
    def important_sites(self):
        """
        Dict of sites corresponding to the important site geoms (e.g.: used to aid visualization during sim).

        Returns:
            dict: Important site geoms, where each specific geom name is mapped from keyword string entries
                in the dict
        """
        raise NotImplementedError

    @property
    def sensors(self):
        """
        Dict of sensors enabled for this model.

        Returns:
            dict: Sensors for this model, where each specific sensor name is mapped from keyword string entries
                in the dict
        """
        raise NotImplementedError
