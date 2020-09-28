import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import io
import numpy as np

from robosuite.utils import XMLError


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
        self.default = self.create_default_element("default")
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

    def merge(self, others, merge_body=True):
        """
        Default merge method.

        Args:
            others (MujocoXML or list of MujocoXML): other xmls to merge into this one
                raises XML error if @others is not a MujocoXML instance.
                merges <worldbody/>, <actuator/> and <asset/> of @others into @self
            merge_body (bool): True if merging child bodies of @others

        Raises:
            XMLError: [Invalid XML instance]
        """
        if type(others) is not list:
            others = [others]
        for idx, other in enumerate(others):
            if not isinstance(other, MujocoXML):
                raise XMLError("{} is not a MujocoXML instance.".format(type(other)))
            if merge_body:
                for body in other.worldbody:
                    self.worldbody.append(body)
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
            for one_default in other.default:
                self.default.append(one_default)

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
            other (MujocoXML): other xml file whose assets will be merged into this one
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
            tags (list or tuple): Tags to be searched in the XML. All elements with specified tags will have "prefix"
                prepended to it
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
            tags (list or tuple): Tags to be searched in the XML. All elements with specified tags will have "prefix"
                prepended to it
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
