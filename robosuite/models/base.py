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
    """

    def __init__(self, fname):
        """
        Loads a mujoco xml from file.

        Args:
            fname (str): path to the MJCF xml file.
        """
        self.file = fname
        self.folder = os.path.dirname(fname)
        self.tree = ET.parse(fname)
        self.root = self.tree.getroot()
        self.name = self.root.get("model")
        self.worldbody = self.create_default_element("worldbody")
        self.actuator = self.create_default_element("actuator")
        self.asset = self.create_default_element("asset")
        self.equality = self.create_default_element("equality")
        self.sensor = self.create_default_element("sensor")
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
        """

        found = self.root.find(name)
        if found is not None:
            return found
        ele = ET.Element(name)
        self.root.append(ele)
        return ele

    def merge(self, other, merge_body=True):
        """
        Default merge method.

        Args:
            other: another MujocoXML instance
                raises XML error if @other is not a MujocoXML instance.
                merges <worldbody/>, <actuator/> and <asset/> of @other into @self
            merge_body: True if merging child bodies of @other. Defaults to True.
        """
        if not isinstance(other, MujocoXML):
            raise XMLError("{} is not a MujocoXML instance.".format(type(other)))
        if merge_body:
            for body in other.worldbody:
                self.worldbody.append(body)
        self.merge_asset(other)
        for one_actuator in other.actuator:
            self.actuator.append(one_actuator)
        for one_equality in other.equality:
            self.equality.append(one_equality)
        for one_sensor in other.sensor:
            self.sensor.append(one_sensor)
        for one_contact in other.contact:
            self.contact.append(one_contact)
        for one_default in other.default:
            self.default.append(one_default)
        # self.config.append(other.config)

    def get_model(self, mode="mujoco_py"):
        """
        Returns a MjModel instance from the current xml tree.
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
        Returns a string of the MJCF XML file.
        """
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            return string.getvalue()

    def save_model(self, fname, pretty=False):
        """
        Saves the xml to file.

        Args:
            fname: output file location
            pretty: attempts!! to pretty print the output
        """
        with open(fname, "w") as f:
            xml_str = ET.tostring(self.root, encoding="unicode")
            if pretty:
                # TODO: get a better pretty print library
                parsed_xml = xml.dom.minidom.parseString(xml_str)
                xml_str = parsed_xml.toprettyxml(newl="")
            f.write(xml_str)

    def merge_asset(self, other):
        """
        Useful for merging other files in a custom logic.
        """
        for asset in other.asset:
            asset_name = asset.get("name")
            asset_type = asset.tag
            # Avoids duplication
            pattern = "./{}[@name='{}']".format(asset_type, asset_name)
            if self.asset.find(pattern) is None:
                self.asset.append(asset)
