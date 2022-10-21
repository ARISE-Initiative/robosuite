import io
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET

import robosuite.macros as macros
from robosuite.utils import XMLError
from robosuite.utils.mjcf_utils import (
    _element_filter,
    add_material,
    add_prefix,
    find_elements,
    recolor_collision_geoms,
    sort_elements,
    string_to_array,
)


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

        # Remove original default classes
        self.root.remove(default)

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
                that exists in this XML. None results in no merging of @other's bodies in its worldbody.

        Raises:
            XMLError: [Invalid XML instance]
        """
        if type(others) is not list:
            others = [others]
        for idx, other in enumerate(others):
            if not isinstance(other, MujocoXML):
                raise XMLError("{} is not a MujocoXML instance.".format(type(other)))
            if merge_body is not None:
                root = (
                    self.worldbody
                    if merge_body == "default"
                    else find_elements(
                        root=self.worldbody, tags="body", attribs={"name": merge_body}, return_first=True
                    )
                )
                for body in other.worldbody:
                    root.append(body)
            self.merge_assets(other)
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

    @property
    def name(self):
        """
        Returns name of this MujocoXML

        Returns:
            str: Name of this MujocoXML
        """
        return self.root.get("model")


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
            return self.naming_prefix + names if not self.exclude_from_prefixing(names) else names
        elif type(names) is list:
            return [self.naming_prefix + name if not self.exclude_from_prefixing(name) else name for name in names]
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
            if (visible and sim.model.site_rgba[vis_g_id][3] < 0) or (
                not visible and sim.model.site_rgba[vis_g_id][3] > 0
            ):
                # We toggle the alpha value
                sim.model.site_rgba[vis_g_id][3] = -sim.model.site_rgba[vis_g_id][3]

    def exclude_from_prefixing(self, inp):
        """
        A function that should take in an arbitrary input and return either True or False, determining whether the
        corresponding name to @inp should have naming_prefix added to it. Must be defined by subclass.

        Args:
            inp (any): Arbitrary input, depending on subclass. Can be str, ET.Element, etc.

        Returns:
            bool: True if we should exclude the associated name(s) with @inp from being prefixed with naming_prefix
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        Name for this model. Should be unique.

        Returns:
            str: Unique name for this model.
        """
        raise NotImplementedError

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
    def sensors(self):
        """
        Returns:
             list: Sensor names for this model
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
    def important_sensors(self):
        """
        Dict of important sensors enabled for this model.

        Returns:
            dict: Important sensors for this model, where each specific sensor name is mapped from keyword string
                entries in the dict
        """
        raise NotImplementedError

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


class MujocoXMLModel(MujocoXML, MujocoModel):
    """
    Base class for all MujocoModels that are based on a raw XML file.

    Args:
        fname (str): Path to relevant xml file from which to create this robot instance
        idn (int or str): Number or some other unique identification string for this model instance
    """

    def __init__(self, fname, idn=0):
        super().__init__(fname)

        # Set id and add prefixes to all body names to prevent naming clashes
        self.idn = idn

        # Define other variables that get filled later
        self.mount = None

        # Define filter method to automatically add a default name to visual / collision geoms if encountered
        group_mapping = {
            None: "col",
            "0": "col",
            "1": "vis",
        }
        ctr_mapping = {
            "col": 0,
            "vis": 0,
        }

        def _add_default_name_filter(element, parent):
            # Run default filter
            filter_key = _element_filter(element=element, parent=parent)
            # Also additionally modify element if it is (a) a geom and (b) has no name
            if element.tag == "geom" and element.get("name") is None:
                group = group_mapping[element.get("group")]
                element.set("name", f"g{ctr_mapping[group]}_{group}")
                ctr_mapping[group] += 1
            # Return default filter key
            return filter_key

        # Parse element tree to get all relevant bodies, joints, actuators, and geom groups
        self._elements = sort_elements(root=self.root, element_filter=_add_default_name_filter)
        assert (
            len(self._elements["root_body"]) == 1
        ), "Invalid number of root bodies found for robot model. Expected 1," "got {}".format(
            len(self._elements["root_body"])
        )
        self._elements["root_body"] = self._elements["root_body"][0]
        self._elements["bodies"] = (
            [self._elements["root_body"]] + self._elements["bodies"]
            if "bodies" in self._elements
            else [self._elements["root_body"]]
        )
        self._root_body = self._elements["root_body"].get("name")
        self._bodies = [e.get("name") for e in self._elements.get("bodies", [])]
        self._joints = [e.get("name") for e in self._elements.get("joints", [])]
        self._actuators = [e.get("name") for e in self._elements.get("actuators", [])]
        self._sites = [e.get("name") for e in self._elements.get("sites", [])]
        self._sensors = [e.get("name") for e in self._elements.get("sensors", [])]
        self._contact_geoms = [e.get("name") for e in self._elements.get("contact_geoms", [])]
        self._visual_geoms = [e.get("name") for e in self._elements.get("visual_geoms", [])]
        self._base_offset = string_to_array(self._elements["root_body"].get("pos", "0 0 0"))

        # Update all xml element prefixes
        add_prefix(root=self.root, prefix=self.naming_prefix, exclude=self.exclude_from_prefixing)

        # Recolor all collision geoms appropriately
        recolor_collision_geoms(root=self.worldbody, rgba=self.contact_geom_rgba)

        # Add default materials
        if macros.USING_INSTANCE_RANDOMIZATION:
            tex_element, mat_element, _, used = add_material(root=self.worldbody, naming_prefix=self.naming_prefix)
            # Only add if material / texture was actually used
            if used:
                self.asset.append(tex_element)
                self.asset.append(mat_element)

    def exclude_from_prefixing(self, inp):
        """
        By default, don't exclude any from being prefixed
        """
        return False

    @property
    def base_offset(self):
        """
        Provides position offset of root body.

        Returns:
            3-array: (x,y,z) pos value of root_body body element. If no pos in element, returns all zeros.
        """
        return self._base_offset

    @property
    def name(self):
        return "{}{}".format(type(self).__name__, self.idn)

    @property
    def naming_prefix(self):
        return "{}_".format(self.idn)

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
    def important_sites(self):
        return self.correct_naming(self._important_sites)

    @property
    def important_geoms(self):
        return self.correct_naming(self._important_geoms)

    @property
    def important_sensors(self):
        return self.correct_naming(self._important_sensors)

    @property
    def _important_sites(self):
        """
        Dict of sites corresponding to the important site geoms (e.g.: used to aid visualization during sim).

        Returns:
            dict: Important site geoms, where each specific geom name is mapped from keyword string entries
                in the dict. Note that the mapped sites should be the RAW site names found directly in the XML file --
                the naming prefix will be automatically added in the public method call
        """
        raise NotImplementedError

    @property
    def _important_geoms(self):
        """
        Geoms corresponding to important components of this model. String keywords should be mapped to lists of geoms.

        Returns:
            dict of list: Important set of geoms, where each set of geoms are grouped as a list and are
                organized by keyword string entries into a dict. Note that the mapped geoms should be the RAW geom
                names found directly in the XML file -- the naming prefix will be automatically added in the
                public method call
        """
        raise NotImplementedError

    @property
    def _important_sensors(self):
        """
        Dict of important sensors enabled for this model.

        Returns:
            dict: Important sensors for this model, where each specific sensor name is mapped from keyword string
                entries in the dict. Note that the mapped geoms should be the RAW sensor names found directly in the
                XML file -- the naming prefix will be automatically added in the public method call
        """
        raise NotImplementedError

    @property
    def contact_geom_rgba(self):
        """
        RGBA color to assign to all contact geoms for this model

        Returns:
            4-array: (r,g,b,a) values from 0 to 1 for this model's set of contact geoms
        """
        raise NotImplementedError

    @property
    def bottom_offset(self):
        """
        Returns vector from model root body to model bottom.
        Useful for, e.g. placing models on a surface.
        By default, this corresponds to the root_body's base offset.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        return self.base_offset

    @property
    def top_offset(self):
        raise NotImplementedError

    @property
    def horizontal_radius(self):
        raise NotImplementedError
