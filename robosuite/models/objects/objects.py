import copy
import xml.etree.ElementTree as ET

from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import string_to_array, array_to_string, CustomMaterial


class MujocoObject:
    """
    Base class for all objects.

    We use Mujoco Objects to implement all objects that
        1) may appear for multiple times in a task
        2) can be swapped between different tasks

    Typical methods return copy so the caller can all joints/attributes as wanted

    Attributes:
        asset (TYPE): Description
    """

    def __init__(self):
        self.asset = ET.Element("asset")

    def get_bottom_offset(self):
        """
        Returns vector from object center to object bottom
        Helps us put objects on a surface

        Returns:
            np.array: eg. np.array([0, 0, -2])

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def get_top_offset(self):
        """
        Returns vector from object center to object top
        Helps us put other objects on this object

        Returns:
            np.array: eg. np.array([0, 0, 2])

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def get_horizontal_radius(self):
        """
        Returns scalar
        If object a,b has horizontal distance d
        a.get_horizontal_radius() + b.get_horizontal_radius() < d
        should mean that a, b has no contact

        Helps us put objects programmatically without them flying away due to
        a huge initial contact force

        Returns:
            Float: radius

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError
        # return 2

    def get_collision(self, name=None, site=False):
        """
        Returns a ET.Element
        It is a <body/> subtree that defines all collision related fields
        of this object.

        Return is a copy

        Args:
            name (None, optional): Assign name to body
            site (False, optional): Add a site (with name @name
                 when applicable) to the returned body

        Returns:
            ET.Element: body

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def get_visual(self, name=None, site=False):
        """
        Returns a ET.Element
        It is a <body/> subtree that defines all visualization related fields
        of this object.

        Return is a copy

        Args:
            name (None, optional): Assign name to body
            site (False, optional): Add a site (with name @name
                 when applicable) to the returned body

        Returns:
            ET.Element: body

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def get_site_attrib_template(self):
        """
        Returns attribs of spherical site used to mark body origin

        Returns:
            Dictionary of default site attributes
        """
        return {
            "pos": "0 0 0",
            "size": "0.002 0.002 0.002",
            "rgba": "1 0 0 1",
            "type": "sphere",
        }


class MujocoXMLObject(MujocoXML, MujocoObject):
    """
    MujocoObjects that are loaded from xml files
    """

    def __init__(self, fname, name=None, joints=None):
        """
        Args:
            fname (TYPE): XML File path
        """
        MujocoXML.__init__(self, fname)

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

    def get_collision(self, site=False):

        collision = copy.deepcopy(self.worldbody.find("./body/body[@name='collision']"))
        collision.attrib.pop("name")
        if self.name is not None:
            collision.attrib["name"] = self.name
            geoms = collision.findall("geom")
            if len(geoms) == 1:
                geoms[0].set("name", self.name)
            else:
                for i in range(len(geoms)):
                    geoms[i].set("name", "{}-{}".format(self.name, i))
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if self.name is not None:
                template["name"] = self.name
            collision.append(ET.Element("site", attrib=template))
        return collision

    def get_visual(self, site=False):

        visual = copy.deepcopy(self.worldbody.find("./body/body[@name='visual']"))
        visual.attrib.pop("name")
        if self.name is not None:
            visual.attrib["name"] = self.name
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if self.name is not None:
                template["name"] = self.name
            visual.append(ET.Element("site", attrib=template))
        return visual


class MujocoGeneratedObject(MujocoObject):
    """
    Base class for all programmatically generated mujoco object
    i.e., every MujocoObject that does not have an corresponding xml file
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
        joints=None,
    ):
        """
        Args:
            size ([float], optional): of size 1 - 3

            rgba (([float, float, float, float]), optional): Color

            density (float, optional): Density

            friction ([float], optional): of size 3, corresponding to sliding friction,
                torsional friction, and rolling friction. A single float can also be
                specified, in order to set the sliding friction (the other values) will
                be set to the MuJoCo default. See http://www.mujoco.org/book/modeling.html#geom 
                for details.

            solref ([float], optional): of size 2. MuJoCo solver parameters that handle contact.
                See http://www.mujoco.org/book/XMLreference.html for more details.

            solimp ([float], optional): of size 3. MuJoCo solver parameters that handle contact.
                See http://www.mujoco.org/book/XMLreference.html for more details.

            material (CustomMaterial, optional): if "default", add a template material and texture for this
                object that is used to color the geom(s).
                Otherwise, input is expected to be a CustomMaterial object

                See http://www.mujoco.org/book/XMLreference.html#asset for specific details on attributes expected for
                Mujoco texture / material tags, respectively

                Note that specifying a custom texture in this way automatically overrides any rgba values set

            joints ([dict]): list of dictionaries - each dictionary corresponds to a joint that will be created for this
                object. The dictionary should specify the joint attributes (type, pos, etc.) according to the MuJoCo
                xml specification.
        """
        super().__init__()

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
        if joints is None:
            self.joints = [{'type': 'free'}]  # default free joint
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
        return {"pos": "0 0 0", "group": "1"}

    @staticmethod
    def get_visual_attrib_template():
        return {"conaffinity": "0", "contype": "0", "group": "1"}

    def append_material(self, material):
        """
        Adds a new texture / material combination to the assets subtree of this XML
        Input is expected to be a CustomMaterial object

        See http://www.mujoco.org/book/XMLreference.html#asset for specific details on attributes expected for
        Mujoco texture / material tags, respectively

        Note that the "file" attribute for the "texture" tag should be specified relative to the textures directory
            located in robosuite/models/assets/textures/
        """
        # First check if asset attribute exists; if not, define the asset attribute
        if not hasattr(self, "asset"):
            self.asset = ET.Element("asset")
        # Add texture and material inputs to asset
        self.asset.append(ET.Element("texture", attrib=material.tex_attrib))
        self.asset.append(ET.Element("material", attrib=material.mat_attrib))

    def _get_collision(self, site=False, ob_type="box"):
        main_body = ET.Element("body")
        main_body.set("name", self.name)
        template = self.get_collision_attrib_template()
        template["name"] = self.name
        template["type"] = ob_type
        if self.material == "default":
            template["rgba"] = "0.5 0.5 0.5 1" # mujoco default
            template["material"] = "{}_mat".format(self.name)
        elif self.material is not None:
            template["material"] = self.material.mat_attrib["name"]
        else:
            template["rgba"] = array_to_string(self.rgba)
        template["size"] = array_to_string(self.size)
        template["density"] = str(self.density)
        template["friction"] = array_to_string(self.friction)
        template["solref"] = array_to_string(self.solref)
        template["solimp"] = array_to_string(self.solimp)
        main_body.append(ET.Element("geom", attrib=template))
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["name"] = self.name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def _get_visual(self, site=False, ob_type="box"):
        main_body = ET.Element("body")
        main_body.set("name", self.name)
        template = self.get_visual_attrib_template()
        template["type"] = ob_type
        if self.material == "default":
            template["material"] = "{}_mat".format(self.name)
        elif self.material is not None:
            template["material"] = self.material["material"]["name"]
        else:
            template["rgba"] = array_to_string(self.rgba)
        template["size"] = array_to_string(self.size)
        main_body.append(ET.Element("geom", attrib=template))
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["name"] = self.name
            main_body.append(ET.Element("site", attrib=template))
        return main_body
