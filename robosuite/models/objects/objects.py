import copy
import xml.etree.ElementTree as ET
import numpy as np

from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import string_to_array, array_to_string


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

    def __init__(self, fname):
        """
        Args:
            fname (TYPE): XML File path
        """
        MujocoXML.__init__(self, fname)

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

    def get_collision(self, name=None, site=False):

        collision = copy.deepcopy(self.worldbody.find("./body/body[@name='collision']"))
        collision.attrib.pop("name")
        if name is not None:
            collision.attrib["name"] = name
            geoms = collision.findall("geom")
            if len(geoms) == 1:
                geoms[0].set("name", name)
            else:
                for i in range(len(geoms)):
                    geoms[i].set("name", "{}-{}".format(name, i))
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            collision.append(ET.Element("site", attrib=template))
        return collision

    def get_visual(self, name=None, site=False):

        visual = copy.deepcopy(self.worldbody.find("./body/body[@name='visual']"))
        visual.attrib.pop("name")
        if name is not None:
            visual.attrib["name"] = name
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
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
        density_range=None,
        friction_range=None,
        add_material=False,
    ):
        """
        Provides default initialization of physical attributes:
            also supports randomization of (rgba, density, friction).
            - rgb is randomly generated if rgba='random' (alpha will be 1 in this case)
            - If density is None and density_range is not:
              Density is chosen uniformly at random specified from density range,
                  i.e. density_range = [50, 100, 1000]
            - If friction is None and friction_range is not:
              Tangential Friction is chosen uniformly at random from friction_range

        Args:
            size ([float], optional): of size 1 - 3
            rgba (([float, float, float, float]), optional): Color
            density (float, optional): Density
            friction (float, optional): tangential friction
                see http://www.mujoco.org/book/modeling.html#geom for details
            density_range ([float,float], optional): range for random choice
            friction_range ([float,float], optional): range for random choice
            add_material (bool, optional): if True, add a material and texture for this 
                object that is used to color the geom(s).
        """
        super().__init__()

        self.name = name

        if size is None:
            self.size = [0.05, 0.05, 0.05]
        else:
            self.size = size

        if rgba is None:
            self.rgba = [1, 0, 0, 1]
        elif rgba == "random":
            self.rgba = np.array([np.random.uniform(0, 1) for i in range(3)] + [1])
        else:
            assert len(rgba) == 4, "rgba must be a length 4 array"
            self.rgba = rgba

        if density is None:
            if density_range is not None:
                self.density = np.random.choice(density_range)
            else:
                self.density = 1000  # water
        else:
            self.density = density

        if friction is None:
            if friction_range is not None:
                self.friction = [np.random.choice(friction_range), 0.005, 0.0001]
            else:
                self.friction = [1, 0.005, 0.0001]  # MuJoCo default
        elif hasattr(type(friction), "__len__"):
            assert len(friction) == 3, "friction must be a length 3 array or a float"
            self.friction = friction
        else:
            self.friction = [friction, 0.005, 0.0001]

        # add in texture and material for this object (for domain randomization)
        self.add_material = add_material
        if add_material:
            self.asset = self._get_asset()

        self.sanity_check()

    def sanity_check(self):
        """
        Checks if data provided makes sense.
        Called in __init__()
        For subclasses to inherit from
        """
        pass

    def get_collision_attrib_template(self):
        return {"pos": "0 0 0", "group": "1"}

    def get_visual_attrib_template(self):
        return {"conaffinity": "0", "contype": "0", "group": "1"}

    def get_texture_attrib_template(self):
        return {
            "name": "{}_tex".format(self.name), 
            "type": "cube", 
            "builtin": "flat", 
            "rgb1": array_to_string(self.rgba[:3]), 
            "rgb2": array_to_string(self.rgba[:3]), 
            "width": "100", 
            "height": "100",
        }

    def get_material_attrib_template(self):
        return {
            "name": "{}_mat".format(self.name), 
            "texture": "{}_tex".format(self.name), 
            # "specular": "0.75", 
            # "shininess": "0.03",
        }

    def _get_collision(self, site=False, ob_type="box"):
        main_body = ET.Element("body")
        main_body.set("name", self.name)
        template = self.get_collision_attrib_template()
        template["name"] = self.name
        template["type"] = ob_type
        if self.add_material:
            template["rgba"] = "0.5 0.5 0.5 1" # mujoco default
            template["material"] = "{}_mat".format(self.name)
        else:
            template["rgba"] = array_to_string(self.rgba)
        template["size"] = array_to_string(self.size)
        template["density"] = str(self.density)
        template["friction"] = array_to_string(self.friction)
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
        if self.add_material:
            template["material"] = "{}_mat".format(self.name)
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

    def _get_asset(self):
        # Add texture and material elements
        assert(self.add_material)
        asset = ET.Element("asset")
        tex_template = self.get_texture_attrib_template()
        mat_template = self.get_material_attrib_template()
        asset.append(ET.Element("texture", attrib=tex_template))
        asset.append(ET.Element("material", attrib=mat_template))
        return asset
