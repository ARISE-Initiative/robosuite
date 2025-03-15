import numpy as np
import xml.etree.ElementTree as ET
import os
import random
import string

import robosuite
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion, string_to_array

def postprocess_model_xml(xml_str):
    """
    New version of postprocess model xml that only replaces robosuite file paths if necessary (otherwise
    there is an error with the "max" operation), and also replaces robosuite-model-zoo file paths
    if necessary.
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
        # maybe replace all paths to robosuite assets
        check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite"]
        if len(check_lst) > 0:
            ind = max(check_lst)  # last occurrence index
            new_path_split = path_split + old_path_split[ind + 1:]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)

        # maybe replace all paths to robosuite model zoo assets
        # check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite-model-zoo-dev"]
        # if len(check_lst) > 0:
        #     ind = max(check_lst)  # last occurrence index
        #     new_path_split = os.path.split(robosuite_model_zoo.__file__)[0].split("/")[:-1] + old_path_split[ind + 1:]
        #     new_path = "/".join(new_path_split)
        #     elem.set("file", new_path)

    return ET.tostring(root, encoding="utf8").decode("utf8")

class ScaledMujocoXMLObject(MujocoXMLObject):
    def __init__(self, name, xml_path, scale):
        if isinstance(scale, float):
            scale = [scale, scale, scale]
        elif isinstance(scale, tuple) or isinstance(scale, list):
            assert len(scale) == 3
            scale = tuple(scale)
        else:
            raise Exception("got invalid scale: {}".format(scale))
        scale = np.array(scale)

        tree = ET.parse(xml_path)
        folder = os.path.dirname(xml_path)
        root = tree.getroot()

        # modify mesh scales
        asset = root.find("asset")
        mesh = asset.find("mesh")
        scale_to_set = scale
        existing_scale = mesh.get("scale")
        if existing_scale is not None:
            scale_to_set = string_to_array(existing_scale) * scale
        mesh.set("scale", array_to_string(scale_to_set))

        # modify sites for collision (assumes we can just scale up the locations - may or may not work)
        for n in ["bottom_site", "top_site", "horizontal_radius_site"]:
            site = root.find("worldbody/body/site[@name='{}']".format(n))
            pos = string_to_array(site.get("pos"))
            pos = scale * pos
            site.set("pos", array_to_string(pos))

        # write modified xml (and make sure to postprocess any paths just in case)
        xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
        xml_str = postprocess_model_xml(xml_str)
        random_code = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        new_xml_path = os.path.join(folder, "{}.xml".format(random_code))
        f = open(new_xml_path, "w")
        f.write(xml_str)
        f.close()
        # print(f"Write to {new_xml_path}")

        # initialize object with new xml we wrote
        super().__init__(
            # xml_path_completion("objects/{}.xml".format(obj_name)),
            fname=new_xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

        # clean up xml - we don't need it anymore
        os.remove(new_xml_path)

class StoveObject(MujocoXMLObject):
    def __init__(
            self,
            name,
            joints=None):
        super().__init__(xml_path_completion("objects/stove.xml"),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

    @property
    def bottom_offset(self):
        return np.array([0, 0, -2 * self.height])

    @property
    def top_offset(self):
        return np.array([0, 0, 2 * self.height])

    @property
    def horizontal_radius(self):
        return self.length * np.sqrt(2)


class ServingRegionObject(MujocoXMLObject):
    def __init__(
            self,
            name,
            joints=None):

        super().__init__(xml_path_completion("objects/serving_region.xml"),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)


class CabinetObject(MujocoXMLObject):
    def __init__(
            self,
            name,
            joints=None):
        super().__init__(xml_path_completion("objects/cabinet.xml"),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

    @property
    def bottom_offset(self):
        return np.array([0, 0, -2 * self.height])

    @property
    def top_offset(self):
        return np.array([0, 0, 2 * self.height])

    @property
    def horizontal_radius(self):
        return self.length * np.sqrt(2)


class ButtonObject(MujocoXMLObject):
    def __init__(self, name, friction=None, damping=None):
        super().__init__(xml_path_completion("objects/button.xml"),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

        # Set relevant body names
        self.hinge_joint = self.naming_prefix + "hinge"

        self.friction = friction
        self.damping = damping

    def _set_friction(self, friction):
        """
        Helper function to override the drawer friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_damping(self, damping):
        """
        Helper function to override the drawer friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of drawer handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle"
        })
        return dic

class AppleObject(MujocoXMLObject):
    """
    Apple object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/apple.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class AppleJuiceObject(ScaledMujocoXMLObject):
    """
    Can object
    """

    def __init__(self, name, scale=1.0):
        xml_path = xml_path_completion("objects/apple_juice.xml")
        super().__init__(
            name=name,
            xml_path=xml_path,
            scale=scale,
        )

class CoffeeObject(ScaledMujocoXMLObject):
    """
    Can object
    """

    def __init__(self, name, scale=1.0):
        xml_path = xml_path_completion("objects/coffee.xml")
        super().__init__(
            name=name,
            xml_path=xml_path,
            scale=scale,
        )

class LimoncelloObject(ScaledMujocoXMLObject):
    """
    Can object
    """

    def __init__(self, name, scale=1.0):
        xml_path = xml_path_completion("objects/limoncello.xml")
        super().__init__(
            name=name,
            xml_path=xml_path,
            scale=scale,
        )

class JuiceGrayObject(ScaledMujocoXMLObject):
    """
    Can object
    """

    def __init__(self, name, scale=1.0):
        xml_path = xml_path_completion("objects/juice_gray.xml")
        super().__init__(
            name=name,
            xml_path=xml_path,
            scale=scale,
        )

class StrawberryJuiceObject(ScaledMujocoXMLObject):
    """
    Can object
    """

    def __init__(self, name, scale=1.0):
        xml_path = xml_path_completion("objects/strawberry_juice.xml")
        super().__init__(
            name=name,
            xml_path=xml_path,
            scale=scale,
        )

class BeerObject(ScaledMujocoXMLObject):
    """
    Can object
    """

    def __init__(self, name, scale=1.0):
        xml_path = xml_path_completion("objects/beer.xml")
        super().__init__(
            name=name,
            xml_path=xml_path,
            scale=scale,
        )

class SodaObject(ScaledMujocoXMLObject):
    """
    Can object
    """

    def __init__(self, name, scale=1.0):
        xml_path = xml_path_completion("objects/soda.xml")
        super().__init__(
            name=name,
            xml_path=xml_path,
            scale=scale,
        )

class TeaObject(ScaledMujocoXMLObject):
    """
    Can object
    """

    def __init__(self, name, scale=1.0):
        xml_path = xml_path_completion("objects/tea.xml")
        super().__init__(
            name=name,
            xml_path=xml_path,
            scale=scale,
        )

class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/milk.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bottle.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class CanObject(ScaledMujocoXMLObject):
    """
    Coke can object (used in PickPlace)
    """

    def __init__(self, name, scale=1.0):
        super().__init__(
            xml_path=xml_path_completion("objects/can.xml"),
            name=name,
            scale=scale,
        )


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/lemon.xml"), name=name, obj_type="all", duplicate_collision_geoms=True
        )


class MilkBlueObject(ScaledMujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name, scale=1.0):
        super().__init__(
            xml_path=xml_path_completion("objects/milk_blue.xml"),
            name=name,
            scale=scale
        )

class MilkGreenObject(ScaledMujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name, scale=1.0):
        super().__init__(
            xml_path=xml_path_completion("objects/milk_green.xml"),
            name=name,
            scale=scale
        )

class MilkBlackObject(ScaledMujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name, scale=1.0):
        super().__init__(
            xml_path=xml_path_completion("objects/milk_black.xml"),
            name=name,
            scale=scale
        )

class MilkRedObject(ScaledMujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name, scale=1.0):
        super().__init__(
            xml_path=xml_path_completion("objects/milk_red.xml"),
            name=name,
            scale=scale
        )

class MilkGrayObject(ScaledMujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name, scale=1.0):
        super().__init__(
            xml_path=xml_path_completion("objects/milk_gray.xml"),
            name=name,
            scale=scale
        )

class OatsMilkObject(ScaledMujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name, scale=1.0):
        super().__init__(
            xml_path=xml_path_completion("objects/oats_milk.xml"),
            name=name,
            scale=scale
        )

class ChocolateMilkObject(ScaledMujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name, scale=1.0):
        super().__init__(
            xml_path=xml_path_completion("objects/chocolate_milk.xml"),
            name=name,
            scale=scale
        )

class BreadObject(ScaledMujocoXMLObject):
    """
    Bread loaf object (used in PickPlace)
    """

    def __init__(self, name, scale=1.0):
        super().__init__(
            name=name,
            xml_path=xml_path_completion("objects/bread.xml"),
            scale=scale
        )


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-nut.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/milk-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/bread-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cereal-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/can-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in PegInHole)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/plate-with-hole.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class DoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name, friction=None, damping=None, lock=False):
        xml_path = "objects/door.xml"
        if lock:
            xml_path = "objects/door_lock.xml"
        super().__init__(
            xml_path_completion(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.door_body = self.naming_prefix + "door"
        self.frame_body = self.naming_prefix + "frame"
        self.latch_body = self.naming_prefix + "latch"
        self.hinge_joint = self.naming_prefix + "hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic
