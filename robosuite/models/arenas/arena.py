import numpy as np

from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import (
    ENVIRONMENT_COLLISION_COLOR,
    array_to_string,
    find_elements,
    new_body,
    new_element,
    new_geom,
    new_joint,
    recolor_collision_geoms,
    string_to_array,
)


class Arena(MujocoXML):
    """Base arena class."""

    def __init__(self, fname):
        super().__init__(fname)
        # Get references to floor and bottom
        self.bottom_pos = np.zeros(3)
        self.floor = self.worldbody.find("./geom[@name='floor']")

        # Add mocap bodies to self.root for mocap control in mjviewer UI for robot control
        mocap_body_1 = new_body(name="left_eef_target", pos="0 0 -1", mocap=True)
        mocap_body_1_geom = new_geom(
            name="left_eef_target",
            type="box",
            size="0.05 0.05 0.05",
            rgba="0.898 0.420 0.435 0.5",
            conaffinity="0",
            contype="0",
            group="2",
        )
        mocap_body_2 = new_body(name="right_eef_target", pos="0 0 -1", mocap=True)
        mocap_body_2_geom = new_geom(
            name="right_eef_target",
            type="box",
            size="0.05 0.05 0.05",
            rgba="0.208 0.314 0.439 0.5",
            conaffinity="0",
            contype="0",
            group="2",
        )
        mocap_body_1.append(mocap_body_1_geom)
        mocap_body_2.append(mocap_body_2_geom)
        self.worldbody.append(mocap_body_1)
        self.worldbody.append(mocap_body_2)

        # Run any necessary post-processing on the model
        self._postprocess_arena()

        # Recolor all geoms
        recolor_collision_geoms(
            root=self.worldbody,
            rgba=ENVIRONMENT_COLLISION_COLOR,
            exclude=lambda e: True if e.get("name", None) == "floor" else False,
        )

    def set_origin(self, offset):
        """
        Applies a constant offset to all objects.

        Args:
            offset (3-tuple): (x,y,z) offset to apply to all nodes in this XML
        """
        offset = np.array(offset)
        for node in self.worldbody.findall("./*[@pos]"):
            cur_pos = string_to_array(node.get("pos"))
            new_pos = cur_pos + offset
            node.set("pos", array_to_string(new_pos))

    def set_camera(self, camera_name, pos, quat, camera_attribs=None):
        """
        Sets a camera with @camera_name. If the camera already exists, then this overwrites its pos and quat values.

        Args:
            camera_name (str): Camera name to search for / create
            pos (3-array): (x,y,z) coordinates of camera in world frame
            quat (4-array): (w,x,y,z) quaternion of camera in world frame
            camera_attribs (dict): If specified, should be additional keyword-mapped attributes for this camera.
                See http://www.mujoco.org/book/XMLreference.html#camera for exact attribute specifications.
        """
        # Determine if camera already exists
        camera = find_elements(root=self.worldbody, tags="camera", attribs={"name": camera_name}, return_first=True)

        # Compose attributes
        if camera_attribs is None:
            camera_attribs = {}
        camera_attribs["pos"] = array_to_string(pos)
        camera_attribs["quat"] = array_to_string(quat)

        if camera is None:
            # If camera doesn't exist, then add a new camera with the specified attributes
            self.worldbody.append(new_element(tag="camera", name=camera_name, **camera_attribs))
        else:
            # Otherwise, we edit all specified attributes in that camera
            for attrib, value in camera_attribs.items():
                camera.set(attrib, value)

    def _postprocess_arena(self):
        """
        Runs any necessary post-processing on the imported Arena model
        """
        pass
