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
        self.object_scales = {}

        # Add mocap bodies to self.root for mocap control in mjviewer UI for robot control
        mocap_body_1 = new_body(name="left_eef_target", pos="0 0 -1", mocap=True)
        mocap_body_1_geom = new_geom(
            name="left_eef_target_box",
            type="box",
            size="0.05 0.05 0.05",
            rgba="0.898 0.420 0.435 0.5",
            conaffinity="0",
            contype="0",
            group="2",
        )
        mocap_body_1_sphere = new_geom(
            name="left_eef_target_sphere",
            type="sphere",
            size="0.01",
            pos="0 0 0",
            rgba="0.898 0.420 0.435 0.8",
            conaffinity="0",
            contype="0",
            group="2",
        )
        mocap_body_2 = new_body(name="right_eef_target", pos="0 0 -1", mocap=True)
        mocap_body_2_geom = new_geom(
            name="right_eef_target_box",
            type="box",
            size="0.05 0.05 0.05",
            rgba="0.208 0.314 0.439 0.5",
            conaffinity="0",
            contype="0",
            group="2",
        )
        mocap_body_2_sphere = new_geom(
            name="right_eef_target_sphere",
            type="sphere",
            size="0.01",
            pos="0 0 0",
            rgba="0.208 0.314 0.439 0.8",
            conaffinity="0",
            contype="0",
            group="2",
        )
        # Append the box and sphere geometries to their respective mocap bodies
        mocap_body_1.append(mocap_body_1_geom)
        mocap_body_1.append(mocap_body_1_sphere)
        mocap_body_2.append(mocap_body_2_geom)
        mocap_body_2.append(mocap_body_2_sphere)
        # Add the mocap bodies to the world
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

    def _get_geoms(self, root, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a geom element

        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call

        Returns:
            list: array of (parent, child) tuples where the child element is a geom type
        """
        return self._get_elements(root, "geom", _parent)

    def _get_elements(self, root, type, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a specific type of element

        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call

        Returns:
            list: array of (parent, child) tuples where the child element is of type
        """
        # Initialize return array
        elem_pairs = []
        # If the parent exists and this is a desired element, we add this current (parent, element) combo to the output
        if _parent is not None and root.tag == type:
            elem_pairs.append((_parent, root))
        # Loop through all children elements recursively and add to pairs
        for child in root:
            elem_pairs += self._get_elements(child, type, _parent=root)

        # Return all found pairs
        return elem_pairs

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

        self.object_scales[obj.get("name")] = scale

        # scale geoms
        geom_pairs = self._get_geoms(obj)
        for _, (_, element) in enumerate(geom_pairs):
            g_pos = element.get("pos")
            g_size = element.get("size")
            if g_pos is not None:
                g_pos = array_to_string(string_to_array(g_pos) * scale)
                element.set("pos", g_pos)
            if g_size is not None:
                g_size_np = string_to_array(g_size)
                # handle cases where size is not 3 dimensional
                if len(g_size_np) == 3:
                    g_size_np = g_size_np * scale
                elif len(g_size_np) == 2:
                    scale = np.array(scale).reshape(-1)
                    if len(scale) == 1:
                        g_size_np[1] *= scale
                    elif len(scale) == 3:
                        # g_size_np[0] *= np.mean(scale[:2])
                        g_size_np[0] *= np.mean(scale[:2])  # width
                        g_size_np[1] *= scale[2]  # height
                    else:
                        raise ValueError
                else:
                    raise ValueError
                g_size = array_to_string(g_size_np)
                element.set("size", g_size)

        # scale meshes
        meshes = self.asset.findall("mesh")
        for elem in meshes:
            m_scale = elem.get("scale")
            if m_scale is not None:
                m_scale = string_to_array(m_scale)
            else:
                m_scale = np.ones(3)

            m_scale *= scale
            elem.set("scale", array_to_string(m_scale))

        # scale bodies
        body_pairs = self._get_elements(obj, "body")
        for (_, elem) in body_pairs:
            b_pos = elem.get("pos")
            if b_pos is not None:
                b_pos = string_to_array(b_pos) * scale
                elem.set("pos", array_to_string(b_pos))

        # scale joints
        joint_pairs = self._get_elements(obj, "joint")
        for (_, elem) in joint_pairs:
            j_pos = elem.get("pos")
            if j_pos is not None:
                j_pos = string_to_array(j_pos) * scale
                elem.set("pos", array_to_string(j_pos))

        # scale sites
        site_pairs = self._get_elements(self.worldbody, "site")
        for (_, elem) in site_pairs:
            s_pos = elem.get("pos")
            if s_pos is not None:
                s_pos = string_to_array(s_pos) * scale
                elem.set("pos", array_to_string(s_pos))

            s_size = elem.get("size")
            if s_size is not None:
                s_size_np = string_to_array(s_size)
                # handle cases where size is not 3 dimensional
                if len(s_size_np) == 3:
                    s_size_np = s_size_np * scale
                elif len(s_size_np) == 2:
                    scale = np.array(scale).reshape(-1)
                    if len(scale) == 1:
                        s_size_np *= scale
                    elif len(scale) == 3:
                        s_size_np[0] *= np.mean(scale[:2])  # width
                        s_size_np[1] *= scale[2]  # height
                    else:
                        raise ValueError
                elif len(s_size_np) == 1:
                    s_size_np *= np.mean(scale)
                else:
                    raise ValueError
                s_size = array_to_string(s_size_np)
                elem.set("size", s_size)
