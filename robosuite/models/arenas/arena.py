import numpy as np

from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from robosuite.utils.mjcf_utils import new_geom, new_body, new_joint


class Arena(MujocoXML):
    """Base arena class."""

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

    def add_pos_indicator(self):
        """Adds a new position indicator."""
        body = new_body(name="pos_indicator")
        body.append(
            new_geom(
                "sphere",
                [0.03],
                rgba=[1, 0, 0, 0.5],
                group=1,
                contype="0",
                conaffinity="0",
            )
        )
        body.append(new_joint(type="free", name="pos_indicator"))
        self.worldbody.append(body)
