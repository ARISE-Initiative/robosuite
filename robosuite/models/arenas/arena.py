import numpy as np

from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from robosuite.utils.mjcf_utils import new_geom, new_body, new_joint


class Arena(MujocoXML):
    """Base arena class."""

    def set_origin(self, offset):
        """Applies a constant offset to all objects."""
        offset = np.array(offset)
        for node in self.worldbody.findall("./*[@pos]"):
            cur_pos = string_to_array(node.get("pos"))
            new_pos = cur_pos + offset
            node.set("pos", array_to_string(new_pos))

    def add_pos_indicator(self, num=1, geom_types=None, geom_sizes=None, geom_rgbas=None):
        """Adds a new position indicator."""
        for i in range(num):
            body = new_body(name="pos_indicator_{}".format(i))
            if geom_types is not None:
                geom_type = geom_types[i]
            else:
                geom_type = "sphere"
            if geom_rgbas is not None:
                rgba = geom_rgbas[i]
            else:
                rgba = [1, 0, 0, 0.5]
            if geom_sizes is not None:
                geom_size = geom_sizes[i]
            else:
                geom_size = [0.03]
            body.append(
                new_geom(
                    geom_type,
                    geom_size,
                    rgba=rgba,
                    group=1,
                    contype="0",
                    conaffinity="0",
                )
            )
            body.append(new_joint(type="free", name="pos_indicator_{}".format(i)))
            self.worldbody.append(body)
