import numpy as np

from robosuite.utils.mjcf_utils import get_size
from robosuite.models.objects import PrimitiveObject, MujocoGeneratedObject


class CylinderObject(PrimitiveObject):
    """
    A cylinder object.

    Args:
        size (2-tuple of float): (radius, half-length) size parameters for this cylinder object
    """

    def __init__(
        self,
        name,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        friction=None,
        rgba=None,
        solref=None,
        solimp=None,
        material=None,
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=True,
    ):
        size = get_size(size, size_max, size_min, [0.07, 0.07], [0.03, 0.03])

        # We override solref, solimp, and joint default values for better stability
        if friction is None:
            friction = [1, 0.01, 0.001]
        if solref is None:
            solref = [0.01, 0.5]
        if joints == "default":
            joints = [{"type": "free", "damping": "0.0001"}]

        super().__init__(
            name=name,
            size=size,
            rgba=rgba,
            density=density,
            friction=friction,
            solref=solref,
            solimp=solimp,
            material=material,
            joints=joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )

    def sanity_check(self):
        """
        Checks to make sure inputted size is of correct length

        Raises:
            AssertionError: [Invalid size length]
        """
        assert len(self.size) == 2, "cylinder size should have length 2"

    def _get_object_subtree(self):
        return self._get_object_subtree_(ob_type="cylinder")

    @staticmethod
    def get_collision_attrib_template():
        """
        Generates template with collision attributes for a given geom

        Extends super method for better stability for contacts

        Returns:
            dict: Initial template with `'pos'` and `'group'` already specified
        """
        template = MujocoGeneratedObject.get_collision_attrib_template()
        # Add condim value
        template["margin"] = "0.001"
        return template

    @property
    def bottom_offset(self):
        return np.array([0, 0, -1 * self.size[1]])

    @property
    def top_offset(self):
        return np.array([0, 0, self.size[1]])

    @property
    def horizontal_radius(self):
        return self.size[0]
