import numpy as np

from robosuite.models.objects import PrimitiveObject
from robosuite.utils.mjcf_utils import get_size


class BallObject(PrimitiveObject):
    """
    A ball (sphere) object.

    Args:
        size (1-tuple of float): (radius) size parameters for this ball object
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
        size = get_size(size, size_max, size_min, [0.07], [0.03])
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
        assert len(self.size) == 1, "ball size should have length 1"

    def _get_object_subtree(self):
        return self._get_object_subtree_(ob_type="sphere")

    @property
    def bottom_offset(self):
        return np.array([0, 0, -1 * self.size[0]])

    @property
    def top_offset(self):
        return np.array([0, 0, self.size[0]])

    @property
    def horizontal_radius(self):
        return self.size[0]
