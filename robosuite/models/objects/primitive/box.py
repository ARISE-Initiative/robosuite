import numpy as np

from robosuite.models.objects import PrimitiveObject
from robosuite.utils.mjcf_utils import get_size


class BoxObject(PrimitiveObject):
    """
    A box object.

    Args:
        size (3-tuple of float): (half-x, half-y, half-z) size parameters for this box object
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
        rng=None,
    ):
        size = get_size(size, size_max, size_min, [0.07, 0.07, 0.07], [0.03, 0.03, 0.03], rng=rng)
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
        assert len(self.size) == 3, "box size should have length 3"

    def _get_object_subtree(self):
        return self._get_object_subtree_(ob_type="box")

    @property
    def bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    @property
    def top_offset(self):
        return np.array([0, 0, self.size[2]])

    @property
    def horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2)

    def get_bounding_box_half_size(self):
        return np.array([self.size[0], self.size[1], self.size[2]])
