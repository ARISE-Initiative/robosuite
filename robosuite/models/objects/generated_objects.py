import random
import numpy as np
import xml.etree.ElementTree as ET

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site
from robosuite.utils.mjcf_utils import RED, GREEN, BLUE


class PotWithHandlesObject(MujocoGeneratedObject):
    """
    Generates the Pot object with side handles (used in BaxterLift)
    """

    def __init__(
        self,
        body_half_size=None,
        handle_radius=0.01,
        handle_length=0.09,
        handle_width=0.09,
        rgba_body=None,
        rgba_handle_1=None,
        rgba_handle_2=None,
        solid_handle=False,
        thickness=0.025,  # For body
    ):
        super().__init__()
        if body_half_size:
            self.body_half_size = body_half_size
        else:
            self.body_half_size = np.array([0.07, 0.07, 0.07])
        self.thickness = thickness
        self.handle_radius = handle_radius
        self.handle_length = handle_length
        self.handle_width = handle_width
        if rgba_body:
            self.rgba_body = np.array(rgba_body)
        else:
            self.rgba_body = RED
        if rgba_handle_1:
            self.rgba_handle_1 = np.array(rgba_handle_1)
        else:
            self.rgba_handle_1 = GREEN
        if rgba_handle_2:
            self.rgba_handle_2 = np.array(rgba_handle_2)
        else:
            self.rgba_handle_2 = BLUE
        self.solid_handle = solid_handle

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.body_half_size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.body_half_size[2]])

    def get_horizontal_radius(self):
        return np.sqrt(2) * (max(self.body_half_size) + self.handle_length)

    @property
    def handle_distance(self):
        return self.body_half_size[1] * 2 + self.handle_length * 2

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)

        for geom in five_sided_box(
            self.body_half_size, self.rgba_body, 1, self.thickness
        ):
            main_body.append(geom)
        handle_z = self.body_half_size[2] - self.handle_radius
        handle_1_center = [0, self.body_half_size[1] + self.handle_length, handle_z]
        handle_2_center = [
            0,
            -1 * (self.body_half_size[1] + self.handle_length),
            handle_z,
        ]
        # the bar on handle horizontal to body
        main_bar_size = [
            self.handle_width / 2 + self.handle_radius,
            self.handle_radius,
            self.handle_radius,
        ]
        side_bar_size = [self.handle_radius, self.handle_length / 2, self.handle_radius]
        handle_1 = new_body(name="handle_1")
        if self.solid_handle:
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1",
                    pos=[0, self.body_half_size[1] + self.handle_length / 2, handle_z],
                    size=[
                        self.handle_width / 2,
                        self.handle_length / 2,
                        self.handle_radius,
                    ],
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
        else:
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_c",
                    pos=handle_1_center,
                    size=main_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_+",  # + for positive x
                    pos=[
                        self.handle_width / 2,
                        self.body_half_size[1] + self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_-",
                    pos=[
                        -self.handle_width / 2,
                        self.body_half_size[1] + self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )

        handle_2 = new_body(name="handle_2")
        if self.solid_handle:
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2",
                    pos=[0, -self.body_half_size[1] - self.handle_length / 2, handle_z],
                    size=[
                        self.handle_width / 2,
                        self.handle_length / 2,
                        self.handle_radius,
                    ],
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
        else:
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_c",
                    pos=handle_2_center,
                    size=main_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_+",  # + for positive x
                    pos=[
                        self.handle_width / 2,
                        -self.body_half_size[1] - self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_-",
                    pos=[
                        -self.handle_width / 2,
                        -self.body_half_size[1] - self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )

        main_body.append(handle_1)
        main_body.append(handle_2)
        main_body.append(
            new_site(
                name="pot_handle_1",
                rgba=self.rgba_handle_1,
                pos=handle_1_center - np.array([0, 0.005, 0]),
                size=[0.005],
            )
        )
        main_body.append(
            new_site(
                name="pot_handle_2",
                rgba=self.rgba_handle_2,
                pos=handle_2_center + np.array([0, 0.005, 0]),
                size=[0.005],
            )
        )
        main_body.append(new_site(name="pot_center", pos=[0, 0, 0], rgba=[1, 0, 0, 0]))

        return main_body

    def handle_geoms(self):
        return self.handle_1_geoms() + self.handle_2_geoms()

    def handle_1_geoms(self):
        if self.solid_handle:
            return ["handle_1"]
        return ["handle_1_c", "handle_1_+", "handle_1_-"]

    def handle_2_geoms(self):
        if self.solid_handle:
            return ["handle_2"]
        return ["handle_2_c", "handle_2_+", "handle_2_-"]

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


def five_sided_box(size, rgba, group, thickness):
    """
    Args:
        size ([float,flat,float]):
        rgba ([float,float,float,float]): color
        group (int): Mujoco group
        thickness (float): wall thickness

    Returns:
        []: array of geoms corresponding to the
            5 sides of the pot used in BaxterLift
    """
    geoms = []
    x, y, z = size
    r = thickness / 2
    geoms.append(
        new_geom(
            geom_type="box", size=[x, y, r], pos=[0, 0, -z + r], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[x, r, z], pos=[0, -y + r, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[x, r, z], pos=[0, y - r, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[r, y, z], pos=[x - r, 0, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[r, y, z], pos=[-x + r, 0, 0], rgba=rgba, group=group
        )
    )
    return geoms


DEFAULT_DENSITY_RANGE = [200, 500, 1000, 3000, 5000]
DEFAULT_FRICTION_RANGE = [0.25, 0.5, 1, 1.5, 2]


def _get_size(size,
              size_max,
              size_min,
              default_max,
              default_min):
    """
        Helper method for providing a size,
        or a range to randomize from
    """
    if len(default_max) != len(default_min):
        raise ValueError('default_max = {} and default_min = {}'
                         .format(str(default_max), str(default_min)) +
                         ' have different lengths')
    if size is not None:
        if (size_max is not None) or (size_min is not None):
            raise ValueError('size = {} overrides size_max = {}, size_min = {}'
                             .format(size, size_max, size_min))
    else:
        if size_max is None:
            size_max = default_max
        if size_min is None:
            size_min = default_min
        size = np.array([np.random.uniform(size_min[i], size_max[i])
                         for i in range(len(default_max))])
    return size


def _get_randomized_range(val,
                          provided_range,
                          default_range):
    """
        Helper to initialize by either value or a range
        Returns a range to randomize from
    """
    if val is None:
        if provided_range is None:
            return default_range
        else:
            return provided_range
    else:
        if provided_range is not None:
            raise ValueError('Value {} overrides range {}'
                             .format(str(val), str(provided_range)))
        return [val]

class CompositeBoxObject(MujocoGeneratedObject):
    """
    An object constructed out of box geoms to make more intricate shapes.
    """

    def __init__(
        self,
        total_size,
        unit_size,
        geom_locations,
        geom_sizes,
        geom_names=None,
        geom_rgbas=None,
        joint=None,
        rgba=None,
    ):
        """
        Args:
            total_size (list): half-size in each dimension for the complete box

            unit_size (list): half-size in each dimension for the geom grid. The
                @geom_locations are specified in these units.

            geom_locations (list): list of geom locations in the composite. Each 
                location should be a list or tuple of 3 elements and all 
                locations are specified in terms of 2 * @unit_size and are relative
                to the lower left corner of the total box (e.g. (0, 0, 0)
                corresponds to this corner). Note the factor of 2! The x, y, and z
                directions are aligned with the MuJoCo world frame.

            geom_sizes (list): list of geom sizes ordered the same as @geom_locations

            geom_names (list): list of geom names ordered the same as @geom_locations. The
                names will get appended with an underscore to the passed name in @get_collision
                and @get_visual

            geom_rgbas (list): list of geom colors ordered the same as @geom_locations. If 
                passed as an argument, @rgba is ignored.
        """
        super().__init__(joint=joint, rgba=rgba)

        self.total_size = np.array(total_size)
        self.unit_size = np.array(unit_size)
        self.geom_locations = np.array(geom_locations)
        self.geom_sizes = np.array(geom_sizes)
        self.geom_names = list(geom_names) if geom_names is not None else None
        self.geom_rgbas = list(geom_rgbas) if geom_rgbas is not None else None
        self.rgba = rgba

    def get_bottom_offset(self):
        return np.array([0., 0., -self.total_size[2]])

    def get_top_offset(self):
        return np.array(0., 0., self.total_size[2])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.total_size[:2], 2)

    def _make_geoms(self, name=None, site=None, **geom_properties):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)

        for i in range(self.geom_locations.shape[0]):

            # scale each dimension's size by the unit size in that dimension
            size = [
                self.geom_sizes[i][0] * self.unit_size[0],
                self.geom_sizes[i][1] * self.unit_size[1],
                self.geom_sizes[i][2] * self.unit_size[2],
            ]

            # use geom location to convert to position coordinate (the origin is the
            # center of the composite object)
            loc = self.geom_locations[i]
            pos = [
                (-self.total_size[0] + size[0]) + loc[0] * (2. * self.unit_size[0]),
                (-self.total_size[1] + size[1]) + loc[1] * (2. * self.unit_size[1]),
                (-self.total_size[2] + size[2]) + loc[2] * (2. * self.unit_size[2]),
            ]

            # geom name
            if self.geom_names is not None:
                geom_name = "{}_{}".format(name, self.geom_names[i])
            else:
                geom_name = "{}_{}".format(name, i)

            # geom rgba
            if self.geom_rgbas is not None and self.geom_rgbas[i] is not None:
                geom_rgba = self.geom_rgbas[i]
            else:
                geom_rgba = self.rgba

            # add geom
            main_body.append(
                new_geom(
                    size=size, 
                    pos=pos, 
                    name=geom_name,
                    rgba=geom_rgba,
                    **geom_properties,
                )
            )

        return main_body

    def get_collision(self, name=None, site=None):
        geom_properties = {
            'geom_type': 'box',
            'group': 1,
            'density': '100',
        }
        if self.rgba is None:
            # if no color, default to lego material
            geom_properties['material'] = 'lego'
        return self._make_geoms(name=name, site=site, **geom_properties)

    def get_visual(self, name=None, site=None):
        geom_properties = {
            'geom_type': 'box',
            'group': 1,
            'conaffinity': '0', 
            'contype': '0',
            'density': '100',
        }
        if self.rgba is None:
            # if no color, default to lego material
            geom_properties['material'] = 'lego'
        return self._make_geoms(name=name, site=site, **geom_properties)

    def in_box(self, position, object_position):
        """
        Checks whether the object is contained within this CompositeBoxObject.
        Useful for when the CompositeBoxObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the CompositeBoxObject as an axis-aligned grid.

        Args:
            position: 3D body position of CompositeBoxObject
            object_position: 3D position of object to test for insertion
        """
        ub = position + self.total_size
        lb = position - self.total_size

        # fudge factor for the z-check, since after insertion the object falls to table
        lb[2] -= 0.01

        return np.all(object_position > lb) and np.all(object_position < ub)


class BoundingObject(CompositeBoxObject):
    """
    Generates a box with a box-shaped hole cut out of it.
    """

    def __init__(
        self,
        size=[0.1, 0.1, 0.1],
        hole_size=[0.05, 0.05, 0.05],
        hole_location=[0., 0.],
        hole_rgba=None,
        joint=None,
        rgba=None,
    ):
        """
        NOTE: @hole_location should be relative to the center of the object, and be 2D, since
              the z-location is inferred to be at the top of the box.
        """
        # make sure hole fits within box
        assert np.all(hole_size < size)

        self.hole_size = np.array(hole_size)
        self.hole_rgba = np.array(hole_rgba) if hole_rgba is not None else None
        self.hole_location = np.array(hole_location)

        # if hole_location is None:
        #     # find amount the hole can move within the bounding object, and sample a location
        #     # that's relative to the center of the object
        #     x_hole_lim = size[0] - self.hole_size[0] # these are half-sizes
        #     y_hole_lim = size[1] - self.hole_size[1]
        #     x_hole = np.random.uniform(-0.6 * x_hole_lim, 0.6 * x_hole_lim)
        #     y_hole = np.random.uniform(-0.6 * y_hole_lim, 0.6 * y_hole_lim)
        #     self.hole_location = np.array([x_hole, y_hole])

        # specify all geoms in unnormalized position coordinates
        unit_size = [1., 1., 1.]
        geom_args = self._geoms_from_init(
            size=size, 
            hole_size=self.hole_size, 
            hole_location=self.hole_location, 
            hole_rgba=self.hole_rgba,
        )

        super().__init__(
            total_size=size, 
            unit_size=unit_size, 
            joint=joint, 
            rgba=rgba,
            **geom_args,
        )

    def _geoms_from_init(self, size, hole_size, hole_location, hole_rgba):
        """
        Helper function to retrieve geoms to pass to super class, from the size,
        hole size, and hole location.
        """

        # total size - hole size = remaining space on object
        x_hole_lim = size[0] - hole_size[0]
        y_hole_lim = size[1] - hole_size[1]
        x_hole, y_hole = hole_location[0], hole_location[1]

        # we add a top, bottom, left, and right geom that surround the hole, and
        # a lower base geom that can fill up the bottom of the box to make
        # the hole as shallow as it needs to be.
        geom_names = ['top', 'bottom', 'left', 'right', 'hole_base']
        geom_rgbas = [None, None, None, None, hole_rgba]

        # geom sizes
        #
        # take sizes with hole at center and add sampled hole translation
        top_size = [(x_hole_lim + x_hole) / 2., size[1], size[2]]
        bottom_size = [(x_hole_lim - x_hole) / 2., size[1], size[2]]
        left_size = [size[0], (y_hole_lim + y_hole) / 2., size[2]]
        right_size = [size[0], (y_hole_lim - y_hole) / 2., size[2]]
        hole_base_size = [hole_size[0], hole_size[1], (size[2] - hole_size[2])]
        geom_sizes = [top_size, bottom_size, left_size, right_size, hole_base_size]

        # geom locations
        #
        # top and left are at (0, 0), and bottom and right are just translated by 
        # size of hole, and top and left respectively
        top_loc = [0, 0, 0]
        bottom_loc = [top_size[0] + hole_size[0], 0, 0]
        left_loc = [0, 0, 0]
        right_loc = [0, left_size[1] + hole_size[1], 0]
        hole_base_loc = [top_size[0], left_size[1], 0]
        geom_locations = [top_loc, bottom_loc, left_loc, right_loc, hole_base_loc]

        return {
            "geom_locations" : geom_locations,
            "geom_sizes" : geom_sizes,
            "geom_names" : geom_names,
            "geom_rgbas" : geom_rgbas,
        }

#     def in_grid(self, position, object_position, object_size):
#         """
#         Args:
#             position: 3D body position of BoundingObject
#             object_position: 3D position of object to test for insertion
#             object_size: 3D array of x, y, and z half-size bounding box dimensions for object
#         """

#         # convert into hole frame
#         rel_pos = np.array(object_position) - np.array(position)

#         # some tolerance for the object size
#         object_size = np.array(object_size) * 0.95

#         # bounds for object and for hole location
#         object_lb = rel_pos - object_size
#         object_ub = rel_pos + object_size
#         hole_lb = self.hole_location - self.hole_size
#         hole_ub = self.hole_location + self.hole_size

#         # fudge factor for the z-check, since after insertion the object falls to table
#         hole_lb[2] -= 0.01
#         return np.all(object_lb > hole_lb) and np.all(object_ub < hole_ub)        


class BoxPatternObject(CompositeBoxObject):
    """
    An object constructed out of box geoms to make more intricate shapes.
    """

    def __init__(
        self,
        unit_size,
        pattern,
        joint=None,
        rgba=None,
    ):
        """
        Args:
            unit_size (3d array / list): size of each unit block in each dimension

            pattern (3d array / list): array of normalized sizes specifying the
                geometry of the shape. A "0" indicates the absence of a cube and
                a "1" indicates the presence of a full unit block. The dimensions
                correspond to z, x, and y respectively. 
        """

        # number of blocks in z, x, and y
        self.pattern = np.array(pattern)
        self.nz, self.nx, self.ny = self.pattern.shape

        total_size = [self.nx * unit_size[0], self.ny * unit_size[1], self.nz * unit_size[2]]
        geom_args = self._geoms_from_init(self.pattern)
        super().__init__(
            total_size=total_size, 
            unit_size=unit_size, 
            joint=joint, 
            rgba=rgba,
            **geom_args,
        )

    def _geoms_from_init(self, pattern):
        """
        Helper function to retrieve geoms to pass to super class.
        """
        geom_locations = []
        geom_sizes = []
        geom_names = []
        nz, nx, ny = pattern.shape
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    if pattern[k, i, j] > 0:
                        geom_sizes.append([1, 1, 1])
                        geom_locations.append([i, j, k])
                        geom_names.append("{}_{}_{}".format(k, i, j))
        return {
            "geom_locations" : geom_locations,
            "geom_sizes" : geom_sizes,
            "geom_names" : geom_names,
        }


class BoundingPatternObject(BoundingObject, BoxPatternObject):
    """
    Generates a box with a box-shaped hole cut out of it.
    The box-shaped hole satisfies a pattern so that more intricate
    voxelized holes are created.
    """

    def __init__(
        self,
        unit_size,
        pattern,
        size=[0.1, 0.1, 0.1],
        hole_size=[0.05, 0.05, 0.05],
        hole_location=[0., 0.],
        hole_rgba=None,
        joint=None,
        rgba=None,
    ):
        """
        NOTE: @hole_location should be relative to the center of the object, and be 2D, since
              the z-location is inferred to be at the top of the box.
        """

        # make sure hole fits within box
        assert np.all(hole_size < size)

        # number of blocks in z, x, and y for the pattern
        self.pattern = np.array(pattern)
        self.nz, self.nx, self.ny = self.pattern.shape

        self.hole_size = np.array(hole_size)
        self.hole_rgba = np.array(hole_rgba) if hole_rgba is not None else None
        self.hole_location = np.array(hole_location)

        geom_args = self._geoms_from_init(
            unit_size=unit_size,
            pattern=self.pattern,
            size=size, 
            hole_size=self.hole_size, 
            hole_location=self.hole_location, 
            hole_rgba=self.hole_rgba,
        )

        # specify all geoms in unnormalized position coordinates
        unit_size = [1., 1., 1.]

        CompositeBoxObject.__init__(
            self,
            total_size=size, 
            unit_size=unit_size, 
            joint=joint, 
            rgba=rgba,
            **geom_args,
        )

    def _geoms_from_init(self, unit_size, pattern, size, hole_size, hole_location, hole_rgba):
        """
        Helper function to retrieve geoms to pass to super class, from the size,
        hole size, and hole location.
        """
        bounding_geom_args = BoundingObject._geoms_from_init(
            self, 
            size=size, 
            hole_size=hole_size, 
            hole_location=hole_location, 
            hole_rgba=hole_rgba,
        )
        pattern_geom_args = BoxPatternObject._geoms_from_init(
            self, 
            pattern,
        )

        # use the bottom geom of hole to determine offset for pattern
        hole_base_size = bounding_geom_args["geom_sizes"][-1]
        hole_base_loc = bounding_geom_args["geom_locations"][-1]

        for i in range(len(pattern_geom_args["geom_sizes"])):
            # convert to unnormalized coordinates, since this class
            # does not use normalized coordinates for specifying geoms
            pattern_geom_args["geom_sizes"][i][0] *= unit_size
            pattern_geom_args["geom_sizes"][i][1] *= unit_size
            pattern_geom_args["geom_sizes"][i][2] *= unit_size
            pattern_geom_args["geom_locations"][i][0] *= unit_size
            pattern_geom_args["geom_locations"][i][1] *= unit_size
            pattern_geom_args["geom_locations"][i][2] *= unit_size

            # move locations to account for the bounding box object
            pattern_geom_args["geom_locations"][i][0] += hole_base_loc[0]
            pattern_geom_args["geom_locations"][i][1] += hole_base_loc[1]
            pattern_geom_args["geom_locations"][i][2] += hole_base_loc[2] + hole_base_size[2]

        # add in dummy geom rgbas to merge with bounding geoms
        pattern_geom_args["geom_rgbas"] = [None for _ in range(len(pattern_geom_args["geom_sizes"]))]

        # merge geom lists together
        return {
            "geom_locations" : bounding_geom_args["geom_locations"] + pattern_geom_args["geom_locations"],
            "geom_sizes" : bounding_geom_args["geom_sizes"] + pattern_geom_args["geom_sizes"],
            "geom_names" : bounding_geom_args["geom_names"] + pattern_geom_args["geom_names"],
            "geom_rgbas" : bounding_geom_args["geom_rgbas"] + pattern_geom_args["geom_rgbas"],
        }


class BoxObject(MujocoGeneratedObject):
    """
    An object that is a box
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07, 0.07],
                         [0.03, 0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 3, "box size should have length 3"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2)

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="box")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="box")


class CylinderObject(MujocoGeneratedObject):
    """
    A randomized cylinder object.
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07],
                         [0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 2, "cylinder size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[1]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[1]])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="cylinder")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="cylinder")


class BallObject(MujocoGeneratedObject):
    """
    A randomized ball (sphere) object.
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07],
                         [0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 1, "ball size should have length 1"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[0]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[0]])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="sphere")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="sphere")


class CapsuleObject(MujocoGeneratedObject):
    """
    A randomized capsule object.
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07],
                         [0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 2, "capsule size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * (self.size[0] + self.size[1])])

    def get_top_offset(self):
        return np.array([0, 0, (self.size[0] + self.size[1])])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="capsule")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="capsule")


### More Miscellaneous Objects ###


class AnimalObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.02,0.033)
        self.body_y = random.uniform(0.015,0.03)
        self.body_z = random.uniform(0.01,0.035)
        self.legs_x = random.uniform(0.005,0.01)
        self.legs_z = random.uniform(0.01,0.035)
        self.neck_x = random.uniform(0.005,0.01)
        self.neck_z = random.uniform(0.005,0.01)
        self.head_y = random.uniform(0.010,0.015)
        self.head_z = random.uniform(0.005,0.01)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-2*self.legs_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.neck_z+2*self.head_z])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+self.body_y**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #legs
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[0.9*self.body_x-self.legs_x, 0.9*self.body_y-self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[-0.9*self.body_x+self.legs_x, 0.9*self.body_y-self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[0.9*self.body_x-self.legs_x, -0.9*self.body_y+self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[-0.9*self.body_x+self.legs_x, -0.9*self.body_y+self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1),)
        )
        #neck
        main_body.append(
        new_geom(
            geom_type="box", size=[self.neck_x,self.neck_x,self.neck_z],pos=[self.body_x-self.neck_x, 0, self.neck_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="box", size=[self.head_y,self.neck_x*1.5,self.head_z],pos=[self.body_x-2*self.neck_x+self.head_y, 0, 2*self.neck_z+self.body_z+self.head_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class CarObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.02,0.031)
        self.body_y = random.uniform(0.015,0.03)
        self.body_z = random.uniform(0.01,self.body_x/2)
        self.wheels_r = random.uniform(self.body_x/4.0,self.body_x/3.0)
        self.wheels_z = random.uniform(0.002,0.004)
        self.top_x = random.uniform(0.008,0.9*self.body_x)
        self.top_y = random.uniform(0.007,0.9*self.body_y)
        self.top_z = random.uniform(0.004,0.9*self.body_z)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-self.wheels_r])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.top_z])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+(self.body_y+2*self.wheels_z)**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #wheels
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1),)
        )
        #top
        main_body.append(
        new_geom(
            geom_type="box", size=[self.top_x,self.top_y,self.top_z],pos=[0, 0, self.top_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )

        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class TrainObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.017,0.031)
        self.body_y = random.uniform(0.025,0.045)
        self.body_z = random.uniform(0.01,0.025)
        self.wheels_r = random.uniform(self.body_x/4.0,self.body_x/3.0)
        self.wheels_z = random.uniform(0.002,0.006)
        self.top_x = random.uniform(0.01,0.9*self.body_x)
        self.top_r = 0.99*self.body_x
        self.top_z = 0.99*self.body_y
        self.cabin_x = 0.99*self.body_x
        self.cabin_y = random.uniform(0.20,0.3)*self.body_y
        self.cabin_z = random.uniform(0.5,0.8)*self.top_r
        self.chimney_r = random.uniform(0.004,0.01)
        self.chimney_z = random.uniform(0.01,0.03)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-self.wheels_r])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.chimney_z+self.top_r])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+(self.body_y+2*self.wheels_z)**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #wheels
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1),)
        )
        #top
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.top_r,self.top_z],pos=[0, 0, self.body_z], group=1, zaxis="0 1 0",
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #cabin
        main_body.append(
        new_geom(
            geom_type="box", size=[self.cabin_x,self.cabin_y,self.cabin_z],pos=[0, -self.body_y+self.cabin_y, self.body_z+self.cabin_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #chimney
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.chimney_r,self.chimney_z],pos=[0, self.body_y*.5, self.body_z+self.top_r], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class BipedObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.02,0.031)
        self.body_y = random.uniform(0.017,0.022)
        self.body_z = random.uniform(0.015,0.03)
        self.legs_x = random.uniform(0.005,0.01)
        self.legs_z = random.uniform(0.005,self.body_z)
        self.hands_x = random.uniform(0.005,0.01)
        self.hands_z = random.uniform(0.01,0.3*self.legs_z)
        self.head_y = self.body_y
        self.head_z = random.uniform(0.01,0.02)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-2*self.legs_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+self.body_y**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #legs
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.body_y,self.legs_z],pos=[self.body_x-self.legs_x, 0, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.body_y,self.legs_z],pos=[-self.body_x+self.legs_x, 0, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )

        #hands
        main_body.append(
        new_geom(
            geom_type="box", size=[self.hands_x,2*self.body_y,self.hands_z],pos=[self.body_x+self.hands_x, self.body_y, -self.hands_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.hands_x,2*self.body_y,self.hands_z],pos=[-self.body_x-self.hands_x, self.body_y, -self.hands_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="box", size=[self.head_y,self.head_y,self.head_z],pos=[0, 0, self.body_z+self.head_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class DumbbellObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_r = random.uniform(0.009,0.013)
        self.body_z = random.uniform(0.015,0.025)
        self.head_r = random.uniform(1.6*self.body_r,2*self.body_r)
        self.head_z = random.uniform(0.005,0.01)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-2*self.head_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return self.body_z+self.head_z

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.body_r,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, -self.head_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, self.head_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )        

        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class HammerObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_r = random.uniform(0.009,0.013)
        self.body_z = random.uniform(0.027,0.037)
        self.head_r = random.uniform(1.6*self.body_r,3*self.body_r)
        self.head_z = random.uniform(1.5*self.body_r,2*self.body_r)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return self.body_r+self.head_r

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.body_r,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, 0.95*self.head_r+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1),zaxis='1 0 0')
        )
    

        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class GuitarObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_r = random.uniform(0.021,0.027)/1.7
        self.body_z = random.uniform(0.017,0.025)/1.4
        self.head_r = random.uniform(1.5,2)*self.body_r
        self.head_z = self.body_z
        self.arm_x = random.uniform(0.008,0.010)/2
        self.arm_y = random.uniform(1.2,1.6)*(self.body_r+self.head_r)
        self.arm_z = 0.007/2
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return self.body_r+self.head_r

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        color = np.append(np.random.uniform(size=3),1)
        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.body_r,self.body_z],pos=[0, self.head_r+0.5*self.body_r, 0], group=1,
             rgba=color)
        )
        #head
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, 0], group=1,
             rgba=color)
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r*0.5,self.head_z],pos=[0, 0, 0.001], group=1,
             rgba=[0,0,0,1])
        )
        #arm
        main_body.append(
        new_geom(
            geom_type="box", size=[self.arm_x,self.arm_y,self.arm_z],pos=[0, self.arm_y, self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


