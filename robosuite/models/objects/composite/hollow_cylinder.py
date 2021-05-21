import numpy as np

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict, CustomMaterial, RED
import robosuite.utils.transform_utils as T


class HollowCylinderObject(CompositeObject):
    """
    Generates an approximate hollow cylinder object by using box geoms.

    Args:
        name (str): Name of this HollowCylinder object

        outer_radius (float): Outer radius of hollow cylinder

        inner_radius (float): Inner radius of hollow cylinder

        height (float): Height of hollow cylinder

        ngeoms (int): Number of box geoms used to approximate the cylindrical shell. Use
            more geoms to make the approximation better.

        make_half (bool): If true, only make half of the shell.
    """

    def __init__(
        self,
        name,
        outer_radius=0.0425,
        inner_radius=0.03,
        height=0.05,
        ngeoms=8,
        rgba=None,
        material=None,
        density=1000.,
        friction=None,
        make_half=False,
    ):

        # Set object attributes
        self._name = name
        self.rgba = rgba
        self.density = density
        self.friction = friction if friction is None else np.array(friction)
        self.make_half = make_half # if True, will only make half the hollow cylinder

        # Other private attributes
        self._important_sites = {}

        # radius of the inner cup hole and entire cup
        self.r1 = inner_radius
        self.r2 = outer_radius

        # number of geoms used to approximate the cylindrical shell
        self.n = ngeoms

        # cylinder half-height
        self.height = height

        # half-width of each box inferred from triangle of radius + box half-length
        # since the angle will be (360 / n) / 2 
        self.unit_box_width = self.r2 * np.sin(np.pi / self.n)

        # half-height of each box inferred from the same triangle with inner radius
        self.unit_box_height = (self.r2 - self.r1) * np.cos(np.pi / self.n) / 2.

        # each box geom depth will end up defining the height of the cup
        self.unit_box_depth = self.height

        # radius of intermediate circle that connects all box centers
        self.int_r = (self.r1 * np.cos(np.pi / self.n)) + self.unit_box_height 

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Optionally add material
        self.has_material = (material is not None)
        if self.has_material:
            assert isinstance(material, CustomMaterial)
            self.material = material
            self.append_material(material)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": [self.r2, self.r2, self.height],
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
        }
        obj_args = {}

        n_make = self.n
        if self.make_half:
            # only make half the shell
            n_make = (self.n // 2) + 1

        # infer locations of all geoms with trigonometry
        angle_step = 2. * np.pi / self.n
        for i in range(n_make):
            # we start with the top-most box object and proceed clockwise (thus an offset of np.pi)
            geom_angle = np.pi - i * angle_step
            geom_center = np.array([
                self.int_r * np.cos(geom_angle),
                self.int_r * np.sin(geom_angle),
                0.
            ])
            geom_quat = np.array([np.cos(geom_angle / 2.), 0., 0., np.sin(geom_angle / 2.)])
            geom_size = np.array([self.unit_box_height, self.unit_box_width, self.unit_box_depth])

            add_to_dict(
                dic=obj_args,
                geom_types="box",
                # needle geom needs to be offset from boundary in (x, z)
                geom_locations=tuple(geom_center),
                geom_quats=tuple(geom_quat),
                geom_sizes=tuple(geom_size),
                geom_names="hc_{}".format(i),
                geom_rgbas=None if self.has_material else self.rgba,
                geom_materials=self.material.mat_attrib["name"] if self.has_material else None,
                 # make the needle low friction to ensure easy insertion
                geom_frictions=self.friction,
            )

        # Sites
        obj_args["sites"] = [
            {
                "name": "center",
                "pos": (0, 0, 0),
                "size": "0.002",
                "rgba": RED,
                "type": "sphere",
            }
        ]

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args
