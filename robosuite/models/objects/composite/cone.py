import numpy as np

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict, CustomMaterial, RED
import robosuite.utils.transform_utils as T


class ConeObject(CompositeObject):
    """
    Generates an approximate cone object by using cylinder or box geoms.

    Args:
        name (str): Name of this Cone object

        outer_radius (float): Radius of cone base

        inner_radius (float): Radius of cone tip (since everything is a cylinder or box)

        height (float): Height of cone

        ngeoms (int): Number of cylinder or box geoms used to approximate the cone. Use
            more geoms to make the approximation better.

        use_box (bool): If true, use box geoms instead of cylinders, corresponding to a 
            square pyramid shape instead of a conical shape.
    """

    def __init__(
        self,
        name,
        outer_radius=0.0425,
        inner_radius=0.03,
        height=0.05,
        ngeoms=8,
        use_box=False,
        rgba=None,
        material=None,
        density=1000.,
        solref=(0.02, 1.),
        solimp=(0.9, 0.95, 0.001),
        friction=None,
    ):

        # Set object attributes
        self._name = name
        self.rgba = rgba
        self.density = density
        self.friction = friction if friction is None else np.array(friction)
        self.solref = solref
        self.solimp = solimp

        self.has_material = (material is not None)
        if self.has_material:
            assert isinstance(material, CustomMaterial)
            self.material = material

        # Other private attributes
        self._important_sites = {}

        # radius of the tip and the base
        self.r1 = inner_radius
        self.r2 = outer_radius

        # number of geoms used to approximate the cone
        if ngeoms % 2 == 0:
            # use an odd number of geoms for easier computation
            ngeoms += 1
        self.n = ngeoms

        # cone height
        self.height = height

        # unit half-height for geoms
        self.unit_height = (height / ngeoms) / 2.

        # unit radius for geom radius grid
        self.unit_r = (self.r2 - self.r1) / (self.n - 1)

        self.use_box = use_box

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Optionally add material
        if self.has_material:
            self.append_material(self.material)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": [self.r2, self.r2, self.height / 2.],
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
            "solref": self.solref,
            "solimp": self.solimp,
        }
        obj_args = {}

        # stack the boxes / cylinders in the z-direction
        ngeoms_each_side = (self.n - 1) // 2
        geom_locations = [(0., 0., i * self.unit_height * 2.) for i in range(-ngeoms_each_side, ngeoms_each_side + 1)]

        if self.use_box:
            geom_sizes = [(
                self.r1 + i * self.unit_r, 
                self.r1 + i * self.unit_r,
                self.unit_height,
            ) for i in range(self.n)][::-1]
        else:
            geom_sizes = [(
                self.r1 + i * self.unit_r, 
                self.unit_height,
            ) for i in range(self.n)][::-1]

        for i in range(self.n):
            # note: set geom condim to 4 for consistency with round-nut.xml
            # geom_quat = np.array([np.cos(geom_angle / 2.), 0., 0., np.sin(geom_angle / 2.)])
            add_to_dict(
                dic=obj_args,
                geom_types="box" if self.use_box else "cylinder",
                geom_locations=geom_locations[i],
                geom_quats=None,
                geom_sizes=geom_sizes[i],
                geom_names="c_{}".format(i),
                # geom_rgbas=None if self.has_material else self.rgba,
                geom_rgbas=self.rgba,
                geom_materials=self.material.mat_attrib["name"] if self.has_material else None,
                geom_frictions=self.friction,
                geom_condims=4,
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
