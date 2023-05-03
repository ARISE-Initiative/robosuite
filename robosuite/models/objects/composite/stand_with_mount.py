import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import RED, CustomMaterial, add_to_dict


class StandWithMount(CompositeObject):
    """
    Generates a flat stand with a four-walled mount sticking out of the top.
    Args:
        name (str): Name of this object
        size (3-array): (x,y,z) full size of object
        mount_location (2-array): (x,y) location to place mount, relative to center of stand
        mount_width (float): How wide mount is (measured from outside of walls!)
        wall_thickness (float): How thick to make walls of mount
        initialize_on_side (bool): If True, will initialize this stand on its side (tipped over)
        add_hole_vis (bool): If True, adds a rim around the top of the walls, to help make the hole more visually distinctive
        friction (3-array or None): If specified, sets friction values for this object. None results in default values
        density (float): Density value to use for all geoms. Defaults to 1000
        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored
        rgba (4-array or None): If specified, sets rgba values for all geoms. None results in default values
    """

    def __init__(
        self,
        name,
        size=(0.3, 0.3, 0.15),
        mount_location=(0.0, 0.0),
        mount_width=0.05,
        wall_thickness=0.01,
        base_thickness=0.01,
        initialize_on_side=True,
        add_hole_vis=False,
        friction=None,
        density=1000.0,
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001),
        use_texture=True,
        rgba=(0.2, 0.1, 0.0, 1.0),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.size = np.array(size)
        self.mount_location = np.array(mount_location)
        self.mount_width = mount_width
        self.wall_thickness = wall_thickness
        self.base_thickness = base_thickness
        self.initialize_on_side = initialize_on_side
        self.add_hole_vis = add_hole_vis
        self.friction = friction if friction is None else np.array(friction)
        self.solref = solref
        self.solimp = solimp
        self.density = density
        self.use_texture = use_texture
        self.rgba = rgba
        self.mat_name = "brass_mat"

        # Element references
        self._base_geom = "base"

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        bin_mat = CustomMaterial(
            texture="Brass",
            tex_name="brass",
            mat_name=self.mat_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(bin_mat)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor
        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": self.size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
            "solref": self.solref,
            "solimp": self.solimp,
        }
        obj_args = {}

        # Base
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, -(self.size[2] - self.base_thickness) / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array((self.size[0], self.size[1], self.base_thickness)) / 2,
            geom_names=self._base_geom,
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )

        # Walls
        x_vals = (
            np.array(
                [0, -(self.mount_width - self.wall_thickness) / 2, 0, (self.mount_width - self.wall_thickness) / 2]
            )
            + self.mount_location[0]
        )
        y_vals = (
            np.array(
                [-(self.mount_width - self.wall_thickness) / 2, 0, (self.mount_width - self.wall_thickness) / 2, 0]
            )
            + self.mount_location[1]
        )
        r_vals = np.array([np.pi / 2, 0, -np.pi / 2, np.pi])
        for i, (x, y, r) in enumerate(zip(x_vals, y_vals, r_vals)):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(x, y, self.base_thickness / 2),
                geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, r])), to="wxyz"),
                geom_sizes=(self.wall_thickness / 2, self.mount_width / 2, (self.size[2] - self.base_thickness) / 2),
                geom_names=f"wall{i}",
                geom_rgbas=None if self.use_texture else self.rgba,
                geom_materials=self.mat_name if self.use_texture else None,
                geom_frictions=self.friction,
            )

        if self.add_hole_vis:
            # add a purely visual rim
            del base_args["obj_types"]
            obj_args["obj_types"] = len(obj_args["geom_types"]) * ["all"]

            vis_geom_side = 0.7 * ((self.mount_width - self.wall_thickness) / 2)
            vis_geom_size = (vis_geom_side, vis_geom_side, self.wall_thickness / 2)
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(self.mount_location[0], self.mount_location[1], (self.size[2] / 2) - vis_geom_size[2]),
                geom_quats=(1, 0, 0, 0),
                geom_sizes=vis_geom_size,
                geom_names="hole_vis",
                geom_rgbas=(0.0, 1.0, 0.0, 0.5),
                geom_materials=None,
                geom_frictions=self.friction,
                obj_types="visual",
            )

        # Sites
        obj_args["sites"] = [
            {
                "name": f"mount_site",
                "pos": (0, 0, self.size[2] / 2),
                "size": "0.002",
                "rgba": RED,
                "type": "sphere",
            }
        ]

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args

    @property
    def init_quat(self):
        """
        Optionally rotate the mount on its side so it is flat
        Returns:
            np.array: (x, y, z, w) quaternion orientation for this object
        """
        # Rotate 90 deg about Y axis if at all
        return np.array([0, 0.707107, 0, 0.707107]) if self.initialize_on_side else np.array([0, 0, 0, 1])

    @property
    def base_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to base
        """
        return [self.correct_naming(self._base_geom)]
