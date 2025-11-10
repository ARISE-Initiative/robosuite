import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import BLUE, GREEN, RED, CustomMaterial, add_to_dict


class HookFrame(CompositeObject):
    """
    Generates an upside down L-shaped frame (a "hook" shape), intended to be used with StandWithMount object.
    Args:
        name (str): Name of this object
        frame_length (float): How long the frame is
        frame_height (float): How tall the frame is
        frame_thickness (float): How thick the frame is
        hook_height (float): if not None, add a box geom at the edge of the hook with this height (not half-height)
        grip_location (float): if not None, adds a grip to passed location, relative to center of the rod corresponding to @frame_height.
        grip_size ([float]): (R, H) radius and half-height for the cylindrical grip. Set to None
            to not add a grip.
        tip_size ([float]): if not None, adds a cone tip to the end of the hook for easier insertion, with the
            provided (CH, LR, UR, H) where CH is the base cylinder height, LR and UR are the lower and upper radius
            of the cone tip, and H is the half-height of the cone tip
        friction (3-array or None): If specified, sets friction values for this object. None results in default values
        density (float): Density value to use for all geoms. Defaults to 1000
        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored
        rgba (4-array or None): If specified, sets rgba values for all geoms. None results in default values
    """

    def __init__(
        self,
        name,
        frame_length=0.3,
        frame_height=0.2,
        frame_thickness=0.025,
        hook_height=None,
        grip_location=None,
        grip_size=None,
        tip_size=None,
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
        self.size = None  # Filled in automatically
        self.frame_length = frame_length
        self.frame_height = frame_height
        self.frame_thickness = frame_thickness
        self.hook_height = hook_height
        self.grip_location = grip_location
        self.grip_size = tuple(grip_size) if grip_size is not None else None
        self.tip_size = tuple(tip_size) if tip_size is not None else None
        self.friction = friction if friction is None else np.array(friction)
        self.solref = solref
        self.solimp = solimp
        self.density = density
        self.use_texture = use_texture
        self.rgba = rgba
        self.mat_name = "brass_mat"
        self.grip_mat_name = "ceramic_mat"
        self.tip_mat_name = "steel_mat"

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
        # optionally add material for grip
        if (self.grip_location is not None) and (self.grip_size is not None):
            grip_mat = CustomMaterial(
                texture="Ceramic",
                tex_name="ceramic",
                mat_name=self.grip_mat_name,
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            self.append_material(grip_mat)
        # optionally add material for tip
        if self.tip_size is not None:
            tip_mat = CustomMaterial(
                texture="SteelScratched",
                tex_name="steel",
                mat_name=self.tip_mat_name,
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            self.append_material(tip_mat)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor
        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        self.size = np.array((self.frame_length, self.frame_thickness, self.frame_height))
        if self.tip_size is not None:
            self.size[2] += 2.0 * (self.tip_size[0] + (2.0 * self.tip_size[3]))
        base_args = {
            "total_size": self.size / 2,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
            "solref": self.solref,
            "solimp": self.solimp,
        }
        obj_args = {}

        # Vertical Frame
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=((self.frame_length - self.frame_thickness) / 2, 0, -self.frame_thickness / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array((self.frame_thickness, self.frame_thickness, self.frame_height - self.frame_thickness))
            / 2,
            geom_names="vertical_frame",
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )

        # Horizontal Frame
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, (self.frame_height - self.frame_thickness) / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array((self.frame_length, self.frame_thickness, self.frame_thickness)) / 2,
            geom_names="horizontal_frame",
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )

        # optionally add hook at the end of the horizontal frame
        if self.hook_height is not None:
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(
                    (-self.frame_length + self.frame_thickness) / 2,
                    0,
                    (self.frame_height + self.hook_height) / 2,
                ),
                geom_quats=(1, 0, 0, 0),
                geom_sizes=np.array((self.frame_thickness, self.frame_thickness, self.hook_height)) / 2,
                geom_names="hook_frame",
                geom_rgbas=None if self.use_texture else self.rgba,
                geom_materials=self.mat_name if self.use_texture else None,
                geom_frictions=self.friction,
            )

        # optionally add a grip
        if (self.grip_location is not None) and (self.grip_size is not None):
            # note: use box grip instead of cylindrical grip for stability
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(
                    (self.frame_length - self.frame_thickness) / 2,
                    0,
                    (-self.frame_thickness / 2) + self.grip_location,
                ),
                geom_quats=(1, 0, 0, 0),
                geom_sizes=(self.grip_size[0], self.grip_size[0], self.grip_size[1]),
                geom_names="grip_frame",
                # geom_rgbas=None if self.use_texture else self.rgba,
                geom_rgbas=(0.13, 0.13, 0.13, 1.0),
                geom_materials=self.grip_mat_name if self.use_texture else None,
                # geom_frictions=self.friction,
                geom_frictions=(1.0, 0.005, 0.0001),  # use default friction
            )

        # optionally add cone tip
        if self.tip_size is not None:
            from robosuite.models.objects import ConeObject

            cone = ConeObject(
                name="cone",
                outer_radius=self.tip_size[2],
                inner_radius=self.tip_size[1],
                height=self.tip_size[3],
                # ngeoms=8,
                ngeoms=50,
                use_box=True,
                # use_box=False,
                rgba=None,
                material=None,
                density=self.density,
                solref=self.solref,
                solimp=self.solimp,
                friction=self.friction,
            )
            cone_args = cone._get_geom_attrs()

            # DIRTY HACK: add them in reverse (in hindsight, should just turn this into a composite body...)
            cone_geom_types = cone_args["geom_types"]
            cone_geom_locations = cone_args["geom_locations"]
            cone_geom_sizes = cone_args["geom_sizes"][::-1]

            # location of mount site is the translation we need
            cylinder_offset = (
                (self.frame_length - self.frame_thickness) / 2,
                0,
                -self.frame_height / 2 - self.tip_size[0],  # account for half-height of cylinder
            )
            cone_offset = (
                cylinder_offset[0],
                cylinder_offset[1],
                cylinder_offset[2]
                - self.tip_size[0]
                - self.tip_size[3] / 2.0,  # need to move below cylinder, and account for half-height
            )

            # first add cylinder
            add_to_dict(
                dic=obj_args,
                geom_types="cylinder",
                geom_locations=cylinder_offset,
                geom_quats=(1, 0, 0, 0),
                geom_sizes=(self.tip_size[2], self.tip_size[0]),
                geom_names="tip_cylinder",
                geom_rgbas=None if self.use_texture else self.rgba,
                geom_materials=self.tip_mat_name if self.use_texture else None,
                geom_frictions=self.friction,
            )

            # then add cone tip geoms
            for i in range(len(cone_geom_types)):
                add_to_dict(
                    dic=obj_args,
                    geom_types=cone_geom_types[i],
                    geom_locations=(
                        cone_geom_locations[i][0] + cone_offset[0],
                        cone_geom_locations[i][1] + cone_offset[1],
                        cone_geom_locations[i][2] + cone_offset[2],
                    ),
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=cone_geom_sizes[i],
                    geom_names="tip_cone_{}".format(i),
                    geom_rgbas=None if self.use_texture else self.rgba,
                    geom_materials=self.tip_mat_name if self.use_texture else None,
                    geom_frictions=self.friction,
                )

        # Sites
        obj_args["sites"] = [
            {
                "name": f"hang_site",
                "pos": (-self.frame_length / 2, 0, (self.frame_height - self.frame_thickness) / 2),
                "size": "0.002",
                "rgba": RED,
                "type": "sphere",
            },
            {
                "name": f"mount_site",
                "pos": ((self.frame_length - self.frame_thickness) / 2, 0, -self.frame_height / 2),
                "size": "0.002",
                "rgba": GREEN,
                "type": "sphere",
            },
            {
                "name": f"intersection_site",
                "pos": (
                    (self.frame_length - self.frame_thickness) / 2,
                    0,
                    (self.frame_height - self.frame_thickness) / 2,
                ),
                "size": "0.002",
                "rgba": BLUE,
                "type": "sphere",
            },
        ]

        if self.tip_size is not None:
            obj_args["sites"].append(
                {
                    "name": f"tip_site",
                    "pos": (
                        ((self.frame_length - self.frame_thickness) / 2),
                        0,
                        (-self.frame_height / 2) - 2.0 * self.tip_size[0] - self.tip_size[3],
                    ),
                    "size": "0.002",
                    "rgba": RED,
                    "type": "sphere",
                },
            )

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args

    @property
    def init_quat(self):
        """
        Rotate the frame on its side so it is flat
        Returns:
            np.array: (x, y, z, w) quaternion orientation for this object
        """
        # Rotate 90 degrees about two consecutive axes to make the hook lie on the table instead of being upright.
        return T.quat_multiply(
            np.array([0, 0.0, np.sqrt(2) / 2.0, np.sqrt(2) / 2.0]),
            np.array([-np.sqrt(2) / 2.0, 0.0, 0.0, np.sqrt(2) / 2.0]),
        )
