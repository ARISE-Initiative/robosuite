import numpy as np

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict
from robosuite.utils.mjcf_utils import CustomMaterial, RED, GREEN, BLUE
import robosuite.utils.transform_utils as T


class HookFrame(CompositeObject):
    """
    Generates an upside down L-shaped frame (a "hook" shape), intended to be used with StandWithMount object.

    Args:
        name (str): Name of this object

        frame_length (float): How long the frame is

        frame_height (float): How tall the frame is

        frame_thickness (float): How thick the frame is

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
        friction=None,
        density=1000.,
        use_texture=True,
        rgba=(0.2, 0.1, 0.0, 1.0),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.size = None                      # Filled in automatically
        self.frame_length = frame_length
        self.frame_height = frame_height
        self.frame_thickness = frame_thickness
        self.friction = friction if friction is None else np.array(friction)
        self.density = density
        self.use_texture = use_texture
        self.rgba = rgba
        self.mat_name = "brass_mat"

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
        self.size = np.array((self.frame_length, self.frame_thickness, self.frame_height))
        base_args = {
            "total_size": self.size / 2,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
        }
        obj_args = {}

        # Vertical Frame
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=((self.frame_length - self.frame_thickness) / 2, 0, -self.frame_thickness / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array((self.frame_thickness, self.frame_thickness, self.frame_height - self.frame_thickness)) / 2,
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

        # Sites
        obj_args["sites"] = [
            {
                "name": f"hang_site",
                "pos": (-self.size[0] / 2, 0, (self.size[2] - self.frame_thickness) / 2),
                "size": "0.002",
                "rgba": RED,
                "type": "sphere",
            },
            {
                "name": f"mount_site",
                "pos": ((self.size[0] - self.frame_thickness) / 2, 0, -self.size[2] / 2),
                "size": "0.002",
                "rgba": GREEN,
                "type": "sphere",
            },
            {
                "name": f"intersection_site",
                "pos": ((self.size[0] - self.frame_thickness) / 2, 0, (self.size[2] - self.frame_thickness) / 2),
                "size": "0.002",
                "rgba": BLUE,
                "type": "sphere",
            },
        ]

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
                np.array([0, 0., np.sqrt(2) / 2., np.sqrt(2) / 2.]),
                np.array([-np.sqrt(2) / 2., 0., 0., np.sqrt(2) / 2.]),
            )

