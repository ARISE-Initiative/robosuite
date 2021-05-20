import numpy as np

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict
from robosuite.utils.mjcf_utils import CustomMaterial, RED
import robosuite.utils.transform_utils as T


class PictureFrame(CompositeObject):
    """
    Generates a (very janky) picture frame object.

    Args:
        name (str): Name of this object

        frame_size (2-array): (W, H) of frame

        frame_thickness (float): Thickness of the frame

        border_size (2-array): (W, H) of frame border that defines the picture frame. Must be smaller than @frame_size / 2

        border_thickness (float): Thickness of the frame border. Must be smaller than @frame_thickness

        mount_hole_offset (float): How high above the frame the mount hole will be placed

        mount_hole_size (float): How wide the mount hole is (internally)

        mount_hole_thickness (float): How thick the material will be for the mount hole frame

        friction (3-array or None): If specified, sets friction values for this object. None results in default values

        density (float): Density value to use for all geoms. Defaults to 1000

        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored

        rgba (4-array or None): If specified, sets rgba values for all geoms. None results in default values
    """

    def __init__(
        self,
        name,
        frame_size=(0.10, 0.15),
        frame_thickness=0.02,
        border_size=(0.02, 0.02),
        border_thickness=0.01,
        mount_hole_offset=0.05,
        mount_hole_size=0.03,
        mount_hole_thickness=0.01,
        friction=None,
        density=1000.,
        use_texture=True,
        rgba=(0.2, 0.1, 0.0, 1.0),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.frame_size = np.array(frame_size)
        self.frame_thickness = frame_thickness
        self.border_size = np.array(border_size)
        self.border_thickness = border_thickness
        assert np.all([self.border_size[i] < self.frame_size[i] / 2. for i in range(2)]), "border size must be smaller than frame size"
        assert self.border_thickness < self.frame_thickness, "border thickness must be smaller than frame thickness"
        self.mount_hole_offset = mount_hole_offset
        self.mount_hole_size = mount_hole_size
        self.mount_hole_thickness = mount_hole_thickness
        self.friction = friction if friction is None else np.array(friction)
        self.density = density
        self.use_texture = use_texture
        self.rgba = rgba
        self.mount_hole_mat_name = "steel_mat"
        self.frame_mat_name = "darkwood_mat"
        self.picture_mat_name = "lightwood_mat"

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
        frame_mat = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name=self.frame_mat_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        mount_hole_mat = CustomMaterial(
            texture="SteelScratched",
            tex_name="steel",
            mat_name=self.mount_hole_mat_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        picture_mat = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name=self.picture_mat_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(mount_hole_mat)
        self.append_material(frame_mat)
        self.append_material(picture_mat)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        mount_center_to_edge = self.mount_hole_size / 2 + self.mount_hole_thickness
        total_size = np.array((
                self.frame_thickness,
                self.frame_size[0],
                self.frame_size[1] + self.mount_hole_offset + mount_center_to_edge
            ))
        base_args = {
            "total_size": total_size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
        }
        obj_args = {}

        # Main Frame
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            # geom_locations=(0, 0, (-total_size[2] + self.frame_size[1]) / 2),
            geom_locations=(-self.border_thickness / 2., 0, (-total_size[2] + self.frame_size[1]) / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array(
                (self.frame_thickness - self.border_thickness,
                 self.frame_size[0] - self.border_size[0],
                 self.frame_size[1] - self.border_size[1])
            ) / 2,
            geom_names="picture_frame",
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.picture_mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )

        # Border Geoms
        border_offset = (self.frame_size - self.border_size) / 2.
        border_offsets = [
            ((self.frame_size[0] - self.border_size[0]) / 2., 0),
            (-(self.frame_size[0] - self.border_size[0]) / 2., 0),
            (0, (self.frame_size[1] - self.border_size[1]) / 2.),
            (0, -(self.frame_size[1] - self.border_size[1]) / 2.),
        ]
        border_sizes = [
            (self.border_size[0], self.frame_size[1]),
            (self.border_size[0], self.frame_size[1]),
            (self.frame_size[0], self.border_size[1]),
            (self.frame_size[0], self.border_size[1]),
        ]
        for i in range(4):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(0, border_offsets[i][0], ((-total_size[2] + self.frame_size[1]) / 2) + border_offsets[i][1]),
                geom_quats=(1, 0, 0, 0),
                geom_sizes=np.array((self.frame_thickness, border_sizes[i][0], border_sizes[i][1])) / 2,
                geom_names=f"border{i}",
                geom_rgbas=None if self.use_texture else self.rgba,
                geom_materials=self.frame_mat_name if self.use_texture else None,
                geom_frictions=self.friction,
            )

        # Mount Hole Connector
        connector_height = self.mount_hole_offset - mount_center_to_edge
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, total_size[2] / 2 - 2 * mount_center_to_edge - connector_height / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array((self.mount_hole_thickness, self.mount_hole_thickness, connector_height)) / 2,
            geom_names="mount_hole_connector",
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.mount_hole_mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )

        # Mount Hole Frames
        offset = (self.mount_hole_size + self.mount_hole_thickness) / 2
        z_offset = total_size[2] / 2 - mount_center_to_edge
        y_vals = np.array([0, -offset, 0, offset])
        z_vals = np.array([-offset, 0, offset, 0]) + z_offset
        r_vals = np.array([0, -np.pi / 2, np.pi, np.pi / 2])
        for i, (y, z, r) in enumerate(zip(y_vals, z_vals, r_vals)):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(0, y, z),
                geom_quats=T.convert_quat(T.axisangle2quat(np.array([r, 0, 0])), to="wxyz"),
                geom_sizes=np.array((self.mount_hole_thickness / 2, mount_center_to_edge, self.mount_hole_thickness / 2)),
                geom_names=f"mount_hole{i}",
                geom_rgbas=None if self.use_texture else self.rgba,
                geom_materials=self.mount_hole_mat_name if self.use_texture else None,
                geom_frictions=self.friction,
            )

        # Sites
        obj_args["sites"] = [
            {
                "name": f"mount_hole_site",
                "pos": (0, 0, z_offset),
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
        Rotate the object on its side so it is flat

        Returns:
            np.array: (x, y, z, w) quaternion orientation for this object
        """
        # Rotate 90 deg about Y axis
        return np.array([0, -0.707107, 0, 0.707107])

