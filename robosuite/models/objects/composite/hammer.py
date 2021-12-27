from collections.abc import Iterable

import numpy as np

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import BLUE, CYAN, GREEN, RED, CustomMaterial, add_to_dict


class HammerObject(CompositeObject):
    """
    Generates a Hammer object with a cylindrical or box-shaped handle, cubic head, cylindrical face and triangular claw
    (used in Handover task)

    Args:
        name (str): Name of this Hammer object

        handle_shape (str): Either "box", for a box-shaped handle, or "cylinder", for a cylindrically-shaped handle

        handle_radius (float or 2-array of float): Either specific or range of values to draw randomly from
            uniformly for the handle radius

        handle_length (float or 2-array of float): Either specific or range of values to draw randomly from
            uniformly for the handle length

        handle_density (float or 2-array of float): Either specific or range of values to draw randomly from
            uniformly for the handle density (in SI units). Note that this value is scaled x4 for the hammer head

        handle_friction (float or 2-array of float): Either specific or range of values to draw randomly from
            uniformly for the handle friction. Note that Mujoco default values are used for the head

        head_density_ratio (float): Ratio of density of handle to head (including face and claw)

        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored

        rgba_handle (4-array or None): If specified, sets handle rgba values

        rgba_head (4-array or None): If specified, sets handle rgba values

        rgba_face (4-array or None): If specified, sets handle rgba values

        rgba_claw (4-array or None): If specified, sets handle rgba values

    Raises:
        ValueError: [Invalid handle shape]
    """

    def __init__(
        self,
        name,
        handle_shape="box",
        handle_radius=(0.015, 0.02),
        handle_length=(0.1, 0.25),
        handle_density=(100, 250),
        handle_friction=(3.0, 5.0),
        head_density_ratio=2.0,
        use_texture=True,
        rgba_handle=None,
        rgba_head=None,
        rgba_face=None,
        rgba_claw=None,
    ):
        # Set name
        self._name = name

        # Set handle type and density ratio
        self.handle_shape = handle_shape
        self.head_density_ratio = head_density_ratio

        # Set radius and length ranges
        self.handle_radius_range = handle_radius if isinstance(handle_radius, Iterable) else [handle_radius] * 2
        self.handle_length_range = handle_length if isinstance(handle_length, Iterable) else [handle_length] * 2
        self.handle_density_range = handle_density if isinstance(handle_density, Iterable) else [handle_density] * 2
        self.handle_friction_range = handle_friction if isinstance(handle_friction, Iterable) else [handle_friction] * 2

        # Sample actual radius and length, as well as head half-size
        self.handle_radius = np.random.uniform(self.handle_radius_range[0], self.handle_radius_range[1])
        self.handle_length = np.random.uniform(self.handle_length_range[0], self.handle_length_range[1])
        self.handle_density = np.random.uniform(self.handle_density_range[0], self.handle_density_range[1])
        self.handle_friction = np.random.uniform(self.handle_friction_range[0], self.handle_friction_range[1])
        self.head_halfsize = np.random.uniform(self.handle_radius, self.handle_radius * 1.2)

        # Initialize RGBA values and texture flag
        self.use_texture = use_texture
        self.rgba_handle = rgba_handle if rgba_handle is not None else RED
        self.rgba_head = rgba_head if rgba_head is not None else CYAN
        self.rgba_face = rgba_face if rgba_face is not None else BLUE
        self.rgba_claw = rgba_claw if rgba_claw is not None else GREEN

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
        metal = CustomMaterial(
            texture="SteelScratched",
            tex_name="metal",
            mat_name="metal_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        wood = CustomMaterial(
            texture="WoodLight",
            tex_name="wood",
            mat_name="wood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # Append materials to object
        self.append_material(metal)
        self.append_material(wood)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        full_size = np.array(
            (3.2 * self.head_halfsize, self.head_halfsize, self.handle_length + 2 * self.head_halfsize)
        )
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": full_size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        obj_args = {}

        # Add handle component
        assert self.handle_shape in {
            "cylinder",
            "box",
        }, "Error loading hammer: Handle type must either be 'box' or 'cylinder', got {}.".format(self.handle_shape)
        add_to_dict(
            dic=obj_args,
            geom_types="cylinder" if self.handle_shape == "cylinder" else "box",
            geom_locations=(0, 0, 0),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array([self.handle_radius, self.handle_length / 2.0])
            if self.handle_shape == "cylinder"
            else np.array([self.handle_radius, self.handle_radius, self.handle_length / 2.0]),
            geom_names="handle",
            geom_rgbas=None if self.use_texture else self.rgba_handle,
            geom_materials="wood_mat" if self.use_texture else None,
            geom_frictions=(self.handle_friction, 0.005, 0.0001),
            density=self.handle_density,
        )

        # Add head component
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, self.handle_length / 2.0 + self.head_halfsize),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array([self.head_halfsize * 2, self.head_halfsize, self.head_halfsize]),
            geom_names="head",
            geom_rgbas=None if self.use_texture else self.rgba_head,
            geom_materials="metal_mat" if self.use_texture else None,
            geom_frictions=None,
            density=self.handle_density * self.head_density_ratio,
        )

        # Add neck component
        add_to_dict(
            dic=obj_args,
            geom_types="cylinder",
            geom_locations=(self.head_halfsize * 2.2, 0, self.handle_length / 2.0 + self.head_halfsize),
            geom_quats=(0.707106, 0, 0.707106, 0),
            geom_sizes=np.array([self.head_halfsize * 0.8, self.head_halfsize * 0.2]),
            geom_names="neck",
            geom_rgbas=None if self.use_texture else self.rgba_face,
            geom_materials="metal_mat" if self.use_texture else None,
            geom_frictions=None,
            density=self.handle_density * self.head_density_ratio,
        )

        # Add face component
        add_to_dict(
            dic=obj_args,
            geom_types="cylinder",
            geom_locations=(self.head_halfsize * 2.8, 0, self.handle_length / 2.0 + self.head_halfsize),
            geom_quats=(0.707106, 0, 0.707106, 0),
            geom_sizes=np.array([self.head_halfsize, self.head_halfsize * 0.4]),
            geom_names="face",
            geom_rgbas=None if self.use_texture else self.rgba_face,
            geom_materials="metal_mat" if self.use_texture else None,
            geom_frictions=None,
            density=self.handle_density * self.head_density_ratio,
        )

        # Add claw component
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(-self.head_halfsize * 2, 0, self.handle_length / 2.0 + self.head_halfsize),
            geom_quats=(0.9238795, 0, 0.3826834, 0),
            geom_sizes=np.array([self.head_halfsize * 0.7072, self.head_halfsize * 0.95, self.head_halfsize * 0.7072]),
            geom_names="claw",
            geom_rgbas=None if self.use_texture else self.rgba_claw,
            geom_materials="metal_mat" if self.use_texture else None,
            geom_frictions=None,
            density=self.handle_density * self.head_density_ratio,
        )

        # Add back in base args
        obj_args.update(base_args)

        # Return this dict
        return obj_args

    @property
    def init_quat(self):
        """
        Generates a new random orientation for the hammer

        Returns:
            np.array: (x, y, z, w) quaternion orientation for the hammer
        """
        # Randomly sample between +/- flip (such that the hammer head faces one way or the other)
        return np.array([0.5, -0.5, 0.5, -0.5]) if np.random.rand() >= 0.5 else np.array([-0.5, -0.5, -0.5, -0.5])

    @property
    def handle_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to hammer handle
        """
        return self.correct_naming(["handle"])

    @property
    def head_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to hammer head
        """
        return self.correct_naming(["head"])

    @property
    def face_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to hammer face
        """
        return self.correct_naming(["neck", "face"])

    @property
    def claw_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to hammer claw
        """
        return self.correct_naming(["claw"])

    @property
    def all_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to all hammer components
        """
        return self.handle_geoms + self.head_geoms + self.face_geoms + self.claw_geoms

    @property
    def bottom_offset(self):
        return np.array([0, 0, -self.handle_radius])

    @property
    def top_offset(self):
        return np.array([0, 0, self.handle_radius])

    @property
    def horizontal_radius(self):
        return self.head_halfsize + 0.5 * self.handle_length
