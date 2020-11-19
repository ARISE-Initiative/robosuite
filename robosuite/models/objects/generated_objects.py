import numpy as np

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site, array_to_string, add_to_dict
from robosuite.utils.mjcf_utils import RED, GREEN, BLUE, CYAN, OBJECT_COLLISION_COLOR, CustomMaterial
import robosuite.utils.transform_utils as T

from collections.abc import Iterable

from copy import deepcopy
import xml.etree.ElementTree as ET


class CompositeObject(MujocoGeneratedObject):
    """
    An object constructed out of basic geoms to make more intricate shapes.

    Note that by default, specifying None for a specific geom element will usually set a value to the mujoco defaults.

    Args:
        total_size (list): (x, y, z) half-size in each dimension for the bounding box for
            this Composite object

        object_name (str): Name of overall object

        geom_types (list): list of geom types in the composite. Must correspond
            to MuJoCo geom primitives, such as "box" or "capsule".

        geom_locations (list): list of geom locations in the composite. Each
            location should be a list or tuple of 3 elements and all
            locations are relative to the lower left corner of the total box
            (e.g. (0, 0, 0) corresponds to this corner).

        geom_sizes (list): list of geom sizes ordered the same as @geom_locations

        geom_names (list): list of geom names ordered the same as @geom_locations. The
            names will get appended with an underscore to the passed name in @get_collision
            and @get_visual

        geom_rgbas (list): list of geom colors ordered the same as @geom_locations. If
            passed as an argument, @rgba is ignored.

        geom_materials (list of CustomTexture): list of custom textures to use for this object material

        geom_frictions (list): list of geom frictions to use for each geom.

        geom_quats (list): list of (w, x, y, z) quaternions for each geom.

        joints (list): list of joints to use for each geom. Note that each entry should be its own list of dictionaries,
            where each dictionary specifies the specific joint attributes necessary.
            See http://www.mujoco.org/book/XMLreference.html#joint for reference.

        rgba (list): (r, g, b, a) default values to use if geom-specific @geom_rgbas isn't specified for a given element

        density (float or list of float): either single value to use for all geom densities or geom-specific values

        solref (list or list of list): parameters used for the mujoco contact solver. Can be single set of values or
            element-specific values. See http://www.mujoco.org/book/modeling.html#CSolver for details.

        solimp (list or list of list): parameters used for the mujoco contact solver. Can be single set of values or
            element-specific values. See http://www.mujoco.org/book/modeling.html#CSolver for details.

        locations_relative_to_center (bool): If true, @geom_locations will be considered relative to the center of the
            overall object bounding box defined by @total_size. Else, the corner of this bounding box is considered the
            origin.

        sites (list of dict): list of sites to add to this object. Each dict should specify the appropriate attributes
            for the given site. See http://www.mujoco.org/book/XMLreference.html#site for reference.

        obj_types (str or list of str): either single obj_type for all geoms or geom-specific type. Choices are
            {"collision", "visual", "all"}
    """

    def __init__(
        self,
        total_size,
        object_name,
        geom_types,
        geom_locations,
        geom_sizes,
        geom_names=None,
        geom_rgbas=None,
        geom_materials=None,
        geom_frictions=None,
        geom_quats=None,
        joints="default",
        rgba=None,
        density=100.,
        solref=(0.02, 1.),
        solimp=(0.9, 0.95, 0.001),
        locations_relative_to_center=False,
        sites=None,
        obj_types="all",
        duplicate_collision_geoms=True,
    ):
        super().__init__(
            name=object_name,
            joints=joints,
            rgba=rgba,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )

        n_geoms = len(geom_types)
        self.total_size = np.array(total_size)
        self.geom_types = np.array(geom_types)
        self.geom_locations = np.array(geom_locations)
        self.geom_sizes = deepcopy(geom_sizes)
        self.geom_names = list(geom_names) if geom_names is not None else None
        self.geom_rgbas = list(geom_rgbas) if geom_rgbas is not None else None
        self.geom_materials = list(geom_materials) if geom_materials is not None else None
        self.geom_frictions = list(geom_frictions) if geom_frictions is not None else None
        self.density = [density] * n_geoms if density is None or type(density) in {float, int} else list(density)
        self.solref = [solref] * n_geoms if solref is None or type(solref[0]) in {float, int} else list(solref)
        self.solimp = [solimp] * n_geoms if obj_types is None or type(solimp[0]) in {float, int} else list(solimp)
        self.rgba = rgba        # override superclass setting of this variable
        self.locations_relative_to_center = locations_relative_to_center
        self.geom_quats = deepcopy(geom_quats) if geom_quats is not None else None
        self.sites = sites
        self.obj_types = [obj_types] * n_geoms if obj_types is None or type(obj_types) is str else list(obj_types)

    def get_bottom_offset(self):
        return np.array([0., 0., -self.total_size[2]])

    def get_top_offset(self):
        return np.array([0., 0., self.total_size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.total_size[:2], 2)

    def _size_to_cartesian_half_lengths(self, geom_type, geom_size):
        """
        converts from geom size specification to x, y, and z half-length bounding box
        """
        if geom_type in ['box', 'ellipsoid']:
            return geom_size
        if geom_type == 'sphere':
            # size is radius
            return [geom_size[0], geom_size[0], geom_size[0]]
        if geom_type == 'capsule':
            # size is radius, half-length of cylinder part
            return [geom_size[0], geom_size[0], geom_size[0] + geom_size[1]]
        if geom_type == 'cylinder':
            # size is radius, half-length
            return [geom_size[0], geom_size[0], geom_size[1]]
        raise Exception("unsupported geom type!")

    def get_object_subtree(self, site=None):
        obj = new_body()
        obj.set("name", self.name)

        for i in range(self.geom_locations.shape[0]):
            # geom type
            geom_type = self.geom_types[i]
            # get cartesian size from size spec
            size = self.geom_sizes[i]
            cartesian_size = self._size_to_cartesian_half_lengths(geom_type, size)
            if self.locations_relative_to_center:
                # no need to convert
                pos = self.geom_locations[i]
            else:
                # use geom location to convert to position coordinate (the origin is the
                # center of the composite object)
                loc = self.geom_locations[i]
                pos = [
                    (-self.total_size[0] + cartesian_size[0]) + loc[0],
                    (-self.total_size[1] + cartesian_size[1]) + loc[1],
                    (-self.total_size[2] + cartesian_size[2]) + loc[2],
                ]

            # geom name
            if self.geom_names is not None:
                geom_name = "{}_{}".format(self.name, self.geom_names[i])
            else:
                geom_name = "{}_{}".format(self.name, i)

            # geom rgba
            if self.geom_rgbas is not None and self.geom_rgbas[i] is not None:
                geom_rgba = self.geom_rgbas[i]
            else:
                geom_rgba = self.rgba

            # geom friction
            if self.geom_frictions is not None and self.geom_frictions[i] is not None:
                geom_friction = array_to_string(self.geom_frictions[i])
            else:
                geom_friction = array_to_string(np.array([1., 0.005, 0.0001])) # mujoco default

            # Define base geom attributes
            geom_attr = {
                "size": size,
                "pos": pos,
                "name": geom_name,
                "geom_type": geom_type,
            }

            # Optionally define quat if specified
            if self.geom_quats is not None:
                geom_attr['quat'] = array_to_string(self.geom_quats[i])

            # Add collision geom if necessary
            if self.obj_types[i] in {"collision", "all"}:
                col_geom_attr = deepcopy(geom_attr)
                col_geom_attr.update(self.get_collision_attrib_template())
                if self.density[i] is not None:
                    col_geom_attr['density'] = str(self.density[i])
                col_geom_attr['friction'] = geom_friction
                col_geom_attr['solref'] = array_to_string(self.solref[i])
                col_geom_attr['solimp'] = array_to_string(self.solimp[i])
                col_geom_attr['rgba'] = OBJECT_COLLISION_COLOR
                obj.append(new_geom(**col_geom_attr))

            # Add visual geom if necessary
            if self.obj_types[i] in {"visual", "all"}:
                vis_geom_attr = deepcopy(geom_attr)
                vis_geom_attr.update(self.get_visual_attrib_template())
                vis_geom_attr["name"] += "_vis"
                if self.geom_materials is not None:
                    vis_geom_attr['material'] = self.geom_materials[i]
                vis_geom_attr["rgba"] = geom_rgba
                obj.append(new_geom(**vis_geom_attr))

        # add top level site if requested
        if site:
            # add a site as well
            site_element_attr = self.get_site_attrib_template()
            site_element_attr["rgba"] = "1 0 0 0"
            site_element_attr["name"] = self.name
            obj.append(ET.Element("site", attrib=site_element_attr))

        # Also create specific sites requested
        if self.sites is not None:
            # add relevant sites
            for s in self.sites:
                obj.append(ET.Element("site", attrib=s))

        return obj

    def get_bounding_box_size(self):
        return np.array(self.total_size)

    def in_box(self, position, object_position):
        """
        Checks whether the object is contained within this CompositeObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the CompositeBoxObject as an axis-aligned grid.
        Args:
            position: 3D body position of CompositeObject
            object_position: 3D position of object to test for insertion
        """
        ub = position + self.total_size
        lb = position - self.total_size

        # fudge factor for the z-check, since after insertion the object falls to table
        lb[2] -= 0.01

        return np.all(object_position > lb) and np.all(object_position < ub)


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
        self.name = name

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

    def get_bottom_offset(self):
        return np.array([0, 0, -self.handle_radius])

    def get_top_offset(self):
        return np.array([0, 0, self.handle_radius])

    def get_horizontal_radius(self):
        return self.head_halfsize + 0.5 * self.handle_length

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        full_size = np.array((
            3.2 * self.head_halfsize,
            self.head_halfsize,
            self.handle_length + 2 * self.head_halfsize
        ))
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        obj_args = {
            "total_size": full_size / 2.0,
            "object_name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }

        # Add handle component
        assert self.handle_shape in {"cylinder", "box"},\
            "Error loading hammer: Handle type must either be 'box' or 'cylinder', got {}.".format(self.handle_shape)
        add_to_dict(
            dic=obj_args,
            geom_types="cylinder" if self.handle_shape == "cylinder" else "box",
            geom_locations=(0, 0, 0),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array([self.handle_radius, self.handle_length / 2.0]) if self.handle_shape == "cylinder" else\
                       np.array([self.handle_radius, self.handle_radius, self.handle_length / 2.0]),
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
        return ["hammer_handle"]

    @property
    def head_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to hammer head
        """
        return ["hammer_head"]

    @property
    def face_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to hammer face
        """
        return ["hammer_neck", "hammer_face"]

    @property
    def claw_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to hammer claw
        """
        return ["hammer_claw"]

    @property
    def all_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to all hammer components
        """
        return self.handle_geoms + self.head_geoms + self.face_geoms + self.claw_geoms


class PotWithHandlesObject(CompositeObject):
    """
    Generates the Pot object with side handles (used in TwoArmLift)

    Args:
        name (str): Name of this Pot object

        body_half_size (3-array of float): If specified, defines the (x,y,z) half-dimensions of the main pot
            body. Otherwise, defaults to [0.07, 0.07, 0.07]

        handle_radius (float): Determines the pot handle radius

        handle_length (float): Determines the pot handle length

        handle_width (float): Determines the pot handle width

        handle_friction (float): Friction value to use for pot handles. Defauls to 1.0

        density (float): Density value to use for all geoms. Defaults to 1000

        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored

        rgba_body (4-array or None): If specified, sets pot body rgba values

        rgba_handle_0 (4-array or None): If specified, sets handle 0 rgba values

        rgba_handle_1 (4-array or None): If specified, sets handle 1 rgba values

        solid_handle (bool): If true, uses a single geom to represent the handle

        thickness (float): How thick to make the pot body walls
    """

    def __init__(
        self,
        name,
        body_half_size=(0.07, 0.07, 0.07),
        handle_radius=0.01,
        handle_length=0.09,
        handle_width=0.09,
        handle_friction=1.0,
        density=1000,
        use_texture=True,
        rgba_body=None,
        rgba_handle_0=None,
        rgba_handle_1=None,
        solid_handle=False,
        thickness=0.01,  # For body
    ):
        # Set name
        self.name = name

        # Set object attributes
        self.body_half_size = np.array(body_half_size)
        self.thickness = thickness
        self.handle_radius = handle_radius
        self.handle_length = handle_length
        self.handle_width = handle_width
        self.handle_friction = handle_friction
        self.density = density
        self.use_texture = use_texture
        self.rgba_body = np.array(rgba_body) if rgba_body else RED
        self.rgba_handle_0 = np.array(rgba_handle_0) if rgba_handle_0 else GREEN
        self.rgba_handle_1 = np.array(rgba_handle_1) if rgba_handle_1 else BLUE
        self.solid_handle = solid_handle

        # Geoms to be filled when generated
        self.handle0_geoms = None
        self.handle1_geoms = None
        self.pot_base = None

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="pot_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="handle0_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(redwood)
        self.append_material(greenwood)
        self.append_material(bluewood)

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.body_half_size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.body_half_size[2]])

    def get_horizontal_radius(self):
        return np.sqrt(2) * (max(self.body_half_size) + self.handle_length)

    @property
    def handle_distance(self):

        """
        Calculates how far apart the handles are

        Returns:
            float: handle distance
        """
        return self.body_half_size[1] * 2 + self.handle_length * 2

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        full_size = np.array((
            self.body_half_size,
            self.body_half_size + self.handle_length * 2,
            self.body_half_size,
        ))
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        obj_args = {
            "total_size": full_size / 2.0,
            "object_name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "sites": [],
        }

        # Initialize geom lists
        self.handle0_geoms = []
        self.handle1_geoms = []

        # Add main pot body
        # Base geom
        name = f"base"
        self.pot_base = [f"{self.name}_{name}"]
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, -self.body_half_size[2] + self.thickness / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array([self.body_half_size[0], self.body_half_size[1], self.thickness / 2]),
            geom_names=name,
            geom_rgbas=None if self.use_texture else self.rgba_body,
            geom_materials="pot_mat" if self.use_texture else None,
            geom_frictions=None,
            density=self.density,
        )

        # Walls
        x_off = np.array([0, -(self.body_half_size[0] - self.thickness / 2),
                          0, self.body_half_size[0] - self.thickness / 2])
        y_off = np.array([-(self.body_half_size[1] - self.thickness / 2),
                          0, self.body_half_size[1] - self.thickness / 2, 0])
        w_vals = np.array([self.body_half_size[1], self.body_half_size[0],
                           self.body_half_size[1], self.body_half_size[0]])
        r_vals = np.array([np.pi / 2, 0, -np.pi / 2, np.pi])
        for i, (x, y, w, r) in enumerate(zip(x_off, y_off, w_vals, r_vals)):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(x, y, 0),
                geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, r])), to="wxyz"),
                geom_sizes=np.array([self.thickness / 2, w, self.body_half_size[2]]),
                geom_names=f"body{i}",
                geom_rgbas=None if self.use_texture else self.rgba_body,
                geom_materials="pot_mat" if self.use_texture else None,
                geom_frictions=None,
                density=self.density,
            )

        # Add handles
        main_bar_size = np.array([
            self.handle_width / 2 + self.handle_radius,
            self.handle_radius,
            self.handle_radius,
        ])
        side_bar_size = np.array([self.handle_radius, self.handle_length / 2, self.handle_radius])
        handle_z = self.body_half_size[2] - self.handle_radius
        for i, (handle_side, rgba) in enumerate(zip([1.0, -1.0], [self.rgba_handle_0, self.rgba_handle_1])):
            handle_center = np.array((0, handle_side * (self.body_half_size[1] + self.handle_length), handle_z))
            # Get reference to relevant geom list
            g_list = getattr(self, f"handle{i}_geoms")
            # Solid handle case
            name = f"handle{i}"
            g_list.append(f"{self.name}_{name}")
            if self.solid_handle:
                add_to_dict(
                    dic=obj_args,
                    geom_types="box",
                    geom_locations=handle_center,
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=np.array([self.handle_width / 2, self.handle_length / 2, self.handle_radius]),
                    geom_names=name,
                    geom_rgbas=None if self.use_texture else rgba,
                    geom_materials=f"handle{i}_mat" if self.use_texture else None,
                    geom_frictions=(self.handle_friction, 0.005, 0.0001),
                    density=self.density,
                )
            # Hollow handle case
            else:
                # Center bar
                name = f"handle{i}_c"
                g_list.append(f"{self.name}_{name}")
                add_to_dict(
                    dic=obj_args,
                    geom_types="box",
                    geom_locations=handle_center,
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=main_bar_size,
                    geom_names=name,
                    geom_rgbas=None if self.use_texture else rgba,
                    geom_materials=f"handle{i}_mat" if self.use_texture else None,
                    geom_frictions=(self.handle_friction, 0.005, 0.0001),
                    density=self.density,
                )
                # Side bars
                for bar_side, suffix in zip([-1., 1.], ["-", "+"]):
                    name = f"handle{i}_{suffix}"
                    g_list.append(f"{self.name}_{name}")
                    add_to_dict(
                        dic=obj_args,
                        geom_types="box",
                        geom_locations=(
                            bar_side * self.handle_width / 2,
                            handle_side * (self.body_half_size[1] + self.handle_length / 2),
                            handle_z
                        ),
                        geom_quats=(1, 0, 0, 0),
                        geom_sizes=side_bar_size,
                        geom_names=name,
                        geom_rgbas=None if self.use_texture else rgba,
                        geom_materials=f"handle{i}_mat" if self.use_texture else None,
                        geom_frictions=(self.handle_friction, 0.005, 0.0001),
                        density=self.density,
                    )
            # Add relevant site
            handle_site = self.get_site_attrib_template()
            handle_site.update({
                "name": f"{self.name}_handle{i}",
                "pos": array_to_string(handle_center - handle_side * np.array([0, 0.005, 0])),
                "size": "0.005",
                "rgba": rgba,
            })
            obj_args["sites"].append(handle_site)

        # Add pot body site
        pot_site = self.get_site_attrib_template()
        pot_site.update({
            "name": f"{self.name}_center",
            "size": "0.005",
        })
        obj_args["sites"].append(pot_site)

        # Return this dict
        return obj_args

    @property
    def handle_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to both handles
        """
        return self.handle0_geoms() + self.handle1_geoms()


def _get_size(size,
              size_max,
              size_min,
              default_max,
              default_min):
    """
    Helper method for providing a size, or a range to randomize from

    Args:
        size (n-array): Array of numbers that explicitly define the size
        size_max (n-array): Array of numbers that define the custom max size from which to randomly sample
        size_min (n-array): Array of numbers that define the custom min size from which to randomly sample
        default_max (n-array): Array of numbers that define the default max size from which to randomly sample
        default_min (n-array): Array of numbers that define the default min size from which to randomly sample

    Returns:
        np.array: size generated

    Raises:
        ValueError: [Inconsistent array sizes]
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
    return np.array(size)


class BoxObject(MujocoGeneratedObject):
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
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07, 0.07],
                         [0.03, 0.03, 0.03])
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

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2)

    def get_object_subtree(self, site=False):
        return self._get_object_subtree(site=site, ob_type="box")


class CylinderObject(MujocoGeneratedObject):
    """
    A cylinder object.

    Args:
        size (2-tuple of float): (radius, half-length) size parameters for this cylinder object
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
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07],
                         [0.03, 0.03])
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
        assert len(self.size) == 2, "cylinder size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[1]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[1]])

    def get_horizontal_radius(self):
        return self.size[0]

    def get_object_subtree(self, site=False):
        return self._get_object_subtree(site=site, ob_type="cylinder")


class BallObject(MujocoGeneratedObject):
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
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07],
                         [0.03])
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

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[0]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[0]])

    def get_horizontal_radius(self):
        return self.size[0]

    def get_object_subtree(self, site=False):
        return self._get_object_subtree(site=site, ob_type="sphere")


class CapsuleObject(MujocoGeneratedObject):
    """
    A capsule object.

    Args:
        size (2-tuple of float): (radius, half-length) size parameters for this capsule object
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
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07],
                         [0.03, 0.03])
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
        assert len(self.size) == 2, "capsule size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * (self.size[0] + self.size[1])])

    def get_top_offset(self):
        return np.array([0, 0, (self.size[0] + self.size[1])])

    def get_horizontal_radius(self):
        return self.size[0]

    def get_object_subtree(self, site=False):
        return self._get_object_subtree(site=site, ob_type="capsule")
