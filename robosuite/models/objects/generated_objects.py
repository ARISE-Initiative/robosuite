import numpy as np

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site, new_joint, array_to_string, add_to_dict, get_size
from robosuite.utils.mjcf_utils import RED, GREEN, BLUE, CYAN, OBJECT_COLLISION_COLOR, CustomMaterial
import robosuite.utils.transform_utils as T

from collections.abc import Iterable

from copy import deepcopy


class CompositeObject(MujocoGeneratedObject):
    """
    An object constructed out of basic geoms to make more intricate shapes.

    Note that by default, specifying None for a specific geom element will usually set a value to the mujoco defaults.

    Args:
        name (str): Name of overall object

        total_size (list): (x, y, z) half-size in each dimension for the bounding box for
            this Composite object

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

        rgba (list): (r, g, b, a) default values to use if geom-specific @geom_rgbas isn't specified for a given element

        density (float or list of float): either single value to use for all geom densities or geom-specific values

        solref (list or list of list): parameters used for the mujoco contact solver. Can be single set of values or
            element-specific values. See http://www.mujoco.org/book/modeling.html#CSolver for details.

        solimp (list or list of list): parameters used for the mujoco contact solver. Can be single set of values or
            element-specific values. See http://www.mujoco.org/book/modeling.html#CSolver for details.

        locations_relative_to_center (bool): If true, @geom_locations will be considered relative to the center of the
            overall object bounding box defined by @total_size. Else, the corner of this bounding box is considered the
            origin.

        body_mapping (None or dict): If specified, defines a multi-body object, where each entry in the dict maps a
            geom_name to a new sub-body definition. This allows for multi-bodied objects with nested geoms. When the
            keyword-specified geom_name is encountered during procedural generation, the corresponding body
            specification will be used to generate a new body element, which will be appended to the current XML Tree.
            All subsequent geoms will then be appended to this new child body. Default is None, which corresponds
            to the single-body case

        joints (None or str or list): Joints to use for each body. If None, no joints will be used for this entire
            object. If "default", a single free joint will be added to the top-level body of this object. Otherwise,
            should be a list equal to length to the number of entries in body_mapping + 1, where each entry should
            either be None or its own list of dictionaries. Each dictionary should specify the specific joint
            attributes necessary. See http://www.mujoco.org/book/XMLreference.html#joint for reference.

        sites (None or list): list of sites to add to each body. If None, only the default top-level object site will
            be used. Otherwise, should be a list equal to number of entries in body_mapping + 1, where each entry should
            either be None or its own list of dictionaries. Each dictionary should specify the appropriate attributes
            for the given site. See http://www.mujoco.org/book/XMLreference.html#site for reference.

        obj_types (str or list of str): either single obj_type for all geoms or geom-specific type. Choices are
            {"collision", "visual", "all"}
    """

    def __init__(
        self,
        name,
        total_size,
        geom_types,
        geom_locations,
        geom_sizes,
        geom_names=None,
        geom_rgbas=None,
        geom_materials=None,
        geom_frictions=None,
        geom_quats=None,
        rgba=None,
        density=100.,
        solref=(0.02, 1.),
        solimp=(0.9, 0.95, 0.001),
        locations_relative_to_center=False,
        body_mapping=None,
        joints="default",
        sites=None,
        obj_types="all",
        duplicate_collision_geoms=True,
    ):
        # Always call superclass first
        super().__init__(duplicate_collision_geoms=duplicate_collision_geoms)

        self.name = name

        # Create bodies
        self.body_mapping = {"root": {"name": "main"}}
        if body_mapping is not None:
            self.body_mapping.update(body_mapping)
        n_bodies = len(self.body_mapping.keys())

        # Set joints
        if joints is None:
            self.body_joint_specs = [None] * n_bodies
        elif joints == "default":
            self.body_joint_specs = [[self.get_joint_attrib_template()]] + [None] * (n_bodies - 1)
        else:
            self.body_joint_specs = joints

        # Make sure all joints are named appropriately
        j_num = 0
        for joint_spec in self.body_joint_specs:
            if type(joint_spec) in {tuple, list}:
                for j_spec in joint_spec:
                    if "name" not in j_spec:
                        j_spec["name"] = "joint{}".format(j_num)
                        j_num += 1

        # Set sites
        self.body_site_specs = sites if sites is not None else [None] * n_bodies

        # Make sure all sites are named appropriately
        s_num = 0
        for site_spec in self.body_site_specs:
            if type(site_spec) in {tuple, list}:
                for s_spec in site_spec:
                    if "name" not in s_spec:
                        s_spec["name"] = "site{}".format(s_num)
                        s_num += 1

        n_geoms = len(geom_types)
        self.total_size = np.array(total_size)
        self.geom_types = np.array(geom_types)
        self.geom_locations = np.array(geom_locations)
        self.geom_sizes = deepcopy(geom_sizes)
        self.geom_names = list(geom_names) if geom_names is not None else [None] * n_geoms
        self.geom_rgbas = list(geom_rgbas) if geom_rgbas is not None else [None] * n_geoms
        self.geom_materials = list(geom_materials) if geom_materials is not None else [None] * n_geoms
        self.geom_frictions = list(geom_frictions) if geom_frictions is not None else [None] * n_geoms
        self.density = [density] * n_geoms if density is None or type(density) in {float, int} else list(density)
        self.solref = [solref] * n_geoms if solref is None or type(solref[0]) in {float, int} else list(solref)
        self.solimp = [solimp] * n_geoms if obj_types is None or type(solimp[0]) in {float, int} else list(solimp)
        self.rgba = rgba        # override superclass setting of this variable
        self.locations_relative_to_center = locations_relative_to_center
        self.geom_quats = deepcopy(geom_quats) if geom_quats is not None else [None] * n_geoms
        self.obj_types = [obj_types] * n_geoms if obj_types is None or type(obj_types) is str else list(obj_types)

        # Always run sanity check
        self.sanity_check()

        # Lastly, parse XML tree appropriately
        self._obj = self._get_object_subtree()

        # Extract the appropriate private attributes for this
        self._get_object_properties()

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

    def get_bottom_offset(self):
        return np.array([0., 0., -self.total_size[2]])

    def get_top_offset(self):
        return np.array([0., 0., self.total_size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.total_size[:2], 2)

    def _get_object_subtree(self):
        # Initialize top-level body
        obj = new_body(**self.body_mapping["root"])
        cur_obj = obj
        body_num = 0

        # Add top level joint(s)
        if self.body_joint_specs[0] is not None:
            for j_spec in self.body_joint_specs[0]:
                obj.append(new_joint(**j_spec))

        # Add top level site(s)
        if self.body_site_specs[0] is not None:
            for s_spec in self.body_site_specs[0]:
                obj.append(new_site(**s_spec))

        # Add default object top-level site
        site_element_attr = self.get_site_attrib_template()
        site_element_attr["rgba"] = "1 0 0 0"
        site_element_attr["name"] = "default_site"
        obj.append(new_site(**site_element_attr))

        # Loop through all geoms and generate the composite object
        for i, (obj_type, g_type, g_size, g_loc, g_name, g_rgba, g_friction,
                g_quat, g_material, g_density, g_solref, g_solimp) in enumerate(zip(
                self.obj_types,
                self.geom_types,
                self.geom_sizes,
                self.geom_locations,
                self.geom_names,
                self.geom_rgbas,
                self.geom_frictions,
                self.geom_quats,
                self.geom_materials,
                self.density,
                self.solref,
                self.solimp,
        )):
            # geom type
            geom_type = g_type
            # get cartesian size from size spec
            size = g_size
            cartesian_size = self._size_to_cartesian_half_lengths(geom_type, size)
            if self.locations_relative_to_center:
                # no need to convert
                pos = g_loc
            else:
                # use geom location to convert to position coordinate (the origin is the
                # center of the composite object)
                pos = [
                    (-self.total_size[0] + cartesian_size[0]) + g_loc[0],
                    (-self.total_size[1] + cartesian_size[1]) + g_loc[1],
                    (-self.total_size[2] + cartesian_size[2]) + g_loc[2],
                ]

            # geom name
            geom_name = g_name if g_name is not None else f"g{i}"

            # geom rgba
            geom_rgba = g_rgba if g_rgba is not None else self.rgba

            # geom friction
            geom_friction = array_to_string(g_friction) if g_friction is not None else \
                            array_to_string(np.array([1., 0.005, 0.0001]))  # mujoco default

            # Define base geom attributes
            geom_attr = {
                "size": size,
                "pos": pos,
                "name": geom_name,
                "type": geom_type,
            }

            # Optionally define quat if specified
            if g_quat is not None:
                geom_attr['quat'] = array_to_string(g_quat)

            # Add collision geom if necessary
            if obj_type in {"collision", "all"}:
                col_geom_attr = deepcopy(geom_attr)
                col_geom_attr.update(self.get_collision_attrib_template())
                if g_density is not None:
                    col_geom_attr['density'] = str(g_density)
                col_geom_attr['friction'] = geom_friction
                col_geom_attr['solref'] = array_to_string(g_solref)
                col_geom_attr['solimp'] = array_to_string(g_solimp)
                col_geom_attr['rgba'] = OBJECT_COLLISION_COLOR
                cur_obj.append(new_geom(**col_geom_attr))

            # Add visual geom if necessary
            if obj_type in {"visual", "all"}:
                vis_geom_attr = deepcopy(geom_attr)
                vis_geom_attr.update(self.get_visual_attrib_template())
                vis_geom_attr["name"] += "_vis"
                if g_material is not None:
                    vis_geom_attr['material'] = g_material
                vis_geom_attr["rgba"] = geom_rgba
                cur_obj.append(new_geom(**vis_geom_attr))

            # If the current geom is in our body_mapping, then we need to create a new nested body to append to
            if g_name in self.body_mapping:
                # Increment body count
                body_num += 1
                child_body = new_body(**self.body_mapping[g_name])
                # Set the current body to be this new body
                cur_obj = child_body
                # Add appropriate joint(s) and site(s)
                if self.body_joint_specs[body_num] is not None:
                    for j_spec in self.body_joint_specs[body_num]:
                        cur_obj.append(new_joint(**j_spec))
                if self.body_site_specs[body_num] is not None:
                    for s_spec in self.body_site_specs[body_num]:
                        cur_obj.append(new_site(**s_spec))

        return obj

    @staticmethod
    def _size_to_cartesian_half_lengths(geom_type, geom_size):
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
        base_args = {
            "total_size": full_size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        obj_args = {}

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

        # Element references to be filled when generated
        self._handle0_geoms = None
        self._handle1_geoms = None
        self.pot_base = None

        # Other private attributes
        self._important_sites = {}

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
        base_args = {
            "total_size": full_size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        site_attrs = []
        obj_args = {}

        # Initialize geom lists
        self._handle0_geoms = []
        self._handle1_geoms = []

        # Add main pot body
        # Base geom
        name = f"base"
        self.pot_base = [name]
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
        for i, (g_list, handle_side, rgba) in enumerate(zip(
                [self._handle0_geoms, self._handle1_geoms],
                [1.0, -1.0],
                [self.rgba_handle_0, self.rgba_handle_1]
        )):
            handle_center = np.array((0, handle_side * (self.body_half_size[1] + self.handle_length), handle_z))
            # Solid handle case
            if self.solid_handle:
                name = f"handle{i}"
                g_list.append(name)
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
                g_list.append(name)
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
                    g_list.append(name)
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
            handle_name = f"handle{i}"
            handle_site.update({
                "name": handle_name,
                "pos": array_to_string(handle_center - handle_side * np.array([0, 0.005, 0])),
                "size": "0.005",
                "rgba": rgba,
            })
            site_attrs.append(handle_site)
            # Add to important sites
            self._important_sites[f"handle{i}"] = self.naming_prefix + handle_name

        # Add pot body site
        pot_site = self.get_site_attrib_template()
        center_name = "center"
        pot_site.update({
            "name": center_name,
            "size": "0.005",
        })
        site_attrs.append(pot_site)
        # Add to important sites
        self._important_sites["center"] = self.naming_prefix + center_name

        # Add back in base args and site args
        obj_args.update(base_args)
        obj_args["sites"] = [site_attrs]        # All sites are part of main (top) body

        # Return this dict
        return obj_args

    @property
    def handle_distance(self):

        """
        Calculates how far apart the handles are

        Returns:
            float: handle distance
        """
        return self.body_half_size[1] * 2 + self.handle_length * 2

    @property
    def handle0_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to handle0 (green handle)
        """
        return self.correct_naming(self._handle0_geoms)

    @property
    def handle1_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to handle1 (blue handle)
        """
        return self.correct_naming(self._handle1_geoms)

    @property
    def handle_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to both handles
        """
        return self.handle0_geoms + self.handle1_geoms

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle0'`: Name of handle0 location site
                :`'handle1'`: Name of handle1 location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update(self._important_sites)
        return dic


class PrimitiveObject(MujocoGeneratedObject):
    """
    Base class for all programmatically generated mujoco object
    i.e., every MujocoObject that does not have an corresponding xml file

    Args:
        name (str): (unique) name to identify this generated object

        size (n-tuple of float): relevant size parameters for the object, should be of size 1 - 3

        rgba (4-tuple of float): Color

        density (float): Density

        friction (3-tuple of float): (sliding friction, torsional friction, and rolling friction).
            A single float can also be specified, in order to set the sliding friction (the other values) will
            be set to the MuJoCo default. See http://www.mujoco.org/book/modeling.html#geom for details.

        solref (2-tuple of float): MuJoCo solver parameters that handle contact.
            See http://www.mujoco.org/book/XMLreference.html for more details.

        solimp (3-tuple of float): MuJoCo solver parameters that handle contact.
            See http://www.mujoco.org/book/XMLreference.html for more details.

        material (CustomMaterial or `'default'` or None): if "default", add a template material and texture for this
            object that is used to color the geom(s).
            Otherwise, input is expected to be a CustomMaterial object

            See http://www.mujoco.org/book/XMLreference.html#asset for specific details on attributes expected for
            Mujoco texture / material tags, respectively

            Note that specifying a custom texture in this way automatically overrides any rgba values set

        joints (None or str or list of dict): Joints for this object. If None, no joint will be created. If "default",
            a single (free) joint will be crated. Else, should be a list of dict, where each dictionary corresponds to
            a joint that will be created for this object. The dictionary should specify the joint attributes
            (type, pos, etc.) according to the MuJoCo xml specification.

        obj_type (str): Geom elements to generate / extract for this object. Must be one of:

            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned

        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy
    """

    def __init__(
        self,
        name,
        size=None,
        rgba=None,
        density=None,
        friction=None,
        solref=None,
        solimp=None,
        material=None,
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=True,
    ):
        # Always call superclass first
        super().__init__(obj_type=obj_type, duplicate_collision_geoms=duplicate_collision_geoms)

        # Set name
        self.name = name

        if size is None:
            size = [0.05, 0.05, 0.05]
        self.size = list(size)

        if rgba is None:
            rgba = [1, 0, 0, 1]
        assert len(rgba) == 4, "rgba must be a length 4 array"
        self.rgba = list(rgba)

        if density is None:
            density = 1000  # water
        self.density = density

        if friction is None:
            friction = [1, 0.005, 0.0001]  # MuJoCo default
        elif isinstance(friction, float) or isinstance(friction, int):
            friction = [friction, 0.005, 0.0001]
        assert len(friction) == 3, "friction must be a length 3 array or a single number"
        self.friction = list(friction)

        if solref is None:
            self.solref = [0.02, 1.]  # MuJoCo default
        else:
            self.solref = solref

        if solimp is None:
            self.solimp = [0.9, 0.95, 0.001]  # MuJoCo default
        else:
            self.solimp = solimp

        self.material = material
        if material == "default":
            # add in default texture and material for this object (for domain randomization)
            default_tex = CustomMaterial(
                texture=self.rgba,
                tex_name="tex",
                mat_name="mat",
            )
            self.append_material(default_tex)
        elif material is not None:
            # add in custom texture and material
            self.append_material(material)

        # joints for this object
        if joints == "default":
            self.joint_specs = [self.get_joint_attrib_template()]  # default free joint
        elif joints is None:
            self.joint_specs = []
        else:
            self.joint_specs = joints

        # Make sure all joints have names!
        for i, joint_spec in enumerate(self.joint_specs):
            if "name" not in joint_spec:
                joint_spec["name"] = "joint{}".format(i)

        # Always run sanity check
        self.sanity_check()

        # Lastly, parse XML tree appropriately
        self._obj = self._get_object_subtree()

        # Extract the appropriate private attributes for this
        self._get_object_properties()

    def _get_object_subtree_(self, ob_type="box"):
        # Create element tree
        obj = new_body(name="main")

        # Get base element attributes
        element_attr = {
            "name": "g0",
            "type": ob_type,
            "size": array_to_string(self.size)
        }

        # Add collision geom if necessary
        if self.obj_type in {"collision", "all"}:
            col_element_attr = deepcopy(element_attr)
            col_element_attr.update(self.get_collision_attrib_template())
            col_element_attr["density"] = str(self.density)
            col_element_attr["friction"] = array_to_string(self.friction)
            col_element_attr["solref"] = array_to_string(self.solref)
            col_element_attr["solimp"] = array_to_string(self.solimp)
            obj.append(new_geom(**col_element_attr))
        # Add visual geom if necessary
        if self.obj_type in {"visual", "all"}:
            vis_element_attr = deepcopy(element_attr)
            vis_element_attr.update(self.get_visual_attrib_template())
            vis_element_attr["name"] += "_vis"
            if self.material == "default":
                vis_element_attr["rgba"] = "0.5 0.5 0.5 1"  # mujoco default
                vis_element_attr["material"] = "mat"
            elif self.material is not None:
                vis_element_attr["material"] = self.material.mat_attrib["name"]
            else:
                vis_element_attr["rgba"] = array_to_string(self.rgba)
            obj.append(new_geom(**vis_element_attr))
        # add joint(s)
        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))
        # add a site as well
        site_element_attr = self.get_site_attrib_template()
        site_element_attr["name"] = "default_site"
        obj.append(new_site(**site_element_attr))
        return obj

    # Methods that still need to be defined by subclass
    def _get_object_subtree(self):
        raise NotImplementedError

    def get_bottom_offset(self):
        raise NotImplementedError

    def get_top_offset(self):
        raise NotImplementedError

    def get_horizontal_radius(self):
        raise NotImplementedError


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
    ):
        size = get_size(size,
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

    def _get_object_subtree(self):
        return self._get_object_subtree_(ob_type="box")


class CylinderObject(PrimitiveObject):
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
        size = get_size(size,
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

    def _get_object_subtree(self):
        return self._get_object_subtree_(ob_type="cylinder")


class BallObject(PrimitiveObject):
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
        size = get_size(size,
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

    def _get_object_subtree(self):
        return self._get_object_subtree_(ob_type="sphere")


class CapsuleObject(PrimitiveObject):
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
        size = get_size(size,
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

    def _get_object_subtree(self):
        return self._get_object_subtree_(ob_type="capsule")
