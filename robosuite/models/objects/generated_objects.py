import numpy as np

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site, new_joint, array_to_string
from robosuite.utils.mjcf_utils import OBJECT_COLLISION_COLOR, CustomMaterial

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

        self._name = name

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

    @property
    def bottom_offset(self):
        return np.array([0., 0., -self.total_size[2]])

    @property
    def top_offset(self):
        return np.array([0., 0., self.total_size[2]])

    @property
    def horizontal_radius(self):
        return np.linalg.norm(self.total_size[:2], 2)


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
        self._name = name

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

    def bottom_offset(self):
        raise NotImplementedError

    def top_offset(self):
        raise NotImplementedError

    def horizontal_radius(self):
        raise NotImplementedError
