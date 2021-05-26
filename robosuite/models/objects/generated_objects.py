import numpy as np

from robosuite.models.objects import MujocoGeneratedObject, MujocoObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site, new_joint, new_inertial,\
    array_to_string, find_elements, add_prefix, OBJECT_COLLISION_COLOR, CustomMaterial

from copy import deepcopy


class CompositeBodyObject(MujocoGeneratedObject):
    """
    An object constructed out of multiple bodies to make more complex shapes.

    Args:
        name (str): Name of overall object

        objects (MujocoObject or list of MujocoObjects): object(s) to combine to form the composite body object.
            Note that these objects will be added sequentially, so if an object is required to be nested relative to
            another object, that nested object should be listed after the parent object. Note that all top-level joints
            for any inputted objects are automatically stripped

        object_locations (list): list of body locations in the composite. Each
            location should be a list or tuple of 3 elements and all
            locations are taken relative to that object's parent body. Giving None for a location results in (0,0,0)
            for that object.

        object_quats (None or list): list of (w, x, y, z) quaternions for each body. None results in (1,0,0,0) for
            that object.

        object_parents (None or list): Parent bodies to append each object to. Note that specifying "None" will
            automatically append all objects to the root body ("root")

        joints (None or list): Joints to use for the top-level composite body object. If None, no joints will be used
            for this top-level object. If "default", a single free joint will be added to the top-level body of this
            object. Otherwise, should be a list of dictionaries, where each dictionary should specify the specific
            joint attributes necessary. See http://www.mujoco.org/book/XMLreference.html#joint for reference.

        body_joints (None or dict): If specified, maps body names to joint specifications to append to that
            body. If None, no extra joints will be used. If mapped value is "default", a single free joint will be
            added to the specified body. Otherwise, should be a list of dictionaries, where each dictionary should
            specify the specific joint attributes necessary. See http://www.mujoco.org/book/XMLreference.html#joint
            for reference.

        sites (None or list): list of sites to add to top-level composite body object. If None, only the default
            top-level object site will be used. Otherwise, should be a list of dictionaries, where each dictionary
            should specify the appropriate attributes for the given site.
            See http://www.mujoco.org/book/XMLreference.html#site for reference.
    """
    def __init__(
        self,
        name,
        objects,
        object_locations,
        object_quats=None,
        object_parents=None,
        joints="default",
        body_joints=None,
        sites=None,
    ):
        # Always call superclass first
        super().__init__()

        self._name = name

        # Set internal variable geometric properties which will be modified later
        self._object_absolute_positions = {"root": np.zeros(3)}     # maps body names to abs positions (rel to root)
        self._top = 0
        self._bottom = 0
        self._horizontal = 0

        # Standardize inputs
        if isinstance(objects, MujocoObject):
            self.objects = [objects]
        elif type(objects) in {list, tuple}:
            self.objects = list(objects)
        else:
            # Invalid objects received
            raise ValueError("Invalid objects received, got type: {}".format(type(objects)))

        n_objects = len(self.objects)
        self.object_locations = np.array(object_locations)
        self.object_quats = deepcopy(object_quats) if object_quats is not None else [None] * n_objects
        self.object_parents = deepcopy(object_parents) if object_parents is not None else ["root"] * n_objects

        # Set joints
        if joints == "default":
            self.joint_specs = [self.get_joint_attrib_template()]  # default free joint
        elif joints is None:
            self.joint_specs = []
        else:
            self.joint_specs = joints

        # Set body joints
        if body_joints is None:
            body_joints = {}
        self.body_joint_specs = body_joints

        # Make sure all joints are named appropriately
        j_num = 0
        for joint_spec in self.joint_specs:
            if "name" not in joint_spec:
                joint_spec["name"] = "joint{}".format(j_num)
                j_num += 1

        # Set sites
        self.site_specs = deepcopy(sites) if sites is not None else []
        # Add default site
        site_element_attr = self.get_site_attrib_template()
        site_element_attr["rgba"] = "1 0 0 0"
        site_element_attr["name"] = "default_site"
        self.site_specs.append(site_element_attr)

        # Make sure all sites are named appropriately
        s_num = 0
        for site_spec in self.site_specs:
            if "name" not in site_spec:
                site_spec["name"] = "site{}".format(s_num)
                s_num += 1

        # Always run sanity check
        self.sanity_check()

        # Lastly, parse XML tree appropriately
        self._obj = self._get_object_subtree()

        # Extract the appropriate private attributes for this
        self._get_object_properties()

    def _get_object_subtree(self):
        # Initialize top-level body
        obj = new_body(name="root")

        # # Give main body a small mass in order to have a free joint (only needed for mujoco 1.5)
        # obj.append(new_inertial(pos=(0, 0, 0), mass=0.0001, diaginertia=(0.0001, 0.0001, 0.0001)))

        # Add all joints and sites
        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))
        for site_spec in self.site_specs:
            obj.append(new_site(**site_spec))

        # Loop through all objects and associated args and append them appropriately
        for o, o_parent, o_pos, o_quat in zip(
                self.objects,
                self.object_parents,
                self.object_locations,
                self.object_quats
        ):
            self._append_object(root=obj, obj=o, parent_name=o_parent, pos=o_pos, quat=o_quat)

        # Loop through all joints and append them appropriately
        for body_name, joint_specs in self.body_joint_specs.items():
            self._append_joints(root=obj, body_name=body_name, joint_specs=joint_specs)

        # Return final object
        return obj

    def _get_object_properties(self):
        """
        Extends the superclass method to add prefixes to all assets
        """
        super()._get_object_properties()
        # Add prefix to all assets
        add_prefix(root=self.asset, prefix=self.naming_prefix, exclude=self.exclude_from_prefixing)

    def _append_object(self, root, obj, parent_name=None, pos=None, quat=None):
        """
        Helper function to add pre-generated object @obj to the body with name @parent_name

        Args:
            root (ET.Element): Top-level element to iteratively search through for @parent_name to add @obj to
            obj (MujocoObject): Object to append to the body specified by @parent_name
            parent_name (None or str): Body name to search for in @root to append @obj to.
                None defaults to "root" (top-level body)
            pos (None or 3-array): (x,y,z) relative offset from parent body when appending @obj.
                None defaults to (0,0,0)
            quat (None or 4-array) (w,x,y,z) relative quaternion rotation from parent body when appending @obj.
                None defaults to (1,0,0,0)
        """
        # Set defaults if any are None
        if parent_name is None:
            parent_name = "root"
        if pos is None:
            pos = np.zeros(3)
        if quat is None:
            quat = np.array([1, 0, 0, 0])
        # First, find parent body
        parent = find_elements(root=root, tags="body", attribs={"name": parent_name}, return_first=True)
        assert parent is not None, "Could not find parent body with name: {}".format(parent_name)
        # Get the object xml element tree, remove its top-level joints, and modify its top-level pos / quat
        child = obj.get_obj()
        self._remove_joints(child)
        child.set("pos", array_to_string(pos))
        child.set("quat", array_to_string(quat))
        # Add this object and its assets to this composite object
        self.merge_assets(other=obj)
        parent.append(child)
        # Update geometric properties for this composite object
        obj_abs_pos = self._object_absolute_positions[parent_name] + np.array(pos)
        self._object_absolute_positions[obj.root_body] = obj_abs_pos
        self._top = max(self._top, obj_abs_pos[2] + obj.top_offset[2])
        self._bottom = min(self._bottom, obj_abs_pos[2] + obj.bottom_offset[2])
        self._horizontal = max(self._horizontal, max(obj_abs_pos[:2]) + obj.horizontal_radius)

    def _append_joints(self, root, body_name=None, joint_specs="default"):
        """
        Appends all joints as specified by @joint_specs to @body.

        Args:
            root (ET.Element): Top-level element to iteratively search through for @body_name
            body_name (None or str): Name of the body to append the joints to.
                None defaults to "root" (top-level body)
            joint_specs (str or list): List of joint specifications to add to the specified body, or
                "default", which results in a single free joint
        """
        # Standardize inputs
        if body_name is None:
            body_name = "root"
        if joint_specs == "default":
            joint_specs = [self.get_joint_attrib_template()]
        for i, joint_spec in enumerate(joint_specs):
            if "name" not in joint_spec:
                joint_spec["name"] = f"{body_name}_joint{i}"
        # Search for body and make sure it exists
        body = find_elements(root=root, tags="body", attribs={"name": body_name}, return_first=True)
        assert body is not None, "Could not find body with name: {}".format(body_name)
        # Add joint(s) to this body
        for joint_spec in joint_specs:
            body.append(new_joint(**joint_spec))

    @staticmethod
    def _remove_joints(body):
        """
        Helper function to strip all joints directly appended to the specified @body.

        Args:
            body (ET.Element): Body to strip joints from
        """
        children_to_remove = []
        for child in body:
            if child.tag == "joint":
                children_to_remove.append(child)
        for child in children_to_remove:
            body.remove(child)

    @property
    def bottom_offset(self):
        return np.array([0., 0., self._bottom])

    @property
    def top_offset(self):
        return np.array([0., 0., self._top])

    @property
    def horizontal_radius(self):
        return self._horizontal


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

        geom_quats (None or list): list of (w, x, y, z) quaternions for each geom.

        geom_names (None or list): list of geom names ordered the same as @geom_locations. The
            names will get appended with an underscore to the passed name in @get_collision
            and @get_visual

        geom_rgbas (None or list): list of geom colors ordered the same as @geom_locations. If
            passed as an argument, @rgba is ignored.

        geom_materials (None or list of CustomTexture): list of custom textures to use for this object material

        geom_frictions (None or list): list of geom frictions to use for each geom.

        rgba (None or list): (r, g, b, a) default values to use if geom-specific @geom_rgbas isn't specified for a given element

        density (float or list of float): either single value to use for all geom densities or geom-specific values

        solref (list or list of list): parameters used for the mujoco contact solver. Can be single set of values or
            element-specific values. See http://www.mujoco.org/book/modeling.html#CSolver for details.

        solimp (list or list of list): parameters used for the mujoco contact solver. Can be single set of values or
            element-specific values. See http://www.mujoco.org/book/modeling.html#CSolver for details.

        locations_relative_to_center (bool): If true, @geom_locations will be considered relative to the center of the
            overall object bounding box defined by @total_size. Else, the corner of this bounding box is considered the
            origin.

        joints (None or list): Joints to use for this composite object. If None, no joints will be used
            for this top-level object. If "default", a single free joint will be added to this object.
            Otherwise, should be a list of dictionaries, where each dictionary should specify the specific
            joint attributes necessary. See http://www.mujoco.org/book/XMLreference.html#joint for reference.

        sites (None or list): list of sites to add to this composite object. If None, only the default
             object site will be used. Otherwise, should be a list of dictionaries, where each dictionary
            should specify the appropriate attributes for the given site.
            See http://www.mujoco.org/book/XMLreference.html#site for reference.

        obj_types (str or list of str): either single obj_type for all geoms or geom-specific type. Choices are
            {"collision", "visual", "all"}
    """

    def __init__(
        self,
        name,
        total_size,
        geom_types,
        geom_sizes,
        geom_locations,
        geom_quats=None,
        geom_names=None,
        geom_rgbas=None,
        geom_materials=None,
        geom_frictions=None,
        geom_condims=None,
        rgba=None,
        density=100.,
        solref=(0.02, 1.),
        solimp=(0.9, 0.95, 0.001),
        locations_relative_to_center=False,
        joints="default",
        sites=None,
        obj_types="all",
        duplicate_collision_geoms=True,
    ):
        # Always call superclass first
        super().__init__(duplicate_collision_geoms=duplicate_collision_geoms)

        self._name = name

        # Set joints
        if joints == "default":
            self.joint_specs = [self.get_joint_attrib_template()]  # default free joint
        elif joints is None:
            self.joint_specs = []
        else:
            self.joint_specs = joints

        # Make sure all joints are named appropriately
        j_num = 0
        for joint_spec in self.joint_specs:
            if "name" not in joint_spec:
                joint_spec["name"] = "joint{}".format(j_num)
                j_num += 1

        # Set sites
        self.site_specs = deepcopy(sites) if sites is not None else []
        # Add default site
        site_element_attr = self.get_site_attrib_template()
        site_element_attr["rgba"] = "1 0 0 0"
        site_element_attr["name"] = "default_site"
        self.site_specs.append(site_element_attr)

        # Make sure all sites are named appropriately
        s_num = 0
        for site_spec in self.site_specs:
            if "name" not in site_spec:
                site_spec["name"] = "site{}".format(s_num)
                s_num += 1

        n_geoms = len(geom_types)
        self.total_size = np.array(total_size)
        self.geom_types = np.array(geom_types)
        self.geom_sizes = deepcopy(geom_sizes)
        self.geom_locations = np.array(geom_locations)
        self.geom_quats = deepcopy(geom_quats) if geom_quats is not None else [None] * n_geoms
        self.geom_names = list(geom_names) if geom_names is not None else [None] * n_geoms
        self.geom_rgbas = list(geom_rgbas) if geom_rgbas is not None else [None] * n_geoms
        self.geom_materials = list(geom_materials) if geom_materials is not None else [None] * n_geoms
        self.geom_frictions = list(geom_frictions) if geom_frictions is not None else [None] * n_geoms
        self.geom_condims = list(geom_condims) if geom_condims is not None else [None] * n_geoms
        self.density = [density] * n_geoms if density is None or type(density) in {float, int} else list(density)
        self.solref = [solref] * n_geoms if solref is None or type(solref[0]) in {float, int} else list(solref)
        self.solimp = [solimp] * n_geoms if obj_types is None or type(solimp[0]) in {float, int} else list(solimp)
        self.rgba = rgba        # override superclass setting of this variable
        self.locations_relative_to_center = locations_relative_to_center
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
        obj = new_body(name="root")

        # Add all joints and sites
        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))
        for site_spec in self.site_specs:
            obj.append(new_site(**site_spec))

        # Loop through all geoms and generate the composite object
        for i, (obj_type, g_type, g_size, g_loc, g_name, g_rgba, g_friction, g_condim,
                g_quat, g_material, g_density, g_solref, g_solimp) in enumerate(zip(
                self.obj_types,
                self.geom_types,
                self.geom_sizes,
                self.geom_locations,
                self.geom_names,
                self.geom_rgbas,
                self.geom_frictions,
                self.geom_condims,
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
                if g_condim is not None:
                    col_geom_attr['condim'] = str(g_condim)
                obj.append(new_geom(**col_geom_attr))

            # Add visual geom if necessary
            if obj_type in {"visual", "all"}:
                vis_geom_attr = deepcopy(geom_attr)
                vis_geom_attr.update(self.get_visual_attrib_template())
                vis_geom_attr["name"] += "_vis"
                if g_material is not None:
                    vis_geom_attr['material'] = g_material
                vis_geom_attr["rgba"] = geom_rgba
                obj.append(new_geom(**vis_geom_attr))

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
