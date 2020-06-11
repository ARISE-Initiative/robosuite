import numpy as np

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site, array_to_string
from robosuite.utils.mjcf_utils import RED, GREEN, BLUE

from collections import Iterable

# Define custom colors
CYAN = [0, 1, 1, 1]


class HammerObject(MujocoGeneratedObject):
    """
    Generates a Hammer object with a cylindrical or box-shaped handle, cubic head, cylindrical face and triangular claw
    (used in Handoff task)
    """

    def __init__(
        self,
        handle_shape="box",
        handle_radius=(0.015, 0.02),
        handle_length=(0.1, 0.25),
        handle_density=(100, 250),
        handle_friction=(3.0, 5.0),
        head_density_ratio=2.0,
        rgba_handle=None,
        rgba_head=None,
        rgba_face=None,
        rgba_claw=None,
        joints=None,
    ):
        """
        handle_shape (str): Either "box", for a box-shaped handle, or "cylinder", for a cylindrically-shaped handle
        handle_radius (float or 2-list of float): Either specific or range of values to draw randomly from
            uniformly for the handle radius
        handle_length (float or 2-list of float): Either specific or range of values to draw randomly from
            uniformly for the handle length
        handle_density (float or 2-list of float): Either specific or range of values to draw randomly from
            uniformly for the handle density (in SI units). Note that this value is scaled x4 for the hammer head
        handle_friction (float or 2-list of float): Either specific or range of values to draw randomly from
            uniformly for the handle friction. Note that Mujoco default values are used for the head
        head_density_ratio (float): Ratio of density of handle to head (including face and claw)
        rgba_handle (3-array or None): If specified, sets handle rgba values
        rgba_head (3-array or None): If specified, sets handle rgba values
        rgba_face (3-array or None): If specified, sets handle rgba values
        rgba_claw (3-array or None): If specified, sets handle rgba values
        joints ([dict]): list of dictionaries - each dictionary corresponds to a joint that will be created for this
            object. The dictionary should specify the joint attributes (type, pos, etc.) according to the MuJoCo
            xml specification.
        """

        # Run super() init
        super().__init__(joints=joints)

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

        # Initialize RGBA values
        self.rgba_handle = rgba_handle if rgba_handle is not None else RED
        self.rgba_head = rgba_head if rgba_head is not None else CYAN
        self.rgba_face = rgba_face if rgba_face is not None else BLUE
        self.rgba_claw = rgba_claw if rgba_claw is not None else GREEN

    def get_bottom_offset(self):
        return np.array([0, 0, -0.5 * self.handle_length])

    def get_top_offset(self):
        return np.array([0, 0, self.handle_radius])

    def get_horizontal_radius(self):
        return self.head_halfsize + 0.5 * self.handle_length

    @property
    def handle_distance(self):
        # TODO
        return 2.0 * self.handle_radius
        return self.body_half_size[1] * 2 + self.handle_length * 2

    def get_collision(self, name=None, site=None):
        # Create new body
        main_body = new_body()

        # Define name for this object if specified
        if name is not None:
            main_body.set("name", name)

        # Define handle and append to the main body
        if self.handle_shape == "cylinder":
            main_body.append(
                new_geom(
                    geom_type="cylinder",
                    name="hammer_handle",
                    size=[self.handle_radius, self.handle_length / 2.0],
                    pos=(0, 0, 0),
                    rgba=self.rgba_handle,
                    group=1,
                    density=str(self.handle_density),
                    friction=array_to_string((self.handle_friction, 0.005, 0.0001)),
                )
            )
        elif self.handle_shape == "box":
            main_body.append(
                new_geom(
                    geom_type="box",
                    name="hammer_handle",
                    size=[self.handle_radius, self.handle_radius, self.handle_length / 2.0],
                    pos=(0, 0, 0),
                    rgba=self.rgba_handle,
                    group=1,
                    density=str(self.handle_density),
                    friction=array_to_string((self.handle_friction, 0.005, 0.0001)),
                )
            )
        else:
            # Raise error
            raise ValueError("Error loading hammer: Handle type must either be 'box' or 'cylinder', got {}.".format(
                self.handle_shape
            ))

        # Define head and append to the main body
        main_body.append(
            new_geom(
                geom_type="box",
                name="hammer_head",
                size=[self.head_halfsize * 2, self.head_halfsize, self.head_halfsize],
                pos=(0, 0, self.handle_length / 2.0 + self.head_halfsize),
                rgba=self.rgba_head,
                group=1,
                density=str(self.handle_density * self.head_density_ratio),
            )
        )

        # Define face (and neck) and append to the main body
        main_body.append(
            new_geom(
                geom_type="cylinder",
                name="hammer_neck",
                size=[self.head_halfsize * 0.8, self.head_halfsize * 0.2],
                pos=(self.head_halfsize * 2.2, 0, self.handle_length / 2.0 + self.head_halfsize),
                quat=array_to_string([0.707106, 0, 0.707106, 0]),
                rgba=self.rgba_face,
                group=1,
                density=str(self.handle_density * self.head_density_ratio),
            )
        )
        main_body.append(
            new_geom(
                geom_type="cylinder",
                name="hammer_face",
                size=[self.head_halfsize, self.head_halfsize * 0.4],
                pos=(self.head_halfsize * 2.8, 0, self.handle_length / 2.0 + self.head_halfsize),
                quat=array_to_string([0.707106, 0, 0.707106, 0]),
                rgba=self.rgba_face,
                group=1,
                density=str(self.handle_density * self.head_density_ratio),
            )
        )

        # Define claw and append to the main body
        main_body.append(
            new_geom(
                geom_type="box",
                name="hammer_claw",
                size=[self.head_halfsize * 0.7072, self.head_halfsize * 0.95, self.head_halfsize * 0.7072],
                pos=(-self.head_halfsize * 2, 0, self.handle_length / 2.0 + self.head_halfsize),
                quat=array_to_string([0.9238795, 0, 0.3826834, 0]),
                rgba=self.rgba_claw,
                group=1,
                density=str(self.handle_density * self.head_density_ratio),
            )
        )

        return main_body

    @property
    def init_quat(self):
        # Randomly sample between +/- flip (such that the hammer head faces one way or the other)
        return np.array([0.5, -0.5, 0.5, -0.5]) if np.random.rand() >= 0.5 else np.array([-0.5, -0.5, -0.5, -0.5])

    @property
    def handle_geoms(self):
        return ["hammer_handle"]

    @property
    def head_geoms(self):
        return ["hammer_head"]

    @property
    def face_geoms(self):
        return ["hammer_neck", "hammer_face"]

    @property
    def claw_geoms(self):
        return ["hammer_claw"]

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class PotWithHandlesObject(MujocoGeneratedObject):
    """
    Generates the Pot object with side handles (used in BaxterLift)
    """

    def __init__(
        self,
        name,
        body_half_size=None,
        handle_radius=0.01,
        handle_length=0.09,
        handle_width=0.09,
        rgba_body=None,
        rgba_handle_1=None,
        rgba_handle_2=None,
        solid_handle=False,
        thickness=0.025,  # For body
        joints=None,
    ):
        super().__init__(name=name, joints=joints)
        if body_half_size:
            self.body_half_size = body_half_size
        else:
            self.body_half_size = np.array([0.07, 0.07, 0.07])
        self.thickness = thickness
        self.handle_radius = handle_radius
        self.handle_length = handle_length
        self.handle_width = handle_width
        if rgba_body:
            self.rgba_body = np.array(rgba_body)
        else:
            self.rgba_body = RED
        if rgba_handle_1:
            self.rgba_handle_1 = np.array(rgba_handle_1)
        else:
            self.rgba_handle_1 = GREEN
        if rgba_handle_2:
            self.rgba_handle_2 = np.array(rgba_handle_2)
        else:
            self.rgba_handle_2 = BLUE
        self.solid_handle = solid_handle

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.body_half_size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.body_half_size[2]])

    def get_horizontal_radius(self):
        return np.sqrt(2) * (max(self.body_half_size) + self.handle_length)

    @property
    def handle_distance(self):
        return self.body_half_size[1] * 2 + self.handle_length * 2

    def get_collision(self, site=None):
        main_body = new_body()
        main_body.set("name", self.name)

        for geom in five_sided_box(
            self.body_half_size, self.rgba_body, 1, self.thickness
        ):
            main_body.append(geom)
        handle_z = self.body_half_size[2] - self.handle_radius
        handle_1_center = [0, self.body_half_size[1] + self.handle_length, handle_z]
        handle_2_center = [
            0,
            -1 * (self.body_half_size[1] + self.handle_length),
            handle_z,
        ]
        # the bar on handle horizontal to body
        main_bar_size = [
            self.handle_width / 2 + self.handle_radius,
            self.handle_radius,
            self.handle_radius,
        ]
        side_bar_size = [self.handle_radius, self.handle_length / 2, self.handle_radius]
        handle_1 = new_body(name="handle_1")
        if self.solid_handle:
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1",
                    pos=[0, self.body_half_size[1] + self.handle_length / 2, handle_z],
                    size=[
                        self.handle_width / 2,
                        self.handle_length / 2,
                        self.handle_radius,
                    ],
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
        else:
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_c",
                    pos=handle_1_center,
                    size=main_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_+",  # + for positive x
                    pos=[
                        self.handle_width / 2,
                        self.body_half_size[1] + self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_-",
                    pos=[
                        -self.handle_width / 2,
                        self.body_half_size[1] + self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )

        handle_2 = new_body(name="handle_2")
        if self.solid_handle:
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2",
                    pos=[0, -self.body_half_size[1] - self.handle_length / 2, handle_z],
                    size=[
                        self.handle_width / 2,
                        self.handle_length / 2,
                        self.handle_radius,
                    ],
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
        else:
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_c",
                    pos=handle_2_center,
                    size=main_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_+",  # + for positive x
                    pos=[
                        self.handle_width / 2,
                        -self.body_half_size[1] - self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_-",
                    pos=[
                        -self.handle_width / 2,
                        -self.body_half_size[1] - self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )

        main_body.append(handle_1)
        main_body.append(handle_2)
        main_body.append(
            new_site(
                name="pot_handle_1",
                rgba=self.rgba_handle_1,
                pos=handle_1_center - np.array([0, 0.005, 0]),
                size=[0.005],
            )
        )
        main_body.append(
            new_site(
                name="pot_handle_2",
                rgba=self.rgba_handle_2,
                pos=handle_2_center + np.array([0, 0.005, 0]),
                size=[0.005],
            )
        )
        main_body.append(new_site(name="pot_center", pos=[0, 0, 0], rgba=[1, 0, 0, 0]))

        return main_body

    def handle_geoms(self):
        return self.handle_1_geoms() + self.handle_2_geoms()

    def handle_1_geoms(self):
        if self.solid_handle:
            return ["handle_1"]
        return ["handle_1_c", "handle_1_+", "handle_1_-"]

    def handle_2_geoms(self):
        if self.solid_handle:
            return ["handle_2"]
        return ["handle_2_c", "handle_2_+", "handle_2_-"]

    def get_visual(self, site=None):
        return self.get_collision(site)


def five_sided_box(size, rgba, group, thickness):
    """
    Args:
        size ([float,flat,float]):
        rgba ([float,float,float,float]): color
        group (int): Mujoco group
        thickness (float): wall thickness

    Returns:
        []: array of geoms corresponding to the
            5 sides of the pot used in BaxterLift
    """
    geoms = []
    x, y, z = size
    r = thickness / 2
    geoms.append(
        new_geom(
            geom_type="box", size=[x, y, r], pos=[0, 0, -z + r], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[x, r, z], pos=[0, -y + r, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[x, r, z], pos=[0, y - r, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[r, y, z], pos=[x - r, 0, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[r, y, z], pos=[-x + r, 0, 0], rgba=rgba, group=group
        )
    )
    return geoms


DEFAULT_DENSITY_RANGE = [200, 500, 1000, 3000, 5000]
DEFAULT_FRICTION_RANGE = [0.25, 0.5, 1, 1.5, 2]


def _get_size(size,
              size_max,
              size_min,
              default_max,
              default_min):
    """
        Helper method for providing a size,
        or a range to randomize from
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
    return size


def _get_randomized_range(val,
                          provided_range,
                          default_range):
    """
        Helper to initialize by either value or a range
        Returns a range to randomize from
    """
    if val is None:
        if provided_range is None:
            return default_range
        else:
            return provided_range
    else:
        if provided_range is not None:
            raise ValueError('Value {} overrides range {}'
                             .format(str(val), str(provided_range)))
        return [val]


class BoxObject(MujocoGeneratedObject):
    """
    An object that is a box
    """

    def __init__(
        self,
        name,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        add_material=False,
        joints=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07, 0.07],
                         [0.03, 0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            name=name,
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction_range=friction_range,
            add_material=add_material,
            joints=joints,
        )

    def sanity_check(self):
        assert len(self.size) == 3, "box size should have length 3"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2)

    # returns a copy, Returns xml body node
    def get_collision(self, site=False):
        return self._get_collision(site=site, ob_type="box")

    # returns a copy, Returns xml body node
    def get_visual(self, site=False):
        return self._get_visual(site=site, ob_type="box")


class CylinderObject(MujocoGeneratedObject):
    """
    A randomized cylinder object.
    """

    def __init__(
        self,
        name,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        add_material=False,
        joints=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07],
                         [0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            name=name,
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction_range=friction_range,
            add_material=add_material,
            joints=joints,
        )

    def sanity_check(self):
        assert len(self.size) == 2, "cylinder size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[1]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[1]])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self, site=False):
        return self._get_collision(site=site, ob_type="cylinder")

    # returns a copy, Returns xml body node
    def get_visual(self, site=False):
        return self._get_visual(site=site, ob_type="cylinder")


class BallObject(MujocoGeneratedObject):
    """
    A randomized ball (sphere) object.
    """

    def __init__(
        self,
        name,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        add_material=False,
        joints=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07],
                         [0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            name=name,
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction_range=friction_range,
            add_material=add_material,
            joints=joints,
        )

    def sanity_check(self):
        assert len(self.size) == 1, "ball size should have length 1"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[0]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[0]])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self, site=False):
        return self._get_collision(site=site, ob_type="sphere")

    # returns a copy, Returns xml body node
    def get_visual(self, site=False):
        return self._get_visual(site=site, ob_type="sphere")


class CapsuleObject(MujocoGeneratedObject):
    """
    A randomized capsule object.
    """

    def __init__(
        self,
        name,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        add_material=False,
        joints=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07],
                         [0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            name=name,
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction_range=friction_range,
            add_material=add_material,
            joints=joints,
        )

    def sanity_check(self):
        assert len(self.size) == 2, "capsule size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * (self.size[0] + self.size[1])])

    def get_top_offset(self):
        return np.array([0, 0, (self.size[0] + self.size[1])])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self, site=False):
        return self._get_collision(site=site, ob_type="capsule")

    # returns a copy, Returns xml body node
    def get_visual(self, site=False):
        return self._get_visual(site=site, ob_type="capsule")
