import collections
import numpy as np

from robosuite.utils import RandomizationError


class ObjectPositionSampler:
    """Base class of object placement sampler."""

    def __init__(self):
        pass

    def setup(self, mujoco_objects, table_top_offset, table_size):
        """
        Args:
            Mujoco_objcts(MujocoObject * n_obj): object to be placed
            table_top_offset(float * 3): location of table top center
            table_size(float * 3): x,y,z-FULLsize of the table
        """
        self.mujoco_objects = mujoco_objects
        self.n_obj = len(self.mujoco_objects)
        self.table_top_offset = table_top_offset
        self.table_size = table_size

    def sample(self):
        """
        Args:
            object_index: index of the current object being sampled
        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        raise NotImplementedError


class UniformRandomSampler(ObjectPositionSampler):
    """Places all objects within the table uniformly random."""

    def __init__(
        self,
        x_range=None,
        y_range=None,
        ensure_object_boundary_in_range=True,
        z_rotation="random",
    ):
        """
        Args:
            x_range(float * 2): override the x_range used to uniformly place objects
                    if None, default to x-range of table
            y_range(float * 2): override the y_range used to uniformly place objects
                    if None default to y-range of table
            x_range and y_range are both with respect to (0,0) = center of table.
            ensure_object_boundary_in_range:
                True: The center of object is at position:
                     [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
                False: 
                    [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
            z_rotation:
                None: Add uniform random random z-rotation
                iterable (a,b): Uniformly randomize rotation angle between a and b (in radians)
                value: Add fixed angle z-rotation
        """
        self.x_range = x_range
        self.y_range = y_range
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.z_rotation = z_rotation

    def sample_x(self, object_horizontal_radius):
        x_range = self.x_range
        if x_range is None:
            x_range = [-self.table_size[0] / 2, self.table_size[0] / 2]
        minimum = min(x_range)
        maximum = max(x_range)
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_y(self, object_horizontal_radius):
        y_range = self.y_range
        if y_range is None:
            y_range = [-self.table_size[0] / 2, self.table_size[0] / 2]
        minimum = min(y_range)
        maximum = max(y_range)
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_quat(self):
        if self.z_rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.z_rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.z_rotation), low=min(self.z_rotation)
            )
        else:
            rot_angle = self.z_rotation

        return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]

    def sample(self):
        pos_arr = []
        quat_arr = []
        placed_objects = []
        index = 0
        for obj_mjcf in self.mujoco_objects:
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            success = False
            for i in range(5000):  # 1000 retries
                object_x = self.sample_x(horizontal_radius)
                object_y = self.sample_y(horizontal_radius)
                # objects cannot overlap
                location_valid = True
                for x, y, r in placed_objects:
                    if (
                        np.linalg.norm([object_x - x, object_y - y], 2)
                        <= r + horizontal_radius
                    ):
                        location_valid = False
                        break
                if location_valid:
                    # location is valid, put the object down
                    pos = (
                        self.table_top_offset
                        - bottom_offset
                        + np.array([object_x, object_y, 0])
                    )
                    placed_objects.append((object_x, object_y, horizontal_radius))
                    # random z-rotation

                    quat = self.sample_quat()

                    quat_arr.append(quat)
                    pos_arr.append(pos)
                    success = True
                    break

                # bad luck, reroll
            if not success:
                raise RandomizationError("Cannot place all objects on the desk")
            index += 1
        return pos_arr, quat_arr


class UniformRandomPegsSampler(ObjectPositionSampler):
    """Places all objects on top of the table uniformly random."""

    def __init__(
        self,
        x_range=None,
        y_range=None,
        z_range=None,
        ensure_object_boundary_in_range=True,
        z_rotation=True,
    ):
        """
        Args:
            x_range(float * 2): override the x_range used to uniformly place objects
                    if None, default to x-range of table
            y_range(float * 2): override the y_range used to uniformly place objects
                    if None default to y-range of table
            x_range and y_range are both with respect to (0,0) = center of table.
            ensure_object_boundary_in_range:
                True: The center of object is at position:
                     [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
                False: 
                    [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
            z_rotation:
                Add random z-rotation
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.z_rotation = z_rotation

    def sample_x(self, object_horizontal_radius, x_range=None):
        if x_range is None:
            x_range = self.x_range
            if x_range is None:
                x_range = [-self.table_size[0] / 2, self.table_size[0] / 2]
        minimum = min(x_range)
        maximum = max(x_range)
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_y(self, object_horizontal_radius, y_range=None):
        if y_range is None:
            y_range = self.y_range
            if y_range is None:
                y_range = [-self.table_size[0] / 2, self.table_size[0] / 2]
        minimum = min(y_range)
        maximum = max(y_range)
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_z(self, object_horizontal_radius, z_range=None):
        if z_range is None:
            z_range = self.z_range
            if z_range is None:
                z_range = [0, 1]
        minimum = min(z_range)
        maximum = max(z_range)
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_quat(self):
        if self.z_rotation:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
            return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
        else:
            return [1, 0, 0, 0]

    def sample(self):
        pos_arr = []
        quat_arr = []
        placed_objects = []

        for obj_name, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            success = False
            for i in range(5000):  # 1000 retries
                if obj_name.startswith("SquareNut"):
                    x_range = [
                        -self.table_size[0] / 2 + horizontal_radius,
                        -horizontal_radius,
                    ]
                    y_range = [horizontal_radius, self.table_size[0] / 2]
                else:
                    x_range = [
                        -self.table_size[0] / 2 + horizontal_radius,
                        -horizontal_radius,
                    ]
                    y_range = [-self.table_size[0] / 2, -horizontal_radius]
                object_x = self.sample_x(horizontal_radius, x_range=x_range)
                object_y = self.sample_y(horizontal_radius, y_range=y_range)
                object_z = self.sample_z(0.01)
                # objects cannot overlap
                location_valid = True
                pos = (
                    self.table_top_offset
                    - bottom_offset
                    + np.array([object_x, object_y, object_z])
                )

                for pos2, r in placed_objects:
                    if (
                        np.linalg.norm(pos - pos2, 2) <= r + horizontal_radius
                        and abs(pos[2] - pos2[2]) < 0.021
                    ):
                        location_valid = False
                        break
                if location_valid:
                    # location is valid, put the object down
                    placed_objects.append((pos, horizontal_radius))
                    # random z-rotation

                    quat = self.sample_quat()

                    quat_arr.append(quat)
                    pos_arr.append(pos)
                    success = True
                    break

                # bad luck, reroll
            if not success:
                raise RandomizationError("Cannot place all objects on the desk")

        return pos_arr, quat_arr

    def setup(self, mujoco_objects, table_top_offset, table_size):
        """
        Note: overrides superclass implementation.

        Args:
            Mujoco_objcts(MujocoObject * n_obj): object to be placed
            table_top_offset(float * 3): location of table top center
            table_size(float * 3): x,y,z-FULLsize of the table
        """
        self.mujoco_objects = mujoco_objects  # should be a dictionary - (name, mjcf)
        self.n_obj = len(self.mujoco_objects)
        self.table_top_offset = table_top_offset
        self.table_size = table_size
