import collections
import numpy as np

from robosuite.utils import RandomizationError
from copy import deepcopy


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
        assert isinstance(self.mujoco_objects, collections.OrderedDict)
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
        z_rotation=None,
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
        elif isinstance(self.z_rotation, collections.abc.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.z_rotation), low=min(self.z_rotation)
            )
        else:
            rot_angle = self.z_rotation

        return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]

    def sample(self, fixtures=None, return_placements=False):
        pos_arr = []
        quat_arr = []
        if fixtures is None:
            placed_objects = {}
        else:
            placed_objects = deepcopy(fixtures)

        index = 0
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            success = False
            for i in range(5000):  # 1000 retries
                object_x = self.sample_x(horizontal_radius)
                object_y = self.sample_y(horizontal_radius)
                # objects cannot overlap
                location_valid = True
                for x, y, r in placed_objects.values():
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
                    placed_objects[obj_name] = (object_x, object_y, horizontal_radius)
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
        if return_placements:
            return pos_arr, quat_arr, placed_objects
        else:
            return pos_arr, quat_arr


class UniformRandomPegsSampler(ObjectPositionSampler):
    """Places all objects on top of the table uniformly random."""

    def __init__(
        self,
        x_range=None,
        y_range=None,
        z_range=None,
        ensure_object_boundary_in_range=True,
        z_rotation=None,
    ):
        """
        Args:
            x_range { object : (float * 2) }: override the x_range used to uniformly place objects
                    if None, defaults to a pre-specified range per object type
            y_range { object : (float * 2) }: override the y_range used to uniformly place objects
                    if None defaults to a pre-specified range per object type
            x_range and y_range are both with respect to (0,0) = center of table.
            ensure_object_boundary_in_range:
                True: The center of object is at position:
                     [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
                False:
                    [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
            z_rotation:
                Add random z-rotation
        """

        # NOTE: defaults for these ranges will be set in @setup if necessary
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.z_rotation = z_rotation

    def sample_x(self, object_name, object_horizontal_radius):
        if object_name.startswith("SquareNut"):
            k = "SquareNut"
        else:
            k = "RoundNut"
        minimum = min(self.x_range[k])
        maximum = max(self.x_range[k])
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_y(self, object_name, object_horizontal_radius):
        if object_name.startswith("SquareNut"):
            k = "SquareNut"
        else:
            k = "RoundNut"
        minimum = min(self.y_range[k])
        maximum = max(self.y_range[k])
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_z(self, object_name, object_horizontal_radius):
        if object_name.startswith("SquareNut"):
            k = "SquareNut"
        else:
            k = "RoundNut"
        minimum = min(self.z_range[k])
        maximum = max(self.z_range[k])
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_quat(self, object_name):
        if object_name.startswith("SquareNut"):
            k = "SquareNut"
        else:
            k = "RoundNut"
        if self.z_rotation is None or self.z_rotation[k] is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.z_rotation[k], collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.z_rotation[k]), low=min(self.z_rotation[k])
            )
        else:
            rot_angle = self.z_rotation[k]

        return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]

    def sample(self, fixtures=None, return_placements=False):
        pos_arr = []
        quat_arr = []
        if fixtures is None:
            placed_objects = {}
        else:
            placed_objects = deepcopy(fixtures)

        for obj_name, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            success = False

            for i in range(5000):  # 5000 retries
                object_x = self.sample_x(obj_name, horizontal_radius)
                object_y = self.sample_y(obj_name, horizontal_radius)
                object_z = self.sample_z(obj_name, 0.01)
                # objects cannot overlap
                location_valid = True
                pos = (
                    self.table_top_offset
                    - bottom_offset
                    + np.array([object_x, object_y, object_z])
                )

                for pos2, r in placed_objects.values():
                    if (
                        np.linalg.norm(pos - pos2, 2) <= r + horizontal_radius
                        and abs(pos[2] - pos2[2]) < 0.021
                    ):
                        location_valid = False
                        break
                if location_valid:
                    # location is valid, put the object down
                    placed_objects[obj_name] = (pos, horizontal_radius)
                    # random z-rotation

                    quat = self.sample_quat(obj_name)

                    quat_arr.append(quat)
                    pos_arr.append(pos)
                    success = True
                    break

                # bad luck, reroll
            if not success:
                raise RandomizationError("Cannot place all objects on the desk")

        if return_placements:
            return pos_arr, quat_arr, placed_objects
        else:
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

        # future proof: make sure all objects of same type have same size
        all_horizontal_radius = {}
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()

            if obj_name.startswith("SquareNut"):
                if "SquareNut" in all_horizontal_radius:
                    assert(all_horizontal_radius["SquareNut"] == horizontal_radius)
                all_horizontal_radius["SquareNut"] = horizontal_radius
            elif obj_name.startswith("RoundNut"):
                if "RoundNut" in all_horizontal_radius:
                    assert(all_horizontal_radius["RoundNut"] == horizontal_radius)
                all_horizontal_radius["RoundNut"] = horizontal_radius
            else:
                raise Exception("Got invalid object to place!")

        # set defaults if necessary
        if self.x_range is None:
            self.x_range = {
                "SquareNut": [
                    -self.table_size[0] / 2 + all_horizontal_radius["SquareNut"],
                    -all_horizontal_radius["SquareNut"],
                ],
                "RoundNut": [
                    -self.table_size[0] / 2 + all_horizontal_radius["RoundNut"],
                    -all_horizontal_radius["RoundNut"],
                ],
            }
        if self.y_range is None:
            self.y_range = {
                "SquareNut": [
                    all_horizontal_radius["SquareNut"],
                    self.table_size[0] / 2,
                ],
                "RoundNut": [
                    -self.table_size[0] / 2,
                    -all_horizontal_radius["RoundNut"]
                ],
            }
        if self.z_range is None:
            self.z_range = {
                "SquareNut": [0., 1.],
                "RoundNut": [0., 1.],
            }


class SequentialCompositeSampler(ObjectPositionSampler):
    """Samples position for each object sequentially. """
    def __init__(self):
        self.mujoco_objects = None
        self.samplers = collections.OrderedDict()
        self.table_top_offset = None
        self.table_size = None
        self.n_obj = None

    def append_sampler(self, object_name, sampler, object_names=None):
        assert object_name not in self.samplers
        self.samplers[object_name] = {'sampler': sampler, 'object_names': [object_name]}
        if object_names is not None:
            self.samplers[object_name]['object_names'] = object_names

    def sample_on_top(
            self,
            object_name,
            surface_name='table',
            x_range=None,
            y_range=None,
            z_rotation=None,
            ensure_object_boundary_in_range=True
    ):
        """Sample placement on top of a surface object"""
        if surface_name == 'table':
            self.append_sampler(
                object_name,
                UniformRandomSampler(
                    x_range=x_range,
                    y_range=y_range,
                    z_rotation=z_rotation,
                    ensure_object_boundary_in_range=ensure_object_boundary_in_range
                )
            )
        else:
            assert surface_name in self.samplers  # surface needs to be placed first
            raise NotImplementedError

    def setup(self, mujoco_objects, table_top_offset, table_size):
        self.mujoco_objects = mujoco_objects
        assert(isinstance(mujoco_objects, collections.OrderedDict))
        self.table_top_offset = table_top_offset
        self.table_size = table_size
        self.n_obj = len(self.mujoco_objects)
        for object_name, sampler_config in self.samplers.items():
            object_names = sampler_config['object_names']
            objs = collections.OrderedDict((o, mujoco_objects[o]) for o in object_names)
            sampler_config['sampler'].setup(
                mujoco_objects=objs, table_top_offset=table_top_offset, table_size=table_size)

    def sample(self, fixtures=None, return_placements=False):
        """
        Run samplers according to the appending order.
        Return samples based on the ordering of self.mujoco_objects
        :param fixtures: a dictionary of object_name: (x, y, r) for the objects that have already been placed
        :param return_placements: return the accumulated placements in the scene
        :return: a list of position and quaternion placements in the order of self.mujoco_objects.
        """
        if fixtures is None:
            placements = {}
        else:
            placements = deepcopy(fixtures)
        # make sure all objects have samplers specified
        named_samples = collections.OrderedDict()
        for k in self.mujoco_objects:
            assert k in self.samplers
            named_samples[k] = None
        for obj_name, sampler in self.samplers.items():
            pos_arr, quat_arr, new_placements = sampler['sampler'].sample(fixtures=placements, return_placements=True)
            named_samples[obj_name] = (pos_arr[0], quat_arr[0])
            placements.update(new_placements)

        all_pos_arr = [p[0] for p in named_samples.values()]
        all_quat_arr = [p[1] for p in named_samples.values()]

        if return_placements:
            return all_pos_arr, all_quat_arr, placements
        else:
            return all_pos_arr, all_quat_arr


class RoundRobinSampler(UniformRandomSampler):
    """Places all objects according to grid and round robin between grid points."""

    def __init__(
        self,
        x_range=None,
        y_range=None,
        ensure_object_boundary_in_range=True,
        z_rotation=None,
    ):
        # x_range, y_range, and z_rotation should all be lists of values to rotate between
        assert(len(x_range) == len(y_range))
        assert(len(z_rotation) == len(y_range))
        self._counter = 0
        self.num_grid = len(x_range)

        super(RoundRobinSampler, self).__init__(
            x_range=x_range,
            y_range=y_range,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            z_rotation=z_rotation,
        )

    @property
    def counter(self):
        return self._counter

    def increment_counter(self):
        """
        Useful for moving on to next placement in the grid.
        """
        self._counter = (self._counter + 1) % self.num_grid

    def decrement_counter(self):
        """
        Useful to reverting to the last placement in the grid.
        """
        self._counter -= 1
        if self._counter < 0:
            self._counter = self.num_grid - 1

    def sample_x(self, object_horizontal_radius):
        minimum = self.x_range[self._counter]
        maximum = self.x_range[self._counter]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_y(self, object_horizontal_radius):
        minimum = self.y_range[self._counter]
        maximum = self.y_range[self._counter]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_quat(self):
        rot_angle = self.z_rotation[self._counter]
        return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]


class RoundRobinPegsSampler(UniformRandomPegsSampler):
    """Places all objects according to grid and round robin between grid points."""

    def __init__(
        self,
        x_range=None,
        y_range=None,
        z_range=None,
        ensure_object_boundary_in_range=True,
        z_rotation=None,
    ):
        for k in x_range:
            # x_range, y_range, and z_rotation should all be lists of values to rotate between
            assert(len(x_range[k]) == len(y_range[k]))
            assert(len(z_rotation[k]) == len(y_range[k]))
            assert(len(z_range[k]) == len(y_range[k]))
        self._counter = 0
        self.num_grid = len(x_range[k])

        super(RoundRobinPegsSampler, self).__init__(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            z_rotation=z_rotation,
        )

    def increment_counter(self):
        """
        Useful for moving on to next placement in the grid.
        """
        self._counter = (self._counter + 1) % self.num_grid

    def decrement_counter(self):
        """
        Useful to reverting to the last placement in the grid.
        """
        self._counter -= 1
        if self._counter < 0:
            self._counter = self.num_grid - 1

    def sample_x(self, object_name, object_horizontal_radius):
        if object_name.startswith("SquareNut"):
            k = "SquareNut"
        else:
            k = "RoundNut"
        minimum = self.x_range[k][self._counter]
        maximum = self.x_range[k][self._counter]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_y(self, object_name, object_horizontal_radius):
        if object_name.startswith("SquareNut"):
            k = "SquareNut"
        else:
            k = "RoundNut"
        minimum = self.y_range[k][self._counter]
        maximum = self.y_range[k][self._counter]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_z(self, object_name, object_horizontal_radius):
        if object_name.startswith("SquareNut"):
            k = "SquareNut"
        else:
            k = "RoundNut"
        minimum = self.z_range[k][self._counter]
        maximum = self.z_range[k][self._counter]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def sample_quat(self, object_name):
        if object_name.startswith("SquareNut"):
            k = "SquareNut"
        else:
            k = "RoundNut"
        rot_angle = self.z_rotation[k][self._counter]
        return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
