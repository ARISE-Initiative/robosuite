import collections
import numpy as np

from copy import deepcopy

from robosuite.utils import RandomizationError
from robosuite.utils.transform_utils import quat_multiply


class ObjectPositionSampler:
    """Base class of object placement sampler."""

    def __init__(self):
        pass

    def setup(self, mujoco_objects, table_top_offset, table_size):
        """
        Args:
            mujoco_objects (OrderedDict): dictionary of MujocoObjects to place
            table_top_offset (float * 3): location of table top center
            table_size (float * 3): x,y,z-FULLsize of the table
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
            xpos ((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat ((float * 4) * n_obj): quaternion of the objects
        """
        raise NotImplementedError


class UniformRandomSampler(ObjectPositionSampler):
    """Places all objects within the table uniformly random."""

    def __init__(
        self,
        x_range=None,
        y_range=None,
        ensure_object_boundary_in_range=True,
        rotation=None,
        rotation_axis='z',
        z_offset=0.,
    ):
        """
        Args:
            x_range (float * 2): override the x_range used to uniformly place objects
                    if None, default to x-range of table

            y_range (float * 2): override the y_range used to uniformly place objects
                    if None default to y-range of table

            x_range and y_range are both with respect to (0,0) = center of table.

            ensure_object_boundary_in_range (bool):
                True: The center of object is at position:
                     [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
                False: 
                    [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

            rotation:
                None: Add uniform random random rotation
                iterable (a,b): Uniformly randomize rotation angle between a and b (in radians)
                value: Add fixed angle rotation

            rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

            z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
                that do not move (i.e. no free joint) to place them above the table.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.z_offset = z_offset

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
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == 'x':
            return [np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0]
        elif self.rotation_axis == 'y':
            return [np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0]
        elif self.rotation_axis == 'z':
            return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
        else:
            # Invalid axis specified, raise error
            raise ValueError("Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(self.rotation_axis))

    def sample(self, fixtures=None, return_placements=False, reference_object_name=None, sample_on_top=False):
        """
        Uniformly sample on a surface (not necessarily table surface).

        Args:
            fixtures (dict): current dictionary of object placements in the scene. Used to make sure
                generated placements are valid.

            return_placements (bool): if True, return the updated dictionary
                of object placements.

            reference_object_name (str): if provided, sample placement relative to this object's
                placement (which must be provided in @fixtures).

            sample_on_top (bool): if True, sample placement on top of the reference object.

        Return:
            pos_arr (list): list of placed object positions

            quat_arr (list): list of placed object quaternions

            placements (dict): if @return_placements is True, returns a dictionary of all
                object placements, including the ones placed by this sampler.
        """
        pos_arr = []
        quat_arr = []

        if fixtures is None:
            placed_objects = {}
        else:
            placed_objects = deepcopy(fixtures)

        # compute reference position
        base_offset = self.table_top_offset
        if reference_object_name is not None:
            assert reference_object_name in placed_objects
            reference_pos, reference_mjcf = placed_objects[reference_object_name]
            base_offset[:2] = reference_pos[:2]
            if sample_on_top:
                base_offset[-1] = reference_pos[-1] + reference_mjcf.get_top_offset()[-1]  # set surface z

        index = 0
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            success = False
            for i in range(5000):  # 5000 retries
                object_x = self.sample_x(horizontal_radius) + base_offset[0]
                object_y = self.sample_y(horizontal_radius) + base_offset[1]
                object_z = base_offset[2] + self.z_offset - bottom_offset[-1]

                # objects cannot overlap
                location_valid = True
                for (x, y, z), other_obj_mjcf in placed_objects.values():
                    if (
                        np.linalg.norm([object_x - x, object_y - y], 2)
                        <= other_obj_mjcf.get_horizontal_radius() + horizontal_radius
                    ) and (
                        object_z - z <= other_obj_mjcf.get_top_offset()[-1] - bottom_offset[-1]
                    ):
                        location_valid = False
                        break

                if location_valid:
                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj_name] = (pos, obj_mjcf)

                    # random z-rotation
                    quat = self.sample_quat()

                    # multiply this quat by the object's initial rotation if it has the attribute specified
                    if hasattr(obj_mjcf, "init_quat"):
                        quat = quat_multiply(quat, obj_mjcf.init_quat)

                    quat_arr.append(quat)
                    pos_arr.append(pos)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects on the desk")
            index += 1

        if return_placements:
            return pos_arr, quat_arr, placed_objects
        return pos_arr, quat_arr


class SequentialCompositeSampler(ObjectPositionSampler):
    """
    Samples position for each object sequentially. Allows chaining
    multiple placement initializers together - so that object locations can
    be sampled on top of other objects or relative to other object placements.
    """
    def __init__(self):
        self.mujoco_objects = None
        self.samplers = collections.OrderedDict()
        self.table_top_offset = None
        self.table_size = None
        self.n_obj = None

    def append_sampler(self, object_name, sampler, **kwargs):
        assert object_name not in self.samplers
        self.samplers[object_name] = {'sampler': sampler, 'object_names': [object_name], 'sample_kwargs': kwargs}

    def hide(self, object_name):
        """
        Helper method to remove an object from the workspace.
        """
        sampler = UniformRandomSampler(
            x_range=[-10, -20],
            y_range=[-10, -20],
            rotation=[0, 0],
            rotation_axis='z',
            z_offset=10,
            ensure_object_boundary_in_range=False
        )
        self.append_sampler(object_name=object_name, sampler=sampler)

    def _sample_on_top(self, object_name, surface_name, sampler):
        if surface_name == 'table':
            self.append_sampler(object_name=object_name, sampler=sampler)
        else:
            assert surface_name in self.samplers  # surface needs to be placed first
            self.append_sampler(
                object_name=object_name,
                sampler=sampler,
                reference_object_name=surface_name,
                sample_on_top=True
            )

    def sample_on_top(
        self,
        object_name,
        surface_name='table',
        x_range=None,
        y_range=None,
        rotation=None,
        rotation_axis='z',
        z_offset=0.0,
        ensure_object_boundary_in_range=True
    ):
        """
        Sample placement on top of a surface object.
        """
        sampler = UniformRandomSampler(
            x_range=x_range,
            y_range=y_range,
            rotation=rotation,
            rotation_axis=rotation_axis,
            z_offset=z_offset,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range
        )
        return self._sample_on_top(object_name, surface_name, sampler)

    def setup(self, mujoco_objects, table_top_offset, table_size):
        """
        Overrides super implementation so that we can setup all placement
        initializers we own.
        """
        self.mujoco_objects = mujoco_objects
        assert(isinstance(mujoco_objects, collections.OrderedDict))
        self.table_top_offset = table_top_offset
        self.table_size = table_size
        self.n_obj = len(self.mujoco_objects)

        for object_name, sampler_config in self.samplers.items():
            object_names = sampler_config['object_names']
            sampler = sampler_config['sampler']
            objs = collections.OrderedDict((o, mujoco_objects[o]) for o in object_names)
            sampler.setup(mujoco_objects=objs, table_top_offset=table_top_offset, table_size=table_size)

    def sample(self, fixtures=None, return_placements=False):
        """
        Sample from each placement initializer sequentially, in the order
        that they were appended.

        Args:
            fixtures (dict): current dictionary of object placements in the scene. Used to make sure
                generated placements are valid.

            return_placements (bool): if True, return the updated dictionary
                of object placements.

        Return:
            pos_arr (list): list of placed object positions

            quat_arr (list): list of placed object quaternions

            placements (dict): if @return_placements is True, returns a dictionary of all
                object placements, including the ones placed by this sampler.
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
            pos_arr, quat_arr, new_placements = \
                sampler['sampler'].sample(fixtures=placements, return_placements=True, **sampler["sample_kwargs"])
            named_samples[obj_name] = (pos_arr[0], quat_arr[0])
            placements.update(new_placements)

        all_pos_arr = [p[0] for p in named_samples.values()]
        all_quat_arr = [p[1] for p in named_samples.values()]

        if return_placements:
            return all_pos_arr, all_quat_arr, placements
        else:
            return all_pos_arr, all_quat_arr
