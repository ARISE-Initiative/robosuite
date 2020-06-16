from collections import OrderedDict
import numpy as np

from robosuite.models.tasks import Task
from robosuite.utils import RandomizationError
from robosuite.utils.mjcf_utils import new_joint, array_to_string, string_to_array


class PickPlaceTask(Task):
    """
    Creates MJCF model of a pick-and-place task.

    A pick-and-place task consists of one robot picking objects from a bin
    and placing them into another bin. This class combines the robot, the
    arena, and the objects into a single MJCF model of the task.
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects, visual_objects, initializer):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            mujoco_objects: a list of MJCF models of physical objects
            visual_objects: a list of MJCF models of visual objects. Visual
                objects are excluded from physical computation, we use them to
                indicate the target destinations of the objects.
            initializer: placement sampler to initialize object positions.
        """
        super().__init__()

        # temp: z-rotation
        self.z_rotation = True

        self.object_metadata = []
        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)
        self.merge_visual(OrderedDict(visual_objects))
        self.visual_objects = visual_objects

        self.initializer = initializer
        self.initializer.setup(self.mujoco_objects, self.bin_offset, self.bin_size)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.bin_offset = mujoco_arena.bin_abs
        self.bin_size = mujoco_arena.table_full_size
        self.bin2_body = mujoco_arena.bin2_body
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.n_objects = len(mujoco_objects)
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.max_horizontal_radius = 0
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free", damping="0.0005"))
            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def merge_visual(self, mujoco_objects):
        """Adds visual objects to the MJCF model."""
        self.visual_obj_mjcf = []
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_visual(name=obj_name, site=False)
            self.visual_obj_mjcf.append(obj)
            self.worldbody.append(obj)

    def sample_quat(self):
        """Samples quaternions of random rotations along the z-axis."""
        if self.z_rotation:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
            return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
        return [1, 0, 0, 0]

    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""
        pos_arr, quat_arr = self.initializer.sample()
        for i in range(len(self.objects)):
            self.objects[i].set("pos", array_to_string(pos_arr[i]))
            self.objects[i].set("quat", array_to_string(quat_arr[i]))

    def place_visual(self):
        """Places visual objects randomly until no collisions or max iterations hit."""
        index = 0
        bin_pos = string_to_array(self.bin2_body.get("pos"))
        bin_size = self.bin_size

        for _, obj_mjcf in self.visual_objects.items():

            bin_x_low = bin_pos[0]
            bin_y_low = bin_pos[1]
            if index == 0 or index == 2:
                bin_x_low -= bin_size[0] / 2
            if index < 2:
                bin_y_low -= bin_size[1] / 2

            bin_x_high = bin_x_low + bin_size[0] / 2
            bin_y_high = bin_y_low + bin_size[1] / 2
            bottom_offset = obj_mjcf.get_bottom_offset()

            bin_range = [bin_x_low + bin_x_high, bin_y_low + bin_y_high, 2 * bin_pos[2]]
            bin_center = np.array(bin_range) / 2.0

            pos = bin_center - bottom_offset
            self.visual_obj_mjcf[index].set("pos", array_to_string(pos))
            index += 1
