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

    def __init__(self, mujoco_arena, mujoco_robots, mujoco_objects, visual_objects):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robots: MJCF model of robot model(s) (list)
            mujoco_objects: a list of MJCF models of physical objects
            visual_objects: a list of MJCF models of visual objects. Visual
                objects are excluded from physical computation, we use them to
                indicate the target destinations of the objects.
        """
        super().__init__()

        # temp: z-rotation
        self.z_rotation = True

        self.object_metadata = []
        self.merge_arena(mujoco_arena)
        for mujoco_robot in mujoco_robots:
            self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)
        self.merge_visual(OrderedDict(visual_objects))
        self.visual_objects = visual_objects

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
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
        placed_objects = []
        index = 0

        # place objects by rejection sampling
        for _, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            success = False
            for _ in range(5000):  # 5000 retries
                bin_x_half = self.bin_size[0] / 2 - horizontal_radius - 0.05
                bin_y_half = self.bin_size[1] / 2 - horizontal_radius - 0.05
                object_x = np.random.uniform(high=bin_x_half, low=-bin_x_half)
                object_y = np.random.uniform(high=bin_y_half, low=-bin_y_half)

                # make sure objects do not overlap
                object_xy = np.array([object_x, object_y, 0])
                pos = self.bin_offset - bottom_offset + object_xy
                location_valid = True
                for pos2, r in placed_objects:
                    dist = np.linalg.norm(pos[:2] - pos2[:2], np.inf)
                    if dist <= r + horizontal_radius:
                        location_valid = False
                        break

                # place the object
                if location_valid:
                    # add object to the position
                    placed_objects.append((pos, horizontal_radius))
                    self.objects[index].set("pos", array_to_string(pos))
                    # random z-rotation
                    quat = self.sample_quat()
                    self.objects[index].set("quat", array_to_string(quat))
                    success = True
                    break

            # raise error if all objects cannot be placed after maximum retries
            if not success:
                raise RandomizationError("Cannot place all objects in the bins")
            index += 1

    def place_visual(self):
        """Places visual objects randomly until no collisions or max iterations hit."""
        index = 0
        bin_pos = string_to_array(self.bin2_body.get("pos"))
        bin_size = self.bin_size

        for _, obj_mjcf in self.visual_objects:

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
