"""
Extension of robosuite Table Top task to support visual objects.
This is so that we can place virtual objects to visualize goals.
"""
import numpy as np

from robosuite.models.tasks import TableTopTask
from robosuite.utils.mjcf_utils import new_joint, array_to_string


class TableTopVisualTask(TableTopTask):
    """
    Creates MJCF model of a tabletop task.
    A tabletop task consists of one robot interacting with a variable number of
    objects placed on the tabletop. This class combines the robot, the table
    arena, and the objetcts into a single MJCF model.
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects, visual_objects=[], initializer=None):
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
        super(TableTopVisualTask, self).__init__(mujoco_arena, mujoco_robot, mujoco_objects, initializer=initializer)
        self.merge_visual(visual_objects)
        self.visual_objects = visual_objects

    def merge_visual(self, mujoco_objects):
        """
        Only loads visual xml attributes for the objects, so that they have no collision.
        Assumes that @merge_objects has already been called.
        """

        self.visual_obj_mjcf = []
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)

            # Load object
            obj = obj_mjcf.get_visual(name=obj_name, site=False)
            obj.append(new_joint(name=obj_name, type="free"))
            self.visual_obj_mjcf.append(obj)
            self.worldbody.append(obj)

    def place_visual(self):
        """
        Place objects randomly until no more collisions or max iterations hit.
        """

        if len(self.objects) > 0:
            # position of first object (used to get z-offset)
            obj_pos = np.fromstring(self.objects[0].get('pos'), sep=" ")

        index = 0
        for _, obj_mjcf in self.visual_objects.items():
            # for now, use a fixed target position
            target_pos = np.array([self.arena.table_full_size[0] - 0.02, 0., obj_pos[2]])
            self.visual_obj_mjcf[index].set('pos', array_to_string(target_pos))
            index += 1