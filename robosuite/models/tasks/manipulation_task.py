import collections
from copy import deepcopy

from robosuite.models.world import MujocoWorldBase
from robosuite.models.tasks import UniformRandomSampler
from robosuite.models.objects import MujocoGeneratedObject, MujocoXMLObject
from robosuite.utils.mjcf_utils import new_joint, array_to_string


class ManipulationTask(MujocoWorldBase):
    """
    Creates MJCF model for a manipulation task performed, usually on a table top (or similar surface).

    A manipulation task consists of one or more robots interacting with a variable number of
    objects placed on a table. This class combines the robot(s), the arena, and the objects 
    into a single MJCF model.

    Args:
        mujoco_arena (Arena): MJCF model of robot workspace

        mujoco_robots (list of RobotModel): MJCF model of robot model(s) (list)

        mujoco_objects (OrderedDict of MujocoObject): a list of MJCF models of physical objects

        visual_objects (OrderedDict of MujocoObject): a list of MJCF models of visual-only objects that do not
            participate in collisions

        initializer (ObjectPositionSampler): placement sampler to initialize object positions.

    Raises:
        AssertionError: [Invalid input object type]
    """

    def __init__(
        self, 
        mujoco_arena, 
        mujoco_robots, 
        mujoco_objects, 
        visual_objects=None, 
        initializer=None,
    ):
        super().__init__()

        self.merge_arena(mujoco_arena)
        for mujoco_robot in mujoco_robots:
            self.merge_robot(mujoco_robot)

        if initializer is None:
            initializer = UniformRandomSampler()

        if visual_objects is None:
            visual_objects = collections.OrderedDict()

        assert isinstance(mujoco_objects, collections.OrderedDict)
        assert isinstance(visual_objects, collections.OrderedDict)

        mujoco_objects = deepcopy(mujoco_objects)
        visual_objects = deepcopy(visual_objects)

        # xml manifestations of all objects
        self.objects = []
        self.merge_objects(mujoco_objects)
        self.merge_objects(visual_objects, is_visual=True)

        merged_objects = collections.OrderedDict(**mujoco_objects, **visual_objects)
        self.mujoco_objects = mujoco_objects
        self.visual_objects = visual_objects

        self.initializer = initializer
        self.initializer.setup(merged_objects, self.table_top_offset, self.table_size)

    def merge_robot(self, mujoco_robot):
        """
        Adds robot model to the MJCF model.

        Args:
            mujoco_robot (RobotModel): robot to merge into this MJCF model
        """
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """
        Adds arena model to the MJCF model.

        Args:
            mujoco_arena (Arena): arena to merge into this MJCF model
        """
        self.arena = mujoco_arena
        self.table_top_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.table_full_size
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects, is_visual=False):
        """
        Adds object models to the MJCF model.

        Args:
            mujoco_objects (OrderedDict or MujocoObject): objects to merge into this MJCF model
            is_visual (bool): Whether the object is a visual object or not
        """
        if not is_visual:
            self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_objects.items():
            assert(isinstance(obj_mjcf, MujocoGeneratedObject) or isinstance(obj_mjcf, MujocoXMLObject))
            self.merge_asset(obj_mjcf)
            # Load object
            if is_visual:
                obj = obj_mjcf.get_visual(site=False)
            else:
                obj = obj_mjcf.get_collision(site=True)

            for i, joint in enumerate(obj_mjcf.joints):
                obj.append(new_joint(name="{}_jnt{}".format(obj_name, i), **joint))
            self.objects.append(obj)
            self.worldbody.append(obj)

            if not is_visual:
                self.max_horizontal_radius = max(
                    self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
                )

    def place_objects(self):
        """
        Places objects randomly until no collisions or max iterations hit.
        """
        pos_arr, quat_arr = self.initializer.sample()
        for i in range(len(self.objects)):
            self.objects[i].set("pos", array_to_string(pos_arr[i]))
            self.objects[i].set("quat", array_to_string(quat_arr[i]))
        return pos_arr, quat_arr
