from robosuite.utils.mjcf_utils import new_joint, array_to_string
from robosuite.models.tasks import Task, UniformRandomPegsSampler


class NutAssemblyTask(Task):
    """
    Creates MJCF model of a nut assembly task.

    A nut assembly task consists of one robot picking up nuts from a table and
    and assembling them into pegs positioned on the tabletop. This class combines
    the robot, the arena with pegs, and the nut objetcts into a single MJCF model.
    """

    def __init__(self, mujoco_arena, mujoco_robots, mujoco_objects, initializer=None):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robots: MJCF model of robot model(s) (list)
            mujoco_objects: a list of MJCF models of physical objects
            initializer: placement sampler to initialize object positions.
        """
        super().__init__()

        self.object_metadata = []
        self.merge_arena(mujoco_arena)
        for mujoco_robot in mujoco_robots:
            self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)

        if initializer is None:
            initializer = UniformRandomPegsSampler()
        self.initializer = initializer
        self.initializer.setup(self.mujoco_objects, self.table_offset, self.table_size)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.table_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.table_full_size
        self.table_body = mujoco_arena.table_body
        self.peg1_body = mujoco_arena.peg1_body
        self.peg2_body = mujoco_arena.peg2_body
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects = {}  # xml manifestation
        self.max_horizontal_radius = 0
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free", damping="0.0005"))
            self.objects[obj_name] = obj
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""
        pos_arr, quat_arr = self.initializer.sample()
        for k, obj_name in enumerate(self.objects):
            self.objects[obj_name].set("pos", array_to_string(pos_arr[k]))
            self.objects[obj_name].set("quat", array_to_string(quat_arr[k]))
