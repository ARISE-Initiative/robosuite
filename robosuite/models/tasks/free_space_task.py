from robosuite.models.tasks import Task


# from robosuite.utils.mjcf_utils import new_joint, array_to_string


class FreeSpaceTask(Task):
    """
    Creates MJCF model of a task in free space.

    A free space task consists of one robot doing something in free space.
    This class consists of only the robot.
    """

    def __init__(self, mujoco_arena, mujoco_robot):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
        """
        super().__init__()

        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.merge(mujoco_arena)

    '''
    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.targets = []  # xml manifestation
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free"))
            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )


    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""
        pos_arr, quat_arr = self.initializer.sample()
        for i in range(len(self.objects)):
            self.objects[i].set("pos", array_to_string(pos_arr[i]))
            self.objects[i].set("quat", array_to_string(quat_arr[i]))
    '''
