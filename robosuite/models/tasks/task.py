from robosuite.models.world import MujocoWorldBase
from robosuite.models.robots import RobotModel
from robosuite.models.objects import MujocoObject


class Task(MujocoWorldBase):
    """
    Creates MJCF model for a task performed.

    A task consists of one or more robots interacting with a variable number of
    objects. This class combines the robot(s), the arena, and the objects
    into a single MJCF model.

    Args:
        mujoco_arena (Arena): MJCF model of robot workspace

        mujoco_robots (RobotModel or list of RobotModel): MJCF model of robot model(s) (list)

        mujoco_objects (None or MujocoObject or list of MujocoObject): a list of MJCF models of physical objects

    Raises:
        AssertionError: [Invalid input object type]
    """

    def __init__(
        self, 
        mujoco_arena, 
        mujoco_robots, 
        mujoco_objects=None,
    ):
        super().__init__()

        # Store references to all models
        self.mujoco_arena = mujoco_arena
        self.mujoco_robots = [mujoco_robots] if isinstance(mujoco_robots, RobotModel) else mujoco_robots
        if mujoco_objects is None:
            self.mujoco_objects = []
        else:
            self.mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else mujoco_objects

        # Merge all models
        self.merge_arena(self.mujoco_arena)
        for mujoco_robot in self.mujoco_robots:
            self.merge_robot(mujoco_robot)
        self.merge_objects(self.mujoco_objects)

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
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """
        Adds object models to the MJCF model.

        Args:
            mujoco_objects (list of MujocoObject): objects to merge into this MJCF model
        """
        for mujoco_obj in mujoco_objects:
            # Make sure we actually got a MujocoObject
            assert isinstance(mujoco_obj, MujocoObject), \
                "Tried to merge non-MujocoObject! Got type: {}".format(type(mujoco_obj))
            # Merge this object
            self.merge_asset(mujoco_obj)
            self.worldbody.append(mujoco_obj.get_obj())
