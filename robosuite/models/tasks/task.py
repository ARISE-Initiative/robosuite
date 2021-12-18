from copy import deepcopy

from robosuite.models.objects import MujocoObject
from robosuite.models.robots import RobotModel
from robosuite.models.world import MujocoWorldBase
from robosuite.utils.mjcf_utils import get_ids


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

        self._instances_to_ids = None
        self._geom_ids_to_instances = None
        self._site_ids_to_instances = None
        self._classes_to_ids = None
        self._geom_ids_to_classes = None
        self._site_ids_to_classes = None

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
            assert isinstance(mujoco_obj, MujocoObject), "Tried to merge non-MujocoObject! Got type: {}".format(
                type(mujoco_obj)
            )
            # Merge this object
            self.merge_assets(mujoco_obj)
            self.worldbody.append(mujoco_obj.get_obj())

    def generate_id_mappings(self, sim):
        """
        Generates IDs mapping class instances to set of (visual) geom IDs corresponding to that class instance

        Args:
            sim (MjSim): Current active mujoco simulation object
        """
        self._instances_to_ids = {}
        self._geom_ids_to_instances = {}
        self._site_ids_to_instances = {}
        self._classes_to_ids = {}
        self._geom_ids_to_classes = {}
        self._site_ids_to_classes = {}

        models = [model for model in self.mujoco_objects]
        for robot in self.mujoco_robots:
            models += [robot] + robot.models

        # Parse all mujoco models from robots and objects
        for model in models:
            # Grab model class name and visual IDs
            cls = str(type(model)).split("'")[1].split(".")[-1]
            inst = model.name
            id_groups = [
                get_ids(sim=sim, elements=model.visual_geoms + model.contact_geoms, element_type="geom"),
                get_ids(sim=sim, elements=model.sites, element_type="site"),
            ]
            group_types = ("geom", "site")
            ids_to_instances = (self._geom_ids_to_instances, self._site_ids_to_instances)
            ids_to_classes = (self._geom_ids_to_classes, self._site_ids_to_classes)

            # Add entry to mapping dicts

            # Instances should be unique
            assert inst not in self._instances_to_ids, f"Instance {inst} already registered; should be unique"
            self._instances_to_ids[inst] = {}

            # Classes may not be unique
            if cls not in self._classes_to_ids:
                self._classes_to_ids[cls] = {group_type: [] for group_type in group_types}

            for ids, group_type, ids_to_inst, ids_to_cls in zip(
                id_groups, group_types, ids_to_instances, ids_to_classes
            ):
                # Add geom, site ids
                self._instances_to_ids[inst][group_type] = ids
                self._classes_to_ids[cls][group_type] += ids

                # Add reverse mappings as well
                for idn in ids:
                    assert idn not in ids_to_inst, f"ID {idn} already registered; should be unique"
                    ids_to_inst[idn] = inst
                    ids_to_cls[idn] = cls

    @property
    def geom_ids_to_instances(self):
        """
        Returns:
            dict: Mapping from geom IDs in sim to specific class instance names
        """
        return deepcopy(self._geom_ids_to_instances)

    @property
    def site_ids_to_instances(self):
        """
        Returns:
            dict: Mapping from site IDs in sim to specific class instance names
        """
        return deepcopy(self._site_ids_to_instances)

    @property
    def instances_to_ids(self):
        """
        Returns:
            dict: Mapping from specific class instance names to {geom, site} IDs in sim
        """
        return deepcopy(self._instances_to_ids)

    @property
    def geom_ids_to_classes(self):
        """
        Returns:
            dict: Mapping from geom IDs in sim to specific classes
        """
        return deepcopy(self._geom_ids_to_classes)

    @property
    def site_ids_to_classes(self):
        """
        Returns:
            dict: Mapping from site IDs in sim to specific classes
        """
        return deepcopy(self._site_ids_to_classes)

    @property
    def classes_to_ids(self):
        """
        Returns:
            dict: Mapping from specific classes to {geom, site} IDs in sim
        """
        return deepcopy(self._classes_to_ids)
