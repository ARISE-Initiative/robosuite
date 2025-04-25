import xml.etree.ElementTree as ET
from copy import deepcopy

import mujoco

from robosuite.models.objects import MujocoObject
from robosuite.models.robots import RobotModel
from robosuite.models.world import MujocoWorldBase
from robosuite.utils.mjcf_utils import get_ids


def get_subtree_geom_ids_by_group(model: mujoco.MjModel, body_id: int, group: int) -> list[int]:
    """Get all geoms belonging to a subtree starting at a given body, filtered by group.

    Args:
        model: MuJoCo model.
        body_id: ID of body where subtree starts.
        group: Group ID to filter geoms.

    Returns:
        A list containing all subtree geom ids in the specified group.

    Adapted from https://github.com/kevinzakka/mink/blob/main/mink/utils.py
    """

    def gather_geoms(body_id: int) -> list[int]:
        geoms: list[int] = []
        geom_start = model.body_geomadr[body_id]
        geom_end = geom_start + model.body_geomnum[body_id]
        geoms.extend(geom_id for geom_id in range(geom_start, geom_end) if model.geom_group[geom_id] == group)
        children = [i for i in range(model.nbody) if model.body_parentid[i] == body_id]
        for child_id in children:
            geoms.extend(gather_geoms(child_id))
        return geoms

    return gather_geoms(body_id)


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

        enable_multiccd (bool) whether to set the multiccd flag in MuJoCo. False by default

    Raises:
        AssertionError: [Invalid input object type]
    """

    def __init__(
        self,
        mujoco_arena,
        mujoco_robots,
        mujoco_objects=None,
        enable_multiccd=False,
    ):
        super().__init__(enable_multiccd=enable_multiccd)

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

        worldbody = self.mujoco_arena.root.find("worldbody")
        exclude_bodies = ["table", "left_eef_target", "right_eef_target"]  # targets used for viz / mjgui
        top_level_bodies = [
            body.attrib.get("name")
            for body in worldbody.findall("body")
            if body.attrib.get("name") not in exclude_bodies
        ]
        models.extend(top_level_bodies)

        # Parse all mujoco models from robots and objects
        for model in models:
            if isinstance(model, str):
                body_name = model
                visual_group_number = 1
                body_id = sim.model.body_name2id(body_name)
                inst, cls = body_name, body_name
                geom_ids = get_subtree_geom_ids_by_group(sim.model, body_id, visual_group_number)
                id_groups = [geom_ids, []]
            else:
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
