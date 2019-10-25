from robosuite.utils.mjcf_utils import new_joint, array_to_string
from robosuite.models.tasks import Task, UniformRandomPegsSampler
import numpy as np

class DoorTask(Task):
    """
    Creates MJCF model of a door opening task.
      
    A door assembly task consists of one robot approaching a door and opening
    it by manipulating the handle. This class combines the robot, empty arena,
    and door object into a single MJCF model.
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            mujoco_objects: a list of MJCF models of physical objects
        """
        super().__init__()

        self.object_metadata = []
        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)


    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot


        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        # print(self.mujoco_objects)
        self.objects = []  # xml manifestation
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_objects.items():
            #self.merge(obj_mjcf)
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            #obj.append(new_joint(name=obj_name, type="free", damping="0.0005"))
            self.objects.append(obj)


            self.worldbody.append(obj)



            #self.max_horizontal_radius = max(
            #    self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            #)
    def sample_quat(self):
        """Samples quaternions of random rotations along the z-axis."""
        if self.z_rotation:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
            return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
        return [1, 0, 0, 0]

    def place_objects(self, randomize=False):
        """Places objects randomly until no collisions or max iterations hit."""
        # pos_arr = [0.762, 0.0, 1.43] //for big door
        # pos_arr = [0.762, 0.0, 1.00]
        pos_arr = [0.85, 0.30, 1.45]
        if randomize==True:
            x, y = np.random.uniform(high=np.array([0.05, 0.05]), low=np.array([-0.05, -0.05]))
            degree = np.random.uniform(high=10, low=-10)
            pos_arr[0]+=x
            pos_arr[1]+=y
        else:
            degree = 0.0
            
        rot_angle = degree*np.pi/180.0

        quat = [np.cos(rot_angle/2), 0, 0, np.sin(rot_angle/2)]


        index = 0
        for k, obj_name in enumerate(self.mujoco_objects):
            # print(array_to_string(pos_arr))
            self.objects[index].set("pos", array_to_string(pos_arr))
            self.objects[index].set("quat", array_to_string(quat))
            #self.objects[obj_name].set("quat", array_to_string(quat_arr[k]))


    def set_door_friction(self, friction):

        node = self.objects[0].find("./body[@name='frame']")
        node = node.find("./body[@name='door']")
        hinge = node.find("./joint[@name='door_hinge']")

        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def set_door_damping(self, damping):
        hinge = self._base_body.find("./joint[@name='door_hinge']")
        node = self.objects[0].find("./body[@name='frame']")
        node = node.find("./body[@name='door']")
        hinge = node.find("./joint[@name='door_hinge']")

        hinge.set("damping", array_to_string(np.array([damping])))



    @property
    def _base_body(self):
        node = self.mujoco_objects['Door'].worldbody.find("./body[@name='door_body']")

        node = node.find("./body[@name='collision']")
        node = node.find("./body[@name='frame']")
        node = node.find("./body[@name='door']")
        return node
    

