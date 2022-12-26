from robosuite.models.objects import ObjectGroup, Bin, Lid
import robosuite.utils.transform_utils as T
import robosuite.utils.sim_utils as SU
import numpy as np


class TransportGroup(ObjectGroup):
    """
    Group of objects that capture transporting a payload placed in a start bin to a target bin, while
    also requiring a piece of trash to be removed from the target bin
    Args:
        name (str): Name of that will the prepended to all geom bodies generated for this group
        payload (MujocoObject): Object that represents payload
        trash (MujocoObject): Object that represents trash
        bin_size (3-tuple): (x,y,z) full size of bins to place on tables
    """

    def __init__(self, name, payload, trash, bin_size=(0.3, 0.3, 0.15)):
        # Store and initialize internal variables
        self.payload = payload
        self.trash = trash
        self.bin_size = bin_size

        # Create bins and lid
        self.start_bin = Bin(name=f"{name}_start_bin", bin_size=bin_size, density=10000.)
        self.target_bin = Bin(name=f"{name}_target_bin", bin_size=bin_size, density=10000.)
        self.trash_bin = Bin(name=f"{name}_trash_bin", bin_size=bin_size, density=10000.)
        self.lid = Lid(name=f"{name}_start_bin_lid", lid_size=(*bin_size[:2], 0.01))

        # Relevant geom ids
        self.payload_geom_ids = None
        self.trash_geom_ids = None
        self.target_bin_base_geom_ids = None
        self.trash_bin_base_geom_ids = None
        self.lid_handle_geom_ids = None
        self.payload_body_id = None
        self.trash_body_id = None

        # Run super init
        super().__init__(name=name)

    def get_states(self):
        """
        Grabs all relevant information for this transport group. Returned dictionary maps keywords to corresponding
        values pulled from the current sim state.
        Returns:
            dict:
                "lid_handle_pose": list of (pos, quat) of lid handle
                "payload_pose": list of (pos, quat) of hammer handle
                "trash_pose": list of (pos, quat) of trash object
                "target_bin_pos": position of target bin (base geom)
                "trash_bin_pos": position of trash bin (base geom)
                "trash_in_trash_bin": True if trash object is touching the base of the trash bin
                "payload_in_target_bin": True if payload object is touching the base of the target bin
        """
        return {
            "lid_handle_pose": (self.lid_handle_pos, self.lid_handle_quat),
            "payload_pose": (self.payload_pos, self.payload_quat),
            "trash_pose": (self.trash_pos, self.trash_quat),
            "target_bin_pos": self.target_bin_pos,
            "trash_bin_pos": self.trash_bin_pos,
            "trash_in_trash_bin": self.trash_in_trash_bin,
            "payload_in_target_bin": self.payload_in_target_bin,
        }

    def _generate_objects(self):
        # Store all relevant objects in self._objects
        self._objects = {
            "payload": self.payload,
            "trash": self.trash,
            "start_bin": self.start_bin,
            "target_bin": self.target_bin,
            "trash_bin": self.trash_bin,
            "lid": self.lid,
        }

    def update_sim(self, sim):
        """
        Updates internal reference to sim and all other references
        Args:
            sim (MjSim): Active mujoco sim reference
        """
        # Always run super first
        super().update_sim(sim=sim)

        # Update internal references to IDs
        self.payload_geom_ids = [self.sim.model.geom_name2id(geom) for geom in self.payload.contact_geoms]
        self.trash_geom_ids = [self.sim.model.geom_name2id(geom) for geom in self.trash.contact_geoms]
        self.target_bin_base_geom_ids = [self.sim.model.geom_name2id(geom) for geom in self.target_bin.base_geoms]
        self.trash_bin_base_geom_ids = [self.sim.model.geom_name2id(geom) for geom in self.trash_bin.base_geoms]
        self.lid_handle_geom_ids = [self.sim.model.geom_name2id(geom) for geom in self.lid.handle_geoms]
        self.payload_body_id = self.sim.model.body_name2id(self.payload.root_body)
        self.trash_body_id = self.sim.model.body_name2id(self.trash.root_body)

    @property
    def lid_handle_pos(self):
        """
        Returns:
            np.array: (x,y,z) absolute position of the lid handle
        """
        return np.array(self.sim.data.geom_xpos[self.lid_handle_geom_ids[0]])

    @property
    def lid_handle_quat(self):
        """
        Returns:
            np.array: (x,y,z,w) quaternion of the lid handle
        """
        return np.array(T.mat2quat(self.sim.data.geom_xmat[self.lid_handle_geom_ids[0]].reshape(3, 3)))

    @property
    def payload_pos(self):
        """
        Returns:
            np.array: (x,y,z) absolute position of the payload
        """
        return np.array(self.sim.data.body_xpos[self.payload_body_id])

    @property
    def payload_quat(self):
        """
        Returns:
            np.array: (x,y,z,w) quaternion of the payload
        """
        return np.array(T.mat2quat(self.sim.data.body_xmat[self.payload_body_id].reshape(3, 3)))

    @property
    def trash_pos(self):
        """
        Returns:
            np.array: (x,y,z) absolute position of the trash
        """
        return np.array(self.sim.data.body_xpos[self.trash_body_id])

    @property
    def trash_quat(self):
        """
        Returns:
            np.array: (x,y,z,w) quaternion of the trash
        """
        return np.array(T.mat2quat(self.sim.data.body_xmat[self.trash_body_id].reshape(3, 3)))

    @property
    def target_bin_pos(self):
        """
        Returns:
            np.array: (x,y,z) absolute position of the target bin
        """
        return np.array(self.sim.data.geom_xpos[self.target_bin_base_geom_ids[0]])

    @property
    def trash_bin_pos(self):
        """
        Returns:
            np.array: (x,y,z) absolute position of the trash bin
        """
        return np.array(self.sim.data.geom_xpos[self.trash_bin_base_geom_ids[0]])

    @property
    def trash_in_trash_bin(self):
        """
        Returns:
            bool: True if trash is in trash bin
        """
        return SU.check_contact(self.sim, self.trash_bin.base_geoms, self.trash.contact_geoms)

    @property
    def payload_in_target_bin(self):
        """
        Returns:
            bool: True if payload is in target bin
        """
        return SU.check_contact(self.sim, self.target_bin.base_geoms, self.payload.contact_geoms)
