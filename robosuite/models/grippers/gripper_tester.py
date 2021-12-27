"""
Defines GripperTester that is used to test the physical properties of various grippers
"""
import xml.etree.ElementTree as ET

import numpy as np
from mujoco_py import MjSim, MjViewer

from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.world import MujocoWorldBase
from robosuite.utils.mjcf_utils import array_to_string, new_actuator, new_joint


class GripperTester:
    """
    A class that is used to test gripper

    Args:
        gripper (GripperModel): A gripper instance to be tested
        pos (str): (x y z) position to place the gripper in string form, e.g. '0 0 0.3'
        quat (str): rotation to apply to gripper in string form, e.g. '0 0 1 0' to flip z axis
        gripper_low_pos (float): controls the gipper y position, larger -> higher
        gripper_high_pos (float): controls the gipper y high position larger -> higher,
            must be larger than gripper_low_pos
        box_size (None or 3-tuple of int): the size of the box to grasp, None defaults to [0.02, 0.02, 0.02]
        box_density (int): the density of the box to grasp
        step_time (int): the interval between two gripper actions
        render (bool): if True, show rendering
    """

    def __init__(
        self,
        gripper,
        pos,
        quat,
        gripper_low_pos,
        gripper_high_pos,
        box_size=None,
        box_density=10000,
        step_time=400,
        render=True,
    ):
        # define viewer
        self.viewer = None

        world = MujocoWorldBase()
        # Add a table
        arena = TableArena(table_full_size=(0.4, 0.4, 0.1), table_offset=(0, 0, 0.1), has_legs=False)
        world.merge(arena)

        # Add a gripper
        self.gripper = gripper
        # Create another body with a slider joint to which we'll add this gripper
        gripper_body = ET.Element("body")
        gripper_body.set("pos", pos)
        gripper_body.set("quat", quat)  # flip z
        gripper_body.append(new_joint(name="gripper_z_joint", type="slide", axis="0 0 -1", damping="50"))
        # Add all gripper bodies to this higher level body
        for body in gripper.worldbody:
            gripper_body.append(body)
        # Merge the all of the gripper tags except its bodies
        world.merge(gripper, merge_body=None)
        # Manually add the higher level body we created
        world.worldbody.append(gripper_body)
        # Create a new actuator to control our slider joint
        world.actuator.append(new_actuator(joint="gripper_z_joint", act_type="position", name="gripper_z", kp="500"))

        # Add an object for grasping
        # density is in units kg / m3
        TABLE_TOP = [0, 0, 0.09]
        if box_size is None:
            box_size = [0.02, 0.02, 0.02]
        box_size = np.array(box_size)
        self.cube = BoxObject(
            name="object", size=box_size, rgba=[1, 0, 0, 1], friction=[1, 0.005, 0.0001], density=box_density
        )
        object_pos = np.array(TABLE_TOP + box_size * [0, 0, 1])
        mujoco_object = self.cube.get_obj()
        # Set the position of this object
        mujoco_object.set("pos", array_to_string(object_pos))
        # Add our object to the world body
        world.worldbody.append(mujoco_object)

        # add reference objects for x and y axes
        x_ref = BoxObject(
            name="x_ref", size=[0.01, 0.01, 0.01], rgba=[0, 1, 0, 1], obj_type="visual", joints=None
        ).get_obj()
        x_ref.set("pos", "0.2 0 0.105")
        world.worldbody.append(x_ref)
        y_ref = BoxObject(
            name="y_ref", size=[0.01, 0.01, 0.01], rgba=[0, 0, 1, 1], obj_type="visual", joints=None
        ).get_obj()
        y_ref.set("pos", "0 0.2 0.105")
        world.worldbody.append(y_ref)

        self.world = world
        self.render = render
        self.simulation_ready = False
        self.step_time = step_time
        self.cur_step = 0
        if gripper_low_pos > gripper_high_pos:
            raise ValueError(
                "gripper_low_pos {} is larger " "than gripper_high_pos {}".format(gripper_low_pos, gripper_high_pos)
            )
        self.gripper_low_pos = gripper_low_pos
        self.gripper_high_pos = gripper_high_pos

    def start_simulation(self):
        """
        Starts simulation of the test world
        """
        model = self.world.get_model(mode="mujoco_py")

        self.sim = MjSim(model)
        if self.render:
            self.viewer = MjViewer(self.sim)
        self.sim_state = self.sim.get_state()

        # For gravity correction
        gravity_corrected = ["gripper_z_joint"]
        self._gravity_corrected_qvels = [self.sim.model.get_joint_qvel_addr(x) for x in gravity_corrected]

        self.gripper_z_id = self.sim.model.actuator_name2id("gripper_z")
        self.gripper_z_is_low = False

        self.gripper_actuator_ids = [self.sim.model.actuator_name2id(x) for x in self.gripper.actuators]

        self.gripper_is_closed = True

        self.object_id = self.sim.model.body_name2id(self.cube.root_body)
        object_default_pos = self.sim.data.body_xpos[self.object_id]
        self.object_default_pos = np.array(object_default_pos, copy=True)

        self.reset()
        self.simulation_ready = True

    def reset(self):
        """
        Resets the simulation to the initial state
        """
        self.sim.set_state(self.sim_state)
        self.cur_step = 0

    def close(self):
        """
        Close the viewer if it exists
        """
        if self.viewer is not None:
            self.viewer.close()

    def step(self):
        """
        Forward the simulation by one timestep

        Raises:
            RuntimeError: if start_simulation is not yet called.
        """
        if not self.simulation_ready:
            raise RuntimeError("Call start_simulation before calling step")
        if self.gripper_z_is_low:
            self.sim.data.ctrl[self.gripper_z_id] = self.gripper_low_pos
        else:
            self.sim.data.ctrl[self.gripper_z_id] = self.gripper_high_pos
        if self.gripper_is_closed:
            self._apply_gripper_action(1)
        else:
            self._apply_gripper_action(-1)
        self._apply_gravity_compensation()
        self.sim.step()
        if self.render:
            self.viewer.render()
        self.cur_step += 1

    def _apply_gripper_action(self, action):
        """
        Applies binary gripper action

        Args:
            action (int): Action to apply. Should be -1 (open) or 1 (closed)
        """
        gripper_action_actual = self.gripper.format_action(np.array([action]))
        # rescale normalized gripper action to control ranges
        ctrl_range = self.sim.model.actuator_ctrlrange[self.gripper_actuator_ids]
        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        applied_gripper_action = bias + weight * gripper_action_actual
        self.sim.data.ctrl[self.gripper_actuator_ids] = applied_gripper_action

    def _apply_gravity_compensation(self):
        """
        Applies gravity compensation to the simulation
        """
        self.sim.data.qfrc_applied[self._gravity_corrected_qvels] = self.sim.data.qfrc_bias[
            self._gravity_corrected_qvels
        ]

    def loop(self, total_iters=1, test_y=False, y_baseline=0.01):
        """
        Performs lower, grip, raise and release actions of a gripper,
                each separated with T timesteps

        Args:
            total_iters (int): Iterations to perform before exiting
            test_y (bool): test if object is lifted
            y_baseline (float): threshold for determining that object is lifted
        """
        seq = [(False, False), (True, False), (True, True), (False, True)]
        for cur_iter in range(total_iters):
            for cur_plan in seq:
                self.gripper_z_is_low, self.gripper_is_closed = cur_plan
                for step in range(self.step_time):
                    self.step()
            if test_y:
                if not self.object_height > y_baseline:
                    raise ValueError(
                        "object is lifed by {}, ".format(self.object_height)
                        + "not reaching the requirement {}".format(y_baseline)
                    )

    @property
    def object_height(self):
        """
        Queries the height (z) of the object compared to on the ground

        Returns:
            float: Object height relative to default (ground) object position
        """
        return self.sim.data.body_xpos[self.object_id][2] - self.object_default_pos[2]
