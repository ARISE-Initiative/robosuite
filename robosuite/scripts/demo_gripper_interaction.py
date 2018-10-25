"""Gripper interaction demo.

This script illustrates the process of importing grippers into a scene and making it interact
with the objects with actuators. It also shows how to procedurally generate a scene with the
APIs of the MJCF utility functions.

Example:
    $ python run_gripper_test.py
"""

import xml.etree.ElementTree as ET
import gc
from mujoco_py import load_model_from_path, MjSim, MjViewer

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.gripper import TwoFingerGripper
from robosuite.models.mujoco_object import BoxObject
from robosuite.utils.mjcf_utils import new_joint, new_actuator


if __name__ == "__main__":

    # start with an empty world
    world = MujocoWorldBase()

    # add a table
    arena = TableArena(table_full_size=(0.4, 0.4, 0.1))
    world.merge(arena)

    # add a gripper
    gripper = TwoFingerGripper()
    gripper_body = ET.Element("body")
    for body in gripper.worldbody:
        gripper_body.append(body)
    gripper_body.set("pos", "0 0 0.3")
    gripper_body.set("quat", "0 0 1 0")  # flip z
    gripper_body.append(
        new_joint(name="gripper_z_joint", type="slide", axis="0 0 1", damping="50")
    )
    world.merge(gripper, merge_body=False)
    world.worldbody.append(gripper_body)
    world.actuator.append(
        new_actuator(
            joint="gripper_z_joint", act_type="position", name="gripper_z", kp="500"
        )
    )

    # add an object for grasping
    mujoco_object = BoxObject(
        size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1], friction=1
    ).get_collision()
    mujoco_object.append(new_joint(name="object_free_joint", type="free"))
    mujoco_object.set("pos", "0 0 0.11")
    geoms = mujoco_object.findall("./geom")
    for geom in geoms:
        if geom.get("contype"):
            pass
        geom.set("name", "object")
        geom.set("density", "10000")  # 1000 for water
    world.worldbody.append(mujoco_object)

    x_ref = BoxObject(size=[0.01, 0.01, 0.01], rgba=[0, 1, 0, 1]).get_visual()
    x_ref.set("pos", "0.2 0 0.105")
    world.worldbody.append(x_ref)
    y_ref = BoxObject(size=[0.01, 0.01, 0.01], rgba=[0, 0, 1, 1]).get_visual()
    y_ref.set("pos", "0 0.2 0.105")
    world.worldbody.append(y_ref)

    # start simulation
    model = world.get_model(mode="mujoco_py")

    sim = MjSim(model)
    viewer = MjViewer(sim)
    sim_state = sim.get_state()

    # for gravity correction
    gravity_corrected = ["gripper_z_joint"]
    _ref_joint_vel_indexes = [
        sim.model.get_joint_qvel_addr(x) for x in gravity_corrected
    ]

    gripper_z_id = sim.model.actuator_name2id("gripper_z")
    gripper_z_low = 0.07
    gripper_z_high = -0.02
    gripper_z_is_low = False

    gripper_joint_ids = [
        sim.model.actuator_name2id("gripper_" + x) for x in gripper.joints
    ]
    gripper_open = [0.0115, -0.0115]
    gripper_closed = [-0.020833, 0.020833]
    gripper_is_closed = True

    seq = [(False, False), (True, False), (True, True), (False, True)]

    sim.set_state(sim_state)
    step = 0
    T = 1000
    while True:

        if step % 100 == 0:
            print("step: {}".format(step))

        if step % 100 == 0:
            for contact in sim.data.contact[0 : sim.data.ncon]:

                geom_name1 = sim.model.geom_id2name(contact.geom1)
                geom_name2 = sim.model.geom_id2name(contact.geom2)
                if geom_name1 == "floor" and geom_name2 == "floor":
                    continue

                print("geom1: {}, geom2: {}".format(geom_name1, geom_name2))
                print("contact id {}".format(id(contact)))
                print("friction: {}".format(contact.friction))
                print("normal: {}".format(contact.frame[0:3]))

        if step % T == 0:
            plan = seq[int(step / T) % len(seq)]
            gripper_z_is_low, gripper_is_closed = plan
            print(
                "changing plan: gripper low: {}, gripper closed {}".format(
                    gripper_z_is_low, gripper_is_closed
                )
            )

        if gripper_z_is_low:
            sim.data.ctrl[gripper_z_id] = gripper_z_low
        else:
            sim.data.ctrl[gripper_z_id] = gripper_z_high
        if gripper_is_closed:
            sim.data.ctrl[gripper_joint_ids] = gripper_closed
        else:
            sim.data.ctrl[gripper_joint_ids] = gripper_open

        sim.step()
        sim.data.qfrc_applied[_ref_joint_vel_indexes] = sim.data.qfrc_bias[
            _ref_joint_vel_indexes
        ]
        viewer.render()
        step += 1
