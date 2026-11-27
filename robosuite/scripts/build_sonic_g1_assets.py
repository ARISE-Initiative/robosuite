"""Generate the SonicG1 robot and Dex3 gripper MJCF assets from SONIC's G1 model.

Splits gear_sonic's g1_29dof_with_hand.xml into a robosuite robot body and two Dex3 grippers and
copies the meshes into robosuite's asset tree. Bodies, geoms and meshes move unchanged, so the
reassembled robot has the same physics as SONIC's model; only joint names and the hand colour
change.

Not used at runtime: rerun it (with gear_sonic installed) when SONIC's model changes.
    python -m robosuite.scripts.build_sonic_g1_assets
Outputs:
    models/assets/robots/sonic_g1/robot.xml   (+ meshes/)
    models/assets/grippers/sonic_dex3_{left,right}.xml   (+ meshes/sonic_dex3/)
"""

import copy
import os
import shutil
import xml.etree.ElementTree as ET

import gear_sonic

from robosuite.models import assets_root

SOURCE_DIR = os.path.join(os.path.dirname(gear_sonic.__file__), "data/robot_model/model_data/g1")
ROBOT_DIR = os.path.join(assets_root, "robots", "sonic_g1")
GRIPPER_DIR = os.path.join(assets_root, "grippers")
GRIPPER_MESH_SUBDIR = os.path.join("meshes", "sonic_dex3")
SIDES = ["left", "right"]
HAND_RGBA = "0.2 0.2 0.2 1"

# robosuite assigns a joint to a part by its name: names containing "leg" go to the legs, "torso"
# to the torso and the rest to the arms. Each (old, new) pair renames the joints containing old.
JOINT_RENAMES = [
    ("left_hip", "l_leg_hip"),
    ("right_hip", "r_leg_hip"),
    ("left_knee", "l_leg_knee"),
    ("right_knee", "r_leg_knee"),
    ("left_ankle", "l_leg_ankle"),
    ("right_ankle", "r_leg_ankle"),
    ("waist_", "torso_waist_"),
    ("left_shoulder", "l_shoulder"),
    ("right_shoulder", "r_shoulder"),
    ("left_elbow", "l_elbow"),
    ("right_elbow", "r_elbow"),
    ("left_wrist", "l_wrist"),
    ("right_wrist", "r_wrist"),
]


def rename_joint(name):
    for old, new in JOINT_RENAMES:
        if old in name:
            return name.replace(old, new)
    return name


def copy_mesh(mesh, destination_dir):
    """Copy a <mesh> element's file out of SONIC's mesh directory and return the file name."""
    file_name = os.path.basename(mesh.get("file"))
    shutil.copy(os.path.join(SOURCE_DIR, "meshes", file_name), os.path.join(destination_dir, file_name))
    return file_name


def main():
    os.makedirs(os.path.join(ROBOT_DIR, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(GRIPPER_DIR, GRIPPER_MESH_SUBDIR), exist_ok=True)
    tree = ET.parse(os.path.join(SOURCE_DIR, "g1_29dof_with_hand.xml"))
    root = tree.getroot()
    asset = root.find("asset")
    meshes = {mesh.get("name"): mesh for mesh in asset.findall("mesh")}
    source_default = copy.deepcopy(root.find("default"))

    # Detach the three finger chains from each wrist (they become the gripper) and leave an eef
    # body and a centre site in their place.
    gripper_fingers, finger_mesh_names, hand_joint_names = {}, set(), set()
    for side in SIDES:
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        fingers = wrist.findall("body")
        assert len(fingers) == 3, side
        palm_geoms = [geom for geom in wrist.findall("geom") if "hand_palm" in geom.get("mesh")]
        for geom in palm_geoms + [geom for finger in fingers for geom in finger.iter("geom")]:
            geom.set("rgba", HAND_RGBA)
        gripper_fingers[side] = [copy.deepcopy(finger) for finger in fingers]
        for finger in fingers:
            finger_mesh_names.update(geom.get("mesh") for geom in finger.iter("geom") if geom.get("mesh"))
            hand_joint_names.update(joint.get("name") for joint in finger.iter("joint"))
            wrist.remove(finger)
        eef = ET.SubElement(wrist, "body", {"name": f"{side}_eef", "pos": "0 0 0"})
        ET.SubElement(eef, "inertial", {"pos": "0 0 0", "mass": "0", "diaginertia": "0 0 0"})
        ET.SubElement(
            wrist,
            "site",
            {"name": f"{side}_center", "pos": "0 0 0", "size": "0.01", "group": "2", "rgba": "1 0.3 0.3 1"},
        )

    # Robot body: move the hand motors to the grippers, drop the hand sensors and the finger
    # meshes, and copy the remaining meshes next to robot.xml.
    actuators = root.find("actuator")
    hand_motors = {side: [] for side in SIDES}
    for motor in list(actuators):
        if motor.get("joint") in hand_joint_names:
            side = "left" if motor.get("joint").startswith("left") else "right"
            hand_motors[side].append(copy.deepcopy(motor))
            actuators.remove(motor)
    sensors = root.find("sensor")
    for sensor in list(sensors):
        if sensor.get("joint") in hand_joint_names:
            sensors.remove(sensor)
    for mesh_name in finger_mesh_names:
        asset.remove(meshes[mesh_name])
    for mesh in asset.findall("mesh"):
        mesh.set("file", os.path.join("meshes", copy_mesh(mesh, os.path.join(ROBOT_DIR, "meshes"))))
    del root.find("compiler").attrib["meshdir"]

    # Turn the pelvis free joint into <freejoint name="root"/>: robosuite does not count a
    # <freejoint> as an arm joint, and robocasa fixtures need every joint to have a name.
    worldbody = root.find("worldbody")
    pelvis = worldbody.find("body")
    free_joint = pelvis.find("joint[@type='free']")
    pelvis.insert(list(pelvis).index(free_joint), ET.Element("freejoint", {"name": "root"}))
    pelvis.remove(free_joint)

    for joint in worldbody.iter("joint"):
        joint.set("name", rename_joint(joint.get("name")))
    for motor in actuators:
        motor.set("name", rename_joint(motor.get("name")))
        motor.set("joint", rename_joint(motor.get("joint")))
    for sensor in sensors:
        sensor.set("name", rename_joint(sensor.get("name")))
        if sensor.get("joint"):  # the IMU and frame sensors have no joint
            sensor.set("joint", rename_joint(sensor.get("joint")))
    tree.write(os.path.join(ROBOT_DIR, "robot.xml"))

    for side in SIDES:
        gripper = ET.Element("mujoco", {"model": f"sonic_dex3_{side}"})
        ET.SubElement(gripper, "compiler", {"angle": "radian", "autolimits": "true"})
        gripper.append(copy.deepcopy(source_default))
        gripper_asset = ET.SubElement(gripper, "asset")
        for mesh_name in sorted(name for name in finger_mesh_names if name.startswith(f"{side}_hand")):
            mesh = copy.deepcopy(meshes[mesh_name])
            file_name = copy_mesh(mesh, os.path.join(GRIPPER_DIR, GRIPPER_MESH_SUBDIR))
            mesh.set("file", os.path.join(GRIPPER_MESH_SUBDIR, file_name))
            gripper_asset.append(mesh)
        gripper_worldbody = ET.SubElement(gripper, "worldbody")
        base = ET.SubElement(
            gripper_worldbody, "body", {"name": f"{side[0]}_gripper_base", "pos": "0 0 0", "quat": "1 0 0 0"}
        )
        ET.SubElement(base, "inertial", {"pos": "0 0 0", "mass": "0", "diaginertia": "0 0 0"})
        ET.SubElement(
            base,
            "site",
            {"name": "ft_frame", "pos": "0 0 0", "size": "0.01", "rgba": "1 0 0 0", "type": "sphere", "group": "1"},
        )
        eef = ET.SubElement(base, "body", {"name": "eef", "pos": "0.05 0 0"})
        ET.SubElement(eef, "inertial", {"pos": "0 0 0", "mass": "0", "diaginertia": "0 0 0"})
        ET.SubElement(
            eef,
            "site",
            {"name": "grip_site", "pos": "0 0 0", "size": "0.01", "rgba": "1 1 0 1", "type": "sphere", "group": "2"},
        )
        for axis, quat, rgba, pos in (
            ("x", "0.707105 0 0.707108 0", "1 0 0 0", "0.1 0 0"),
            ("y", "0.707105 0.707108 0 0", "0 1 0 0", "0 0.1 0"),
            ("z", "1 0 0 0", "0 0 1 0", "0 0 0.1"),
        ):
            ET.SubElement(
                eef,
                "site",
                {
                    "name": f"ee_{axis}",
                    "pos": pos,
                    "size": "0.005 .1",
                    "quat": quat,
                    "rgba": rgba,
                    "type": "cylinder",
                    "group": "1",
                },
            )
        ET.SubElement(
            eef,
            "site",
            {
                "name": "grip_site_cylinder",
                "pos": "0 0 0",
                "quat": "1 0 0 0",
                "size": "0.005 0.5",
                "rgba": "0 1 0 0.3",
                "type": "cylinder",
                "group": "1",
            },
        )
        base.extend(gripper_fingers[side])
        ET.SubElement(gripper, "actuator").extend(hand_motors[side])
        gripper_sensors = ET.SubElement(gripper, "sensor")  # robosuite reads force_ee and torque_ee
        ET.SubElement(gripper_sensors, "force", {"name": "force_ee", "site": "ft_frame"})
        ET.SubElement(gripper_sensors, "torque", {"name": "torque_ee", "site": "ft_frame"})
        ET.ElementTree(gripper).write(os.path.join(GRIPPER_DIR, f"sonic_dex3_{side}.xml"))

    print(f"robot   -> {ROBOT_DIR}/robot.xml ({len(asset.findall('mesh'))} meshes)")
    print(f"grippers-> {GRIPPER_DIR}/sonic_dex3_{{left,right}}.xml")
    print(f"body actuators: {len(actuators)}; hand joints/side: {len(hand_joint_names) // 2}")


if __name__ == "__main__":
    main()
