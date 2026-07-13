"""Generate the SonicG1 robot + Dex3 gripper MJCF assets (robosuite-native).

Splits SONIC's validated 29-DOF model (GR00T-WholeBodyControl model_data
g1_29dof_with_hand.xml) into a robosuite-conformant robot body + two Dex3
three-finger grippers, copying meshes into the robosuite asset tree with relative
paths. Bodies/geoms/joints/meshes move verbatim -> the reassembled robot is
physically identical to the integrated model (mass + DOF + finger poses preserved).

Run once (needs the GR00T model_data + meshes available):
    python -m robosuite.scripts.build_sonic_g1_assets
Outputs:
    models/assets/robots/sonic_g1/robot.xml   (+ meshes/)
    models/assets/grippers/sonic_dex3_{left,right}.xml   (+ meshes/sonic_dex3/)
"""

# Kept in-repo as the provenance/reproducer for the committed SonicG1 + Dex3 assets (run
# once, locally, when the GR00T model_data changes); it is not imported at runtime.

import copy
import os
import shutil
import xml.etree.ElementTree as ET

import robosuite

GEAR = "/home/ajay/code/GR00T-WholeBodyControl/gear_sonic"
SRC = os.path.join(GEAR, "data/robot_model/model_data/g1/g1_29dof_with_hand.xml")
MESHDIR = os.path.join(GEAR, "data/robot_model/model_data/g1/meshes")

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(robosuite.__file__))),
                      "robosuite", "models", "assets")
ROBOT_DIR = os.path.join(ASSETS, "robots", "sonic_g1")
GRIP_DIR = os.path.join(ASSETS, "grippers")
GRIP_MESH_REL = os.path.join("meshes", "sonic_dex3")
SIDES = ["left", "right"]


def _find_body(root, name):
    return next((b for b in root.iter("body") if b.get("name") == name), None)


def _copy_mesh(name, dst_dir, mesh_elems):
    src = os.path.join(MESHDIR, os.path.basename(mesh_elems[name].get("file")))
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))
    return os.path.basename(src)


def main():
    os.makedirs(os.path.join(ROBOT_DIR, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(GRIP_DIR, GRIP_MESH_REL), exist_ok=True)
    tree = ET.parse(SRC)
    root = tree.getroot()
    asset = root.find("asset")
    mesh_elems = {m.get("name"): m for m in asset.findall("mesh")}
    orig_default = copy.deepcopy(root.find("default"))

    # --- detach finger chains; add eef body + center site at each wrist ---
    grip_fingers, finger_meshes, hand_joints = {}, set(), set()
    for side in SIDES:
        wrist = _find_body(root, f"{side}_wrist_yaw_link")
        fingers = [b for b in list(wrist) if b.tag == "body" and "hand" in (b.get("name") or "")]
        assert len(fingers) == 3, side
        grip_fingers[side] = [copy.deepcopy(b) for b in fingers]
        for b in fingers:
            for g in b.iter("geom"):
                if g.get("mesh"):
                    finger_meshes.add(g.get("mesh"))
            for j in b.iter("joint"):
                hand_joints.add(j.get("name"))
            wrist.remove(b)
        eef = ET.SubElement(wrist, "body", {"name": f"{side}_eef", "pos": "0 0 0"})
        ET.SubElement(eef, "inertial", {"pos": "0 0 0", "mass": "0", "diaginertia": "0 0 0"})
        ET.SubElement(wrist, "site", {"name": f"{side}_center", "pos": "0 0 0",
                                      "size": "0.01", "group": "2", "rgba": "1 0.3 0.3 1"})

    # --- BODY: drop hand motors + hand-joint sensors + finger meshes; relative meshes ---
    act = root.find("actuator")
    body_hand_motors = {s: [] for s in SIDES}
    for m in list(act):
        if m.get("joint") in hand_joints:
            side = "left" if m.get("joint").startswith("left") else "right"
            body_hand_motors[side].append(copy.deepcopy(m))
            act.remove(m)
    bsensor = root.find("sensor")
    if bsensor is not None:
        for s in list(bsensor):
            if s.get("joint") in hand_joints:
                bsensor.remove(s)
    for mn in list(finger_meshes):
        if mn in mesh_elems:
            asset.remove(mesh_elems[mn])
    for m in asset.findall("mesh"):  # copy body meshes, set relative path
        fn = _copy_mesh(m.get("name"), os.path.join(ROBOT_DIR, "meshes"),
                        {m.get("name"): m})
        m.set("file", os.path.join("meshes", fn))
    comp = root.find("compiler")
    if comp is not None:
        comp.attrib.pop("meshdir", None)
    wb = root.find("worldbody")
    for geom in wb.findall("geom"):
        if geom.get("type") == "plane" or geom.get("name") == "floor":
            wb.remove(geom)
    for light in wb.findall("light"):
        wb.remove(light)

    # Rename body JOINTS/ACTUATORS/SENSORS to robosuite's part-classification
    # convention (GR1-style: legs -> l_/r_leg_*, waist -> torso_waist_*, arms ->
    # l_/r_*) so robosuite classifies legs/torso/arms correctly (manipulator_model
    # keys on "leg"/"torso" substrings; arms split evenly per side). SONIC's keyword
    # detection (hip/knee/ankle/waist/shoulder/elbow/wrist are still substrings) and
    # the actuator MOTOR_ORDER are preserved. Body/link/mesh names are left alone
    # (classification uses joint/actuator names, not bodies).
    _RN = [("left_hip", "l_leg_hip"), ("right_hip", "r_leg_hip"),
           ("left_knee", "l_leg_knee"), ("right_knee", "r_leg_knee"),
           ("left_ankle", "l_leg_ankle"), ("right_ankle", "r_leg_ankle"),
           ("waist_", "torso_waist_"),
           ("left_shoulder", "l_shoulder"), ("right_shoulder", "r_shoulder"),
           ("left_elbow", "l_elbow"), ("right_elbow", "r_elbow"),
           ("left_wrist", "l_wrist"), ("right_wrist", "r_wrist")]

    def _rn(s):
        for a, b in _RN:
            if a in s:
                return s.replace(a, b)
        return s

    # Convert the named free joint -> <freejoint name="root"/> (robosuite convention): the
    # <freejoint> TAG (not <joint type="free">) keeps robosuite's joint classification from
    # treating it as an arm joint, and lets LeggedManipulatorModel._remove_free_joint() drop
    # it for the fixed variant (both match by tag). We KEEP a name on it ("root") so the joint
    # is named: robocasa fixtures (e.g. oven/toaster_oven update_state) iterate
    # env.sim.model.joint_names doing `"rack" in joint_name`, which crashes on an unnamed
    # (None) joint -- so an unnamed freejoint broke every kitchen layout that has an oven.
    for b in wb.iter("body"):
        for j in list(b):
            if j.tag == "joint" and j.get("type") == "free":
                i = list(b).index(j)
                b.remove(j)
                b.insert(i, ET.Element("freejoint", {"name": "root"}))
    for j in wb.iter("joint"):
        if j.get("name"):
            j.set("name", _rn(j.get("name")))
    for a in act.findall("motor"):
        if a.get("name"):
            a.set("name", _rn(a.get("name")))
        if a.get("joint"):
            a.set("joint", _rn(a.get("joint")))
    if bsensor is not None:
        for s in list(bsensor):
            if s.get("name"):
                s.set("name", _rn(s.get("name")))
            if s.get("joint"):
                s.set("joint", _rn(s.get("joint")))
    tree.write(os.path.join(ROBOT_DIR, "robot.xml"))

    # --- GRIPPERS ---
    for side in SIDES:
        s0 = side[0]
        g = ET.Element("mujoco", {"model": f"sonic_dex3_{side}"})
        ET.SubElement(g, "compiler", {"angle": "radian", "autolimits": "true"})
        g.append(copy.deepcopy(orig_default))
        gasset = ET.SubElement(g, "asset")
        for mn in sorted(n for n in finger_meshes if n.startswith(f"{side}_hand")):
            fn = _copy_mesh(mn, os.path.join(GRIP_DIR, GRIP_MESH_REL), mesh_elems)
            me = copy.deepcopy(mesh_elems[mn])
            me.set("file", os.path.join(GRIP_MESH_REL, fn))
            gasset.append(me)
        wbody = ET.SubElement(g, "worldbody")
        rootb = ET.SubElement(wbody, "body", {"name": f"{s0}_gripper_base", "pos": "0 0 0", "quat": "1 0 0 0"})
        ET.SubElement(rootb, "inertial", {"pos": "0 0 0", "mass": "0", "diaginertia": "0 0 0"})
        ET.SubElement(rootb, "site", {"name": "ft_frame", "pos": "0 0 0", "size": "0.01",
                                      "rgba": "1 0 0 0", "type": "sphere", "group": "1"})
        eefb = ET.SubElement(rootb, "body", {"name": "eef", "pos": "0.05 0 0"})
        ET.SubElement(eefb, "inertial", {"pos": "0 0 0", "mass": "0", "diaginertia": "0 0 0"})
        ET.SubElement(eefb, "site", {"name": "grip_site", "pos": "0 0 0", "size": "0.01",
                                     "rgba": "1 1 0 1", "type": "sphere", "group": "2"})
        for ax, q, color, pos in (("x", "0.707105 0 0.707108 0", "1 0 0 0", "0.1 0 0"),
                                  ("y", "0.707105 0.707108 0 0", "0 1 0 0", "0 0.1 0"),
                                  ("z", "1 0 0 0", "0 0 1 0", "0 0 0.1")):
            ET.SubElement(eefb, "site", {"name": f"ee_{ax}", "pos": pos, "size": "0.005 .1",
                                         "quat": q, "rgba": color, "type": "cylinder", "group": "1"})
        ET.SubElement(eefb, "site", {"name": "grip_site_cylinder", "pos": "0 0 0", "quat": "1 0 0 0",
                                     "size": "0.005 0.5", "rgba": "0 1 0 0.3", "type": "cylinder", "group": "1"})
        for fb in grip_fingers[side]:
            rootb.append(fb)
        gact = ET.SubElement(g, "actuator")
        for m in body_hand_motors[side]:
            gact.append(m)
        gsensor = ET.SubElement(g, "sensor")  # robosuite expects raw names force_ee/torque_ee
        ET.SubElement(gsensor, "force", {"name": "force_ee", "site": "ft_frame"})
        ET.SubElement(gsensor, "torque", {"name": "torque_ee", "site": "ft_frame"})
        ET.ElementTree(g).write(os.path.join(GRIP_DIR, f"sonic_dex3_{side}.xml"))

    print(f"robot   -> {ROBOT_DIR}/robot.xml ({len(asset.findall('mesh'))} meshes)")
    print(f"grippers-> {GRIP_DIR}/sonic_dex3_{{left,right}}.xml")
    print(f"body actuators: {len(act.findall('motor'))}; hand joints/side: {len(hand_joints)//2}")


if __name__ == "__main__":
    main()
