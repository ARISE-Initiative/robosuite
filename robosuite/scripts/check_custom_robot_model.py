from xml.etree import ElementTree as ET
import argparse

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER as logger

def check_parts(world_body, part_name, parts_dict):
    for joint in world_body.findall(".//joint"):
        if part_name in joint.attrib["name"] == part_name:
            parts_dict[part_name].append(joint.attrib["name"])

def check_robot_definition(xml_file):
    root = ET.parse(xml_file).getroot()
    # get all joints
    world_body = root.find(".//worldbody")
    parts_dict = {}
    for part_name in ["torso", "head", "leg"]:
        parts_dict[part_name] = []
        for joint in world_body.findall(".//joint"):
            if part_name in joint.attrib["name"]:
                parts_dict[part_name].append(joint.attrib["name"])
    parts_dict["arm"] = []

    print(len(world_body.findall(".//joint")))
    for joint in world_body.findall(".//joint"):
        is_non_arm_part = False
        for non_arm_part_name in ["torso", "head", "leg"]:
            if non_arm_part_name in joint.attrib["name"]:
                is_non_arm_part = True
                break
        if not is_non_arm_part:
            parts_dict["arm"].append(joint.attrib["name"])
    total_counted_joints = sum([len(parts_dict[part_name]) for part_name in parts_dict.keys()])
    if total_counted_joints != len(world_body.findall(".//joint")):
        logger.erro(f"Error in {file}")
        logger.error(f"Counted {total_counted_joints} joints, but found {len(world_body.findall('.//joint'))} joints")
    else:
        logger.info(f"Your joint definition aligns with robosuite convention.")
        # remove empty list entries
        parts_dict = {part_name: joints for part_name, joints in parts_dict.items() if joints}
        print(f"Robosuite will be able to detect the following body parts:  {list(parts_dict.keys())}")
        print("For each body part, you have defined the following joints:")
        for part_name in parts_dict.keys():
            print(f"{part_name} - {len(parts_dict[part_name])} joints: {parts_dict[part_name]}")

    # checking the order of arm joints defined.
    if parts_dict.get("arm") is not None and parts_dict["arm"] != []:
        arm_joint_names = parts_dict["arm"]
        first_arm_joint_name = arm_joint_names[0]
        if "l_" in first_arm_joint_name or "left" in first_arm_joint_name:
            logger.warning("Incorrect order of the arm joints. THe arm joints needs to be defined first for the right arm and then for the left arm.")
    
    logger.warning("Attention!!! Make sure your definition of the motors for arms are in the same order as the joints defined. This program does not check on this item.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["robot", "gripper"], required=True)
    args = parser.parse_args()
    if args.type == "robot":
        check_robot_definition(args.file)