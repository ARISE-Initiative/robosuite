from xml.etree import ElementTree as ET
import argparse

from robosuite.robots.robot import Robot
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER as logger


def check_xml_definition(root):
    logger.info(f"Successfully loaded the xml file of robot model.")

    # get all joints
    world_body = root.find(".//worldbody")
    parts_dict = {}
    for part_name in ["torso", "head", "leg", "gripper", "base"]:
        parts_dict[part_name] = []
        for joint in world_body.findall(".//joint"):
            if part_name in joint.attrib["name"]:
                parts_dict[part_name].append(joint.attrib["name"])
    parts_dict["arm"] = []

    print(len(world_body.findall(".//joint")))
    for joint in world_body.findall(".//joint"):
        is_non_arm_part = False
        for non_arm_part_name in ["torso", "head", "leg", "gripper", "base"]:
            if non_arm_part_name in joint.attrib["name"]:
                is_non_arm_part = True
                break
        if not is_non_arm_part:
            parts_dict["arm"].append(joint.attrib["name"])
    total_counted_joints = sum([len(parts_dict[part_name]) for part_name in parts_dict.keys()])
    if total_counted_joints != len(world_body.findall(".//joint")):
        logger.error(f"Error in {file}")
        logger.error(f"Counted {total_counted_joints} joints, but found {len(world_body.findall('.//joint'))} joints")
    else:
        logger.info(f"Your joint definition aligns with robosuite convention.")
        # remove empty list entries
        parts_dict = {part_name: joints for part_name, joints in parts_dict.items() if joints}
        print(f"    • Robosuite will be able to detect the following body parts:  {list(parts_dict.keys())}")
        print("    • For each body part, you have defined the following joints:")
        for part_name in parts_dict.keys():
            print(f"        • {part_name} - {len(parts_dict[part_name])} joints: {parts_dict[part_name]}")
    return parts_dict

def check_registered_robot(robot_name):

    print(f"Loading {robot_name} ...")
    controller_config = load_composite_controller_config("BASIC")
    robot = Robot(robot_type=robot_name,      composite_controller_config=controller_config, gripper_type=None)
    logger.info(f"Succcessfully found the defined robot")

    robot.load_model()
    robot_model = robot.robot_model

    root = robot_model.tree.getroot()
    parts_dict = check_xml_definition(root)

    if parts_dict.get("arm") is not None and parts_dict["arm"] != []:
        # checking the order of arm joints defined.
        arm_joint_names = parts_dict["arm"]
        first_arm_joint_name = arm_joint_names[0]
        if "l_" in first_arm_joint_name or "left" in first_arm_joint_name:
            logger.warning("Incorrect order of the arm joints. THe arm joints needs to be defined first for the right arm and then for the left arm.")

        # check if the body for mounting exists
        num_arms = len(robot.arms)

        mount_names = [robot.robot_model._eef_name[name] for name in robot.robot_model._eef_name]
        world_body = root.find(".//worldbody")
        for mount_name in mount_names[:num_arms]:
            # find the body name of mount_name in the worldbody
            mount_body = world_body.find(f".//body[@name='robot0_{mount_name}']")
            if mount_body is None:
                logger.error(f"Error: No body named '{mount_name}' found in the worldbody. Please make sure you have defined the body for mounting the arm.")
                exit(1)
        print("        • Grippers will be mounted on the following bodies ", mount_names[:num_arms])

    logger.warning("Attention!!! Make sure your definition of the motors for arms are in the same order as the joints defined. This program does not check on this item.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default=None)
    parser.add_argument("--robot-xml-file", type=str, default=None)
    args = parser.parse_args()
    if args.robot is None and args.robot_xml_file is None:
        logger.error("Please provide either the robot name or the robot xml file.")
        exit(1)
    if args.robot is not None:
        check_registered_robot(args.robot)

    if args.robot_xml_file is not None:
        etree_root = ET.parse(args.robot_xml_file)
        parts_dict = check_xml_definition(etree_root)
        arm_joint_names = parts_dict["arm"]
        first_arm_joint_name = arm_joint_names[0]
        if "l_" in first_arm_joint_name or "left" in first_arm_joint_name:
            logger.warning("Incorrect order of the arm joints. THe arm joints needs to be defined first for the right arm and then for the left arm.")
