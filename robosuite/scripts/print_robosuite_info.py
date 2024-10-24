from collections import OrderedDict

from robosuite.environments.base import REGISTERED_ENVS
from robosuite.models.bases import BASE_MAPPING
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.robots import ROBOT_CLASS_MAPPING


def bold_green_text(text):
    return f"\033[1m\033[32m{text}\033[0m"


def get_base_type(robot):
    return ROBOT_CLASS_MAPPING[robot].__name__ if robot in ROBOT_CLASS_MAPPING else "N/A"


def get_robot_info_dict():
    info = {}
    for i, key in enumerate(REGISTERED_ROBOTS):
        try:
            info[key] = {}
            robot = REGISTERED_ROBOTS[key](idn=i)
            base_type = get_base_type(key)
            robot_base_model = robot.default_base
            arm_type = robot.arm_type
            gripper = robot.default_gripper
            info[key]["Base_Type"] = base_type
            info[key]["Base_Model"] = robot_base_model
            info[key]["Arm_Type"] = arm_type
            info[key]["Gripper_Model"] = gripper
        except Exception as e:
            # could not instantiate the class on its own
            pass
    return info


print("Available environments/tasks:\n")
for i, env in enumerate(REGISTERED_ENVS):
    print(f"{i}. {bold_green_text(env)}")


info = get_robot_info_dict()
info_sorted = dict(sorted(info.items(), key=lambda item: item[1].get("Base_Type", "zzz")))

print("\n\n\nAvailable Robots:\n")

for i, key in enumerate(info_sorted):
    print(f"{i}. {bold_green_text(key)}")
    for k, item in info[key].items():
        k = k.replace("_", " ")
        print(f"\t{k}: {item}")
    print("\n")
