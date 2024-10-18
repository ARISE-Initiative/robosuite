from robosuite.models.robots import *

class PandaOmron(Panda):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}


class SpotWithArm(SpotArm):
    @property
    def default_base(self):
        return "Spot"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}


class SpotWithArmFloating(SpotArm):
    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

class PandaDexRH(Panda):

    @property
    def default_gripper(self):
        return {"right": "InspireRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0., 0., 0.]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.5, -0.5, 0.5, 0.5]}

class PandaDexLH(Panda):

    @property
    def default_gripper(self):
        return {"right": "InspireLeftHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0., 0., 0.]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.5, -0.5, 0.5, 0.5]}

class GR1Rethink(GR1ArmsOnly):

    @property
    def default_gripper(self):
        return {"right": "RethinkGripper", "left": "RethinkGripper"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0., 0., 0.], "left": [0., 0., 0.]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0, 0, 1, 0], "left": [0, 0, 1, 0]}

class SpotArmRethink(SpotWithArm):

    @property
    def default_gripper(self):
        return {"right": "RethinkGripper"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.05, 0., 0.]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.707107, 0, 0.707107, 0]}

class SpotArmRobotiqGripper(SpotWithArm):

    @property
    def default_gripper(self):
        return {"right": "Robotiq85Gripper"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.05, 0., 0.]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.707107, 0, 0.707107, 0]}