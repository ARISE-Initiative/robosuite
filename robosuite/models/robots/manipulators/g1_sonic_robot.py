"""Unitree G1 Sonic robot model with Dex-3 hands."""

from __future__ import annotations

import numpy as np

from robosuite.models.robots.manipulators.legged_manipulator_model import LeggedManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

XML_PATH = xml_path_completion("robots/g1_sonic/robot.xml")

LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
LEG_STAND_QPOS = np.array(
    [
        -0.10,
        0.0,
        0.0,
        0.30,
        -0.20,
        0.0,
        -0.10,
        0.0,
        0.0,
        0.30,
        -0.20,
        0.0,
    ],
    dtype=float,
)

TORSO_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_HAND_JOINTS = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]

RIGHT_HAND_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]


def _actuator_names(joint_names: list[str]) -> list[str]:
    return [name[: -len("_joint")] if name.endswith("_joint") else name for name in joint_names]


class G1Sonic(LeggedManipulatorModel):
    """Unitree G1 Sonic MuJoCo model with Dex-3 hands."""

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(str(XML_PATH), idn=idn)
        self.update_joints()
        self.update_actuators()

    @property
    def default_base(self):
        return "NullBase"

    @property
    def default_gripper(self):
        return {"right": None, "left": None}

    @property
    def default_controller_config(self):
        return {"right": "default_g1", "left": "default_g1"}

    @property
    def init_qpos(self):
        qpos = np.zeros(len(self._joints))
        stand_by_joint = dict(zip(LEG_JOINTS, LEG_STAND_QPOS))
        for idx, joint_name in enumerate(self._joints):
            qpos[idx] = stand_by_joint.get(joint_name, 0.0)
        return qpos

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0.0),
            "empty": (-0.29, 0, 0.0),
            "table": lambda table_length: (-0.26 - table_length / 2, 0, 0.0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        return {"right": "right_wrist_yaw_link", "left": "left_wrist_yaw_link"}

    def _existing(self, names: list[str], available: set[str]) -> list[str]:
        existing = []
        for name in names:
            if name in available:
                existing.append(name)
                continue
            prefixed = self.correct_naming(name)
            if prefixed in available:
                existing.append(prefixed)
        return existing

    def update_joints(self):
        available = set(self.all_joints)
        self._base_joints = self._existing(["floating_base_joint"], available)
        self._head_joints = []
        self._legs_joints = self._existing(LEG_JOINTS, available)
        self._torso_joints = self._existing(TORSO_JOINTS, available)
        self._arms_joints = self._existing(
            RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS + LEFT_ARM_JOINTS + LEFT_HAND_JOINTS,
            available,
        )

    def update_actuators(self):
        available = set(self.all_actuators)
        self._base_actuators = []
        self._head_actuators = []
        self._legs_actuators = self._existing(_actuator_names(LEG_JOINTS), available)
        self._torso_actuators = self._existing(_actuator_names(TORSO_JOINTS), available)
        self._arms_actuators = self._existing(
            _actuator_names(RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS + LEFT_ARM_JOINTS + LEFT_HAND_JOINTS),
            available,
        )
