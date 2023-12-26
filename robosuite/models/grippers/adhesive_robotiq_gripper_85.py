"""
Gripper with adhesion mechanism for Suction and Particle Jamming
"""
import numpy as np

from robosuite.models.grippers.flex_gripper_model import FlexGripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class AdhesiveRobotiq85Gripper(FlexGripperModel):
    """
    Gripper with adhesion mechanism for Suction and Particle Jamming

    Args:
        n_control (int): Number of control dimensions of the gripper
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/adhesive_robotiq_gripper_85.xml"), idn=idn)

        """
            dofs (dict): a dictionary of the form {
                group_name: [direction1, direction2, ...], ... # dof of each group is length of the list 
            }
        """
        self.dofs = {
            "finger": [1, 1],
            "adhesion": [1],
        }
        self.action_order = ["finger", "adhesion"]

        self.n_control = len(self.dofs.keys())

    @property
    def init_qpos(self):
        return np.array([-0.026, -0.267, -0.200, -0.026, -0.267, -0.200])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision",
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision",
            ],
            "left_fingerpad": ["left_fingerpad_collision"],
            "right_fingerpad": ["right_fingerpad_collision"],
        }

    @property
    def _important_sensors(self):
        return {
            "force_ee": "force_ee",
            "torque_ee": "torque_ee",
        }

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        signs = np.ones(len(self.actuators))
        for group_name, group_signs in self.dofs.items():
            group_entity_indices = []
            group_action_idx = self.action_order.index(group_name)
            group_action = action[group_action_idx]
            for i, actuator in enumerate(self.actuators):
                if actuator.endswith(group_name):
                    group_entity_indices.append(i)
            for i, group_entity_index in enumerate(group_entity_indices):
                signs[group_entity_index] = group_signs[i] * np.sign(group_action)

        self.current_action = np.clip(
            self.current_action + self.speed * signs, -1.0, 1.0
        )
        return self.current_action

    @property
    def dof(self):
        return len(self.actuators)

    @property
    def speed(self):
        return 0.01
