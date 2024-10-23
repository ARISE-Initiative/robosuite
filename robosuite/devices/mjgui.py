from typing import Dict, List, Optional, Tuple
import numpy as np
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix
import mujoco


def set_mocap_pose(
    sim, 
    target_pos: Optional[np.ndarray] = None, 
    target_mat: Optional[np.ndarray] = None, 
    mocap_name: str = "target"
):
    mocap_id = sim.model.body(mocap_name).mocapid[0]
    if target_pos is not None:
        sim.data.mocap_pos[mocap_id] = target_pos
    if target_mat is not None:
        # convert mat to quat
        target_quat = np.empty(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.reshape(9, 1))
        sim.data.mocap_quat[mocap_id] = target_quat

def get_mocap_pose(sim, mocap_name: str = "target") -> Tuple[np.ndarray, np.ndarray]:
    mocap_id = sim.model.body(mocap_name).mocapid[0]
    target_pos = np.copy(sim.data.mocap_pos[mocap_id])
    target_quat = np.copy(sim.data.mocap_quat[mocap_id])
    target_mat = np.empty(9)
    mujoco.mju_quat2Mat(target_mat, target_quat)
    target_mat = target_mat.reshape(3, 3)
    return target_pos, target_mat


class MJGUI(Device):
    """
    Class for 'device' involving mujoco viewer and mocap bodies being dragged by users mouse.

    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
    """

    def __init__(self, env):
        super().__init__(env)

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """
        print("")
        print("Mujoco viewer UI mouse 'device'. Use the mouse to drag mocap bodies. We use the mocap's coordinates " \
              "to output actions.")
        print("")

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the keyboard.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        return dict()

    def _reset_internal_state(self):
        """
        Resets internal state related to robot control
        """
        super()._reset_internal_state()
        self.grasp_states = [[False] * len(self.all_robot_arms[i]) for i in range(self.num_robots)]
        self.active_arm_indices = [0] * len(self.all_robot_arms)
        self.active_robot = 0
        self.base_modes = [False] * len(self.all_robot_arms)

        site_names: List[str] = self.env.robots[0].composite_controller.joint_action_policy.site_names
        for site_name in site_names:
            target_name_prefix = "right" if "right" in site_name else "left"  # hardcoded for now
            target_pos = self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(site_name)]
            target_mat = self.env.sim.data.site_xmat[self.env.sim.model.site_name2id(site_name)]
            set_mocap_pose(self.env.sim, target_pos, target_mat, f"{target_name_prefix}_eef_target")

    def input2action(self) -> Dict[str, np.ndarray]:
        """
        Uses mocap body poses to determine action for robot. Obtain input_type 
        (i.e. absolute actions or delta actions) and input_ref_frame (i.e. world frame, base frame or eef frame)
        from the controller itself.
        """
        action: Dict[str, np.ndarray] = {}
        site_names: List[str] = self.env.robots[0].composite_controller.joint_action_policy.site_names
        for site_name in site_names:
            target_name_prefix = "right" if "right" in site_name else "left"  # hardcoded for now
            target_pos_world, target_ori_mat_world = get_mocap_pose(self.env.sim, f"{target_name_prefix}_eef_target")

        # TODO: update action frames.
        # now convert actions to desired frames (take from controller)
        return action
