import os
import pathlib
import sys
from contextlib import contextmanager
from typing import Dict, List, Literal, Optional, Tuple

import mink
import mujoco
import mujoco.viewer
import numpy as np
from mink.configuration import Configuration
from mink.tasks.exceptions import TargetNotSet
from mink.tasks.frame_task import FrameTask

import robosuite.utils.transform_utils as T
from robosuite.controllers.composite.composite_controller import WholeBody, register_composite_controller
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.binding_utils import MjSim


def update(self, q: Optional[np.ndarray] = None, update_idxs: Optional[np.ndarray] = None) -> None:
    """Run forward kinematics.
    Args:
        q: Optional configuration vector to override internal `data.qpos` with.
    """
    if q is not None:
        if update_idxs is not None:
            self.data.qpos[update_idxs] = q
        else:
            self.data.qpos = q
    # The minimal function call required to get updated frame transforms is
    # mj_kinematics. An extra call to mj_comPos is required for updated Jacobians.
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_comPos(self.model, self.data)


Configuration.update = update


def compute_translation_error(self, configuration: Configuration) -> np.ndarray:
    r"""Compute the translation part of the frame task error.
    Args:
        configuration: Robot configuration :math:`q`.
    Returns:
        Frame task translation error vector :math:`e_p(q)`.
    """
    if self.transform_target_to_world is None:
        raise TargetNotSet(self.__class__.__name__)

    transform_frame_to_world = configuration.get_transform_frame_to_world(self.frame_name, self.frame_type)
    return np.linalg.norm(self.transform_target_to_world.translation() - transform_frame_to_world.translation())


def compute_orientation_error(self, configuration: Configuration) -> np.ndarray:
    r"""Compute the orientation part of the frame task error.
    Args:
        configuration: Robot configuration :math:`q`.
    Returns:
        Frame task orientation error vector :math:`e_o(q)`.
    """
    if self.transform_target_to_world is None:
        raise TargetNotSet(self.__class__.__name__)

    transform_frame_to_world = configuration.get_transform_frame_to_world(self.frame_name, self.frame_type)

    rot_src = self.transform_target_to_world.rotation().as_matrix()
    rot_tgt = transform_frame_to_world.rotation().as_matrix()
    angle_diff = np.arccos(np.clip((np.trace(rot_src @ rot_tgt.T) - 1) / 2, -1, 1))
    angle_diff = angle_diff * 180 / np.pi
    return angle_diff


@contextmanager
def suppress_stdout():
    """
    helper function to supress logging info from mink controller
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# monkey patch the FrameTask class for computing debugging errors
FrameTask.compute_translation_error = compute_translation_error
FrameTask.compute_orientation_error = compute_orientation_error


@register_composite_controller
class WholeBodyIK(WholeBody):
    name = "WHOLE_BODY_IK"

    def __init__(
        self, sim: MjSim, robot_model: RobotModel, grippers: Dict[str, GripperModel], lite_physics: bool = False
    ):
        super().__init__(sim, robot_model, grippers, lite_physics)


class WeightedPostureTask(mink.PostureTask):
    def __init__(
        self, model: mujoco.MjModel, cost: float, weights: np.ndarray, lm_damping: float = 0.0, gain: float = 1.0
    ) -> None:
        r"""Create weighted posture task.
        Args:
            cost: value used to cast joint angle differences to a homogeneous
                cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
            weights: vector of weights for each joint.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are unfeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that too large a
                damping can slow down the task.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.
        """
        super().__init__(model=model, cost=cost, lm_damping=lm_damping, gain=gain)
        self.weights = weights

    def compute_error(self, configuration):
        error = super().compute_error(configuration)
        return self.weights * error

    def compute_jacobian(self, configuration):
        J = super().compute_jacobian(configuration)
        # breakpoint()
        return self.weights[:, np.newaxis] * J

    def __repr__(self):
        """Human-readable representation of the weighted posture task."""
        return (
            "WeightedPostureTask("
            f"cost={self.cost}, "
            f"weights={self.weights}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )


class IKSolverMink:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_names: List[str],
        robot_model: mujoco.MjModel,
        robot_joint_names: Optional[List[str]] = None,
        debug: bool = False,
        input_action_repr: Literal["absolute", "relative", "relative_pose"] = "absolute",
        input_rotation_repr: Literal["quat_wxyz", "axis_angle"] = "axis_angle",
        posture_weights: Dict[str, float] = None,
        solve_freq: float = 20.0,
        hand_pos_cost: float = 1,
        hand_ori_cost: float = 0.5,
    ):
        self.full_model: mujoco.MjModel = model
        self.full_model_data: mujoco.MjData = data

        self.robot_model = robot_model
        self.configuration = mink.Configuration(self.robot_model)
        self.posture_weights = posture_weights
        self.hand_pos_cost = hand_pos_cost
        self.hand_ori_cost = hand_ori_cost

        self.hand_tasks: List[mink.FrameTask]
        self.posture_task: WeightedPostureTask

        if robot_joint_names is None:
            robot_joint_names: List[str] = [
                self.robot_model.joint(i).name
                for i in range(self.robot_model.njnt)
                if self.robot_model.joint(i).type != 0
            ]  # Exclude fixed joints
        # get all ids for robot bodies
        # self.robot_body_ids: List[int] = np.array([self.robot_model.body(name).id for name in robot_joint_names])
        self.full_model_dof_ids: List[int] = np.array([self.full_model.joint(name).id for name in robot_joint_names])

        self.robot_model_dof_ids: List[int] = np.array([self.robot_model.joint(name).id for name in robot_joint_names])
        self.full_model_dof_ids: List[int] = np.array([self.full_model.joint(name).id for name in robot_joint_names])
        self.site_ids = [self.robot_model.site(site_name).id for site_name in site_names]

        self.site_names = site_names
        self._setup_tasks()
        self.set_posture_target(np.zeros(self.robot_model.nq))

        self.solver = "quadprog"

        self.solve_freq = solve_freq

        self.input_action_repr = input_action_repr
        self.input_rotation_repr = input_rotation_repr
        ROTATION_REPRESENTATION_DIMS: Dict[str, int] = {"quat_wxyz": 4, "axis_angle": 3}
        self.rot_dim = ROTATION_REPRESENTATION_DIMS[input_rotation_repr]
        self.pos_dim = 3
        self.control_dim = len(self.site_names) * (self.pos_dim + self.rot_dim)
        # hardcoded control limits for now
        self.control_limits = np.array([-np.inf] * self.control_dim), np.array([np.inf] * self.control_dim)

        self.i = 0
        self.debug = debug
        if debug:
            self.task_errors: List[np.ndarray] = []
            self.trask_translation_errors: List[np.ndarray] = []
            self.task_orientation_errors: List[np.ndarray] = []

    def __repr__(self) -> str:
        return "IKSolverMink"

    def _setup_tasks(self):
        weights = np.ones(self.robot_model.nq)
        for joint_name, posture_weight in self.posture_weights.items():
            joint_idx = self.robot_model.joint(joint_name).id
            weights[joint_idx] = posture_weight

        self.posture_task = WeightedPostureTask(self.robot_model, cost=0.01, weights=weights, lm_damping=2)

        self.tasks = [self.posture_task]

        self.hand_tasks = self._create_frame_tasks(
            self.site_names, position_cost=self.hand_pos_cost, orientation_cost=self.hand_ori_cost
        )
        self.tasks.extend(self.hand_tasks)

    def _create_frame_tasks(self, frame_names: List[str], position_cost: float, orientation_cost: float):
        return [
            mink.FrameTask(
                frame_name=frame,
                frame_type="site",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=1.0,
            )
            for frame in frame_names
        ]

    def set_target_poses(self, target_poses: List[np.ndarray]):
        for task, target in zip(self.hand_tasks, target_poses):
            se3_target = mink.SE3.from_matrix(target)
            task.set_target(se3_target)

    def set_posture_target(self, posture_target: np.ndarray):
        self.posture_task.set_target(posture_target)

    def action_split_indexes(self) -> Dict[str, Tuple[int, int]]:
        action_split_indexes: Dict[str, Tuple[int, int]] = {}
        previous_idx = 0
        for site_name in self.site_names:
            total_dim = self.pos_dim + self.rot_dim
            last_idx = previous_idx + total_dim
            action_split_indexes[site_name + "_pos"] = (previous_idx, previous_idx + self.pos_dim)
            action_split_indexes[site_name + f"_{self.input_rotation_repr}"] = (previous_idx + self.pos_dim, last_idx)
            previous_idx = last_idx
        return action_split_indexes

    def solve(self, target_action: Optional[np.ndarray] = None) -> np.ndarray:
        # update configuration's data to match self.data for the joints we care about
        self.configuration.model.body("robot0_base").pos = self.full_model.body("robot0_base").pos
        self.configuration.model.body("robot0_base").quat = self.full_model.body("robot0_base").quat
        self.configuration.update(
            self.full_model_data.qpos[self.full_model_dof_ids], update_idxs=self.robot_model_dof_ids
        )

        if target_action is not None:
            target_action = target_action.reshape(len(self.site_names), -1)
            target_pos = target_action[:, : self.pos_dim]
            target_ori = target_action[:, self.pos_dim :]
            target_quat_wxyz = None

            if self.input_rotation_repr == "axis_angle":
                target_quat_wxyz = np.array(
                    [np.roll(T.axisangle2quat(target_ori[i]), 1) for i in range(len(target_ori))]
                )
            elif self.input_rotation_repr == "mat":
                target_quat_wxyz = np.array([np.roll(T.mat2quat(target_ori[i])) for i in range(len(target_ori))])
            elif self.input_rotation_repr == "quat_wxyz":
                target_quat_wxyz = target_ori

            if "relative" in self.input_action_repr:
                cur_pos = np.array([self.configuration.data.site(site_id).xpos for site_id in self.site_ids])
                cur_ori = np.array([self.configuration.data.site(site_id).xmat for site_id in self.site_ids])
            if self.input_action_repr == "relative":
                # decoupled pos and rotation deltas
                target_pos += cur_pos
                target_quat_xyzw = np.array(
                    [
                        T.quat_multiply(T.mat2quat(cur_ori[i].reshape(3, 3)), np.roll(target_quat_wxyz[i], -1))
                        for i in range(len(self.site_ids))
                    ]
                )
                target_quat_wxyz = np.array([np.roll(target_quat_xyzw[i], shift=1) for i in range(len(self.site_ids))])
            elif self.input_action_repr == "relative_pose":
                cur_poses = np.zeros((len(self.site_ids), 4, 4))
                for i in range(len(self.site_ids)):
                    cur_poses[i, :3, :3] = cur_ori[i].reshape(3, 3)
                    cur_poses[i, :3, 3] = cur_pos[i]
                    cur_poses[i, 3, :] = [0, 0, 0, 1]

                # Convert target action to target pose
                target_poses = np.zeros_like(cur_poses)
                for i in range(len(self.site_ids)):
                    target_poses[i, :3, :3] = T.quat2mat(target_quat_wxyz[i])
                    target_poses[i, :3, 3] = target_pos[i]
                    target_poses[i, 3, :] = [0, 0, 0, 1]

                # Apply target pose to current pose
                new_target_poses = np.array([np.dot(cur_poses[i], target_poses[i]) for i in range(len(self.site_ids))])

                # Split new target pose back into position and quaternion
                target_pos = new_target_poses[:, :3, 3]
                target_quat_wxyz = np.array(
                    [np.roll(T.mat2quat(new_target_poses[i, :3, :3]), shift=1) for i in range(len(self.site_ids))]
                )

            # create targets list shape is n_sites, 4, 4
            targets = [np.eye(4) for _ in range(len(self.site_names))]
            for i, (pos, quat_wxyz) in enumerate(zip(target_pos, target_quat_wxyz)):
                targets[i][:3, 3] = pos
                targets[i][:3, :3] = T.quat2mat(np.roll(quat_wxyz, -1))

            # set target poses
            for task, target in zip(self.hand_tasks, targets):
                with suppress_stdout():
                    se3_target = mink.SE3.from_matrix(target)
                task.set_target(se3_target)

        with suppress_stdout():
            vel = mink.solve_ik(self.configuration, self.tasks, 1 / self.solve_freq, self.solver, 1e-5)
        self.configuration.integrate_inplace(vel, 1 / self.solve_freq)

        self.i += 1
        if self.debug and self.i % 1 == 0:
            task_errors = self._get_task_errors()
            task_translation_errors = self._get_task_translation_errors()
            task_orientation_errors = self._get_task_orientation_errors()
            self.task_errors.append(task_errors)
            self.trask_translation_errors.append(task_translation_errors)
            self.task_orientation_errors.append(task_orientation_errors)

            # if self.i % 50:
            #     print(f"Task errors: {task_translation_errors}")

        return self.configuration.data.qpos[self.robot_model_dof_ids]

    def _get_task_translation_errors(self) -> List[float]:
        errors = []
        for task in self.hand_tasks:
            error = task.compute_translation_error(self.configuration)
            errors.append(error)
        return errors

    def _get_task_orientation_errors(self) -> List[float]:
        errors = []
        for task in self.hand_tasks:
            error = task.compute_orientation_error(self.configuration)
            errors.append(error)
        return errors

    def _get_task_errors(self) -> List[float]:
        errors = []
        for task in self.hand_tasks:
            error = task.compute_error(self.configuration)
            errors.append(np.linalg.norm(error[:3]))
        return errors


@register_composite_controller
class WholeBodyMinkIK(WholeBody):
    name = "WHOLE_BODY_MINK_IK"

    def __init__(
        self, sim: MjSim, robot_model: RobotModel, grippers: Dict[str, GripperModel], lite_physics: bool = False
    ):
        super().__init__(sim, robot_model, grippers, lite_physics)

    def _init_joint_action_policy(self):
        joint_names: str = []
        for part_name in self.composite_controller_specific_config["ik_target_part_names"]:
            joint_names += self.part_controllers[part_name].joint_names

        self.joint_action_policy = IKSolverMink(
            model=self.sim.model._model,
            data=self.sim.data._data,
            site_names=self.composite_controller_specific_config["ref_name"],
            robot_model=self.robot_model.mujoco_model,
            robot_joint_names=joint_names,
            input_action_repr=self.composite_controller_specific_config.get("ik_input_action_repr", "absolute"),
            input_rotation_repr=self.composite_controller_specific_config.get("ik_input_rotation_repr", "axis_angle"),
            solve_freq=self.composite_controller_specific_config.get("ik_solve_freq", 20),
            posture_weights=self.composite_controller_specific_config.get("ik_posture_weights", {}),
            hand_pos_cost=self.composite_controller_specific_config.get("ik_hand_pos_cost", 1.0),
            hand_ori_cost=self.composite_controller_specific_config.get("ik_hand_ori_cost", 0.5),
            debug=self.composite_controller_specific_config.get("ik_debug", False),
        )
