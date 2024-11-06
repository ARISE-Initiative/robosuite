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
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER


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
        verbose: bool = False,
        input_type: Literal["absolute", "relative", "relative_pose"] = "absolute",
        input_ref_frame: Literal["world", "base", "eef"] = "world",
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

        self.full_model_dof_ids: List[int] = np.array([self.full_model.joint(name).id for name in robot_joint_names])

        self.robot_model_dof_ids: List[int] = np.array([self.robot_model.joint(name).id for name in robot_joint_names])
        self.full_model_dof_ids: List[int] = np.array([self.full_model.joint(name).id for name in robot_joint_names])
        self.site_ids = [self.robot_model.site(site_name).id for site_name in site_names]

        self.site_names = site_names
        self._setup_tasks()
        self.set_posture_target(np.zeros(self.robot_model.nq))

        self.solver = "quadprog"

        self.solve_freq = solve_freq

        self.input_type = input_type
        self.input_ref_frame = input_ref_frame
        self.input_rotation_repr = input_rotation_repr
        ROTATION_REPRESENTATION_DIMS: Dict[str, int] = {"quat_wxyz": 4, "axis_angle": 3}
        self.rot_dim = ROTATION_REPRESENTATION_DIMS[input_rotation_repr]
        self.pos_dim = 3
        self.control_dim = len(self.site_names) * (self.pos_dim + self.rot_dim)
        # hardcoded control limits for now
        self.control_limits = np.array([-np.inf] * self.control_dim), np.array([np.inf] * self.control_dim)

        self.i = 0
        self.verbose = verbose
        if verbose:
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
            simplified_site_name = "left" if "left" in site_name else "right"  # hack to simplify site names
            # goal is to specify the end effector actions as "left" or "right" instead of the actual site name
            # we assume that the site names for the ik solver are unique and contain "left" or "right" in them
            action_split_indexes[simplified_site_name] = (previous_idx, last_idx)
            previous_idx = last_idx

        return action_split_indexes

    def transform_pose(
        self, src_frame_pose: np.ndarray, src_frame: Literal["world", "base"], dst_frame: Literal["world", "base"]
    ) -> np.ndarray:
        """
        Transforms src_frame_pose from src_frame to dst_frame.
        """
        if src_frame == dst_frame:
            return src_frame_pose

        self.configuration.model.body("robot0_base").pos = self.full_model.body("robot0_base").pos
        self.configuration.model.body("robot0_base").quat = self.full_model.body("robot0_base").quat
        self.configuration.update()

        X_src_frame_pose = src_frame_pose
        # convert src frame pose to world frame pose
        if src_frame != "world":
            X_W_src_frame = self.configuration.get_transform_frame_to_world(src_frame, "body").as_matrix()
            X_W_pose = X_W_src_frame @ X_src_frame_pose
        else:
            X_W_pose = src_frame_pose

        # now convert to destination frame
        if dst_frame == "world":
            X_dst_frame_pose = X_W_pose
        elif dst_frame == "base":
            X_dst_frame_W = np.linalg.inv(
                self.configuration.get_transform_frame_to_world("robot0_base", "body").as_matrix()
            )  # hardcode name of base
            X_dst_frame_pose = X_dst_frame_W.dot(X_W_pose)

        return X_dst_frame_pose

    def solve(self, input_action: np.ndarray) -> np.ndarray:
        """
        Solve for the joint angles that achieve the desired target action.

        We assume input_action is specified as self.input_type, self.input_ref_frame, self.input_rotation_repr.
        We also assume input_action has the following format: [site1_pos, site1_rot, site2_pos, site2_rot, ...].

        By updating configuration's bose to match the actual base pose (in 'world' frame),
        we're requiring our tasks' targets to be in the 'world' frame for mink.solve_ik().
        """
        # update configuration's base to match actual base
        self.configuration.model.body("robot0_base").pos = self.full_model.body("robot0_base").pos
        self.configuration.model.body("robot0_base").quat = self.full_model.body("robot0_base").quat
        # update configuration's qpos to match actual qpos
        self.configuration.update(
            self.full_model_data.qpos[self.full_model_dof_ids], update_idxs=self.robot_model_dof_ids
        )

        input_action = input_action.reshape(len(self.site_names), -1)
        input_pos = input_action[:, : self.pos_dim]
        input_ori = input_action[:, self.pos_dim :]

        input_quat_wxyz = None
        if self.input_rotation_repr == "axis_angle":
            input_quat_wxyz = np.array([np.roll(T.axisangle2quat(input_ori[i]), 1) for i in range(len(input_ori))])
        elif self.input_rotation_repr == "mat":
            input_quat_wxyz = np.array([np.roll(T.mat2quat(input_ori[i])) for i in range(len(input_ori))])
        elif self.input_rotation_repr == "quat_wxyz":
            input_quat_wxyz = input_ori

        if self.input_ref_frame == "base":
            input_poses = np.zeros((len(self.site_ids), 4, 4))
            for i in range(len(self.site_ids)):
                base_pos = self.configuration.data.body("robot0_base").xpos
                base_ori = self.configuration.data.body("robot0_base").xmat.reshape(3, 3)
                base_pose = T.make_pose(base_pos, base_ori)
                input_pose = T.make_pose(input_pos[i], T.quat2mat(np.roll(input_quat_wxyz[i], -1)))
                input_poses[i] = np.dot(base_pose, input_pose)
            input_pos = input_poses[:, :3, 3]
            input_quat_wxyz = np.array(
                [np.roll(T.mat2quat(input_poses[i, :3, :3]), shift=1) for i in range(len(self.site_ids))]
            )
        elif self.input_ref_frame == "eef":
            raise NotImplementedError("Input reference frame 'eef' not yet implemented")

        if "delta" in self.input_type:
            cur_pos = np.array([self.configuration.data.site(site_id).xpos for site_id in self.site_ids])
            cur_ori = np.array([self.configuration.data.site(site_id).xmat for site_id in self.site_ids])
        if self.input_type == "delta":
            # decoupled pos and rotation deltas
            target_pos = input_pos + cur_pos
            target_quat_xyzw = np.array(
                [
                    T.quat_multiply(T.mat2quat(cur_ori[i].reshape(3, 3)), np.roll(input_quat_wxyz[i], -1))
                    for i in range(len(self.site_ids))
                ]
            )
            target_quat_wxyz = np.array([np.roll(target_quat_xyzw[i], shift=1) for i in range(len(self.site_ids))])
        elif self.input_type == "delta_pose":
            cur_poses = np.zeros((len(self.site_ids), 4, 4))
            for i in range(len(self.site_ids)):
                cur_poses[i] = T.make_pose(cur_pos[i], cur_ori[i])

            # Convert target action to target pose
            delta_poses = np.zeros_like(cur_poses)
            for i in range(len(self.site_ids)):
                delta_poses[i] = T.make_pose(input_pos[i], T.quat2mat(np.roll(input_quat_wxyz[i], -1)))

            # Apply target pose to current pose
            target_poses = np.array([np.dot(cur_poses[i], delta_poses[i]) for i in range(len(self.site_ids))])

            # Split new target pose back into position and quaternion
            target_pos = target_poses[:, :3, 3]
            target_quat_wxyz = np.array(
                [np.roll(T.mat2quat(target_poses[i, :3, :3]), shift=1) for i in range(len(self.site_ids))]
            )
        elif self.input_type == "absolute":
            target_pos = input_pos
            target_quat_wxyz = input_quat_wxyz
        else:
            raise ValueError(f"Invalid input type: {self.input_type}")

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
        if self.verbose:
            task_errors = self._get_task_errors()
            task_translation_errors = self._get_task_translation_errors()
            task_orientation_errors = self._get_task_orientation_errors()
            self.task_errors.append(task_errors)
            self.trask_translation_errors.append(task_translation_errors)
            self.task_orientation_errors.append(task_orientation_errors)

            if self.i % 50:
                print(f"Task errors: {task_translation_errors}")

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

    def __init__(self, sim: MjSim, robot_model: RobotModel, grippers: Dict[str, GripperModel]):
        super().__init__(sim, robot_model, grippers)

    def _validate_composite_controller_specific_config(self) -> None:
        # Check that all actuation_part_names exist in part_controllers
        original_ik_controlled_parts = self.composite_controller_specific_config["actuation_part_names"]
        self.valid_ik_controlled_parts = []
        valid_ref_names = []

        assert (
            "ref_name" in self.composite_controller_specific_config
        ), "The 'ref_name' key is missing from composite_controller_specific_config."

        for part in original_ik_controlled_parts:
            if part in self.part_controllers:
                self.valid_ik_controlled_parts.append(part)
            else:
                ROBOSUITE_DEFAULT_LOGGER.warning(
                    f"Part '{part}' specified in 'actuation_part_names' "
                    "does not exist in part_controllers. Removing ..."
                )

        # Update the configuration with only the valid parts
        self.composite_controller_specific_config["actuation_part_names"] = self.valid_ik_controlled_parts

        # Loop through ref_names and validate against mujoco model
        original_ref_names = self.composite_controller_specific_config.get("ref_name", [])
        for ref_name in original_ref_names:
            if ref_name in self.sim.model.site_names:  # Check if the site exists in the mujoco model
                valid_ref_names.append(ref_name)
            else:
                ROBOSUITE_DEFAULT_LOGGER.warning(
                    f"Reference name '{ref_name}' specified in configuration"
                    " does not exist in the mujoco model. Removing ..."
                )

        # Update the configuration with only the valid reference names
        self.composite_controller_specific_config["ref_name"] = valid_ref_names

        # Check all the ik posture weights exist in the robot model
        ik_posture_weights = self.composite_controller_specific_config.get("ik_posture_weights", {})
        valid_posture_weights = {}
        for weight_name in ik_posture_weights:
            if weight_name in self.robot_model.joints:
                valid_posture_weights[weight_name] = ik_posture_weights[weight_name]
            else:
                ROBOSUITE_DEFAULT_LOGGER.warning(
                    f"Ik posture weight '{weight_name}' does not exist in the robot model. Removing ..."
                )

        # Update the configuration with only the valid posture weights
        self.composite_controller_specific_config["ik_posture_weights"] = valid_posture_weights

    def _init_joint_action_policy(self):
        joint_names: str = []
        for part_name in self.composite_controller_specific_config["actuation_part_names"]:
            if part_name in self.part_controllers:
                joint_names += self.part_controllers[part_name].joint_names

        default_site_names: List[str] = []
        for arm in ["right", "left"]:
            if arm in self.part_controller_config:
                default_site_names.append(self.part_controller_config[arm]["ref_name"])

        self.joint_action_policy = IKSolverMink(
            model=self.sim.model._model,
            data=self.sim.data._data,
            site_names=self.composite_controller_specific_config["ref_name"]
            if "ref_name" in self.composite_controller_specific_config
            else default_site_names,
            robot_model=self.robot_model.mujoco_model,
            robot_joint_names=joint_names,
            input_type=self.composite_controller_specific_config.get("ik_input_type", "absolute"),
            input_ref_frame=self.composite_controller_specific_config.get("ik_input_ref_frame", "world"),
            input_rotation_repr=self.composite_controller_specific_config.get("ik_input_rotation_repr", "axis_angle"),
            solve_freq=self.composite_controller_specific_config.get("ik_solve_freq", 20),
            posture_weights=self.composite_controller_specific_config.get("ik_posture_weights", {}),
            hand_pos_cost=self.composite_controller_specific_config.get("ik_hand_pos_cost", 1.0),
            hand_ori_cost=self.composite_controller_specific_config.get("ik_hand_ori_cost", 0.5),
            verbose=self.composite_controller_specific_config.get("ik_verbose", False),
        )
