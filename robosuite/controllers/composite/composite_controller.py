from collections import OrderedDict
import json
import re

from typing import Optional, Dict

import numpy as np

from robosuite.controllers import controller_factory
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.ik_utils import IKSolver, get_nullspace_gains
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

COMPOSITE_CONTROLLERS_DICT = {}

def register_composite_controller(target_class):
    if not hasattr(target_class, "name"):
        ROBOSUITE_DEFAULT_LOGGER.warning("The name of the composite controller is not specified. Using the class name as the key.")
        key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).upper()
    else:
        key = target_class.name
    COMPOSITE_CONTROLLERS_DICT[key] = target_class
    return target_class

@register_composite_controller
class CompositeController:
    """This is the basic class for composite controller. If you want to develop an advanced version of your controller, you should subclass from this composite controller."""
    name="BASE"

    def __init__(self, sim: MjSim, robot_model: RobotModel, grippers: Dict[str, GripperModel], lite_physics: bool = False):
        # TODO: grippers repeat with members inside robot_model. Currently having this additioanl field to make naming query easy.
        self.sim = sim
        self.robot_model = robot_model

        self.grippers = grippers

        self.lite_physics = lite_physics

        self.controllers = OrderedDict()

        self._action_split_indexes = OrderedDict()

        self.controller_config = None

        self.arms = self.robot_model.arms

        self._applied_action_dict = {}

    def load_controller_config(self, controller_config, composite_controller_specific_config: Optional[Dict] = None):
        self.controller_config = controller_config
        self.composite_controller_specific_config = composite_controller_specific_config
        self.controllers.clear()
        self._action_split_indexes.clear()
        self._init_controllers()
        self.setup_action_split_idx()

    def _init_controllers(self):
        for part_name in self.controller_config.keys():
            self.controllers[part_name] = controller_factory(
                self.controller_config[part_name]["type"], self.controller_config[part_name]
            )

    def setup_action_split_idx(self):
        previous_idx = 0
        last_idx = 0
        for part_name, controller in self.controllers.items():
            if part_name in self.grippers.keys():
                last_idx += self.grippers[part_name].dof
            else:
                last_idx += controller.control_dim
            self._action_split_indexes[part_name] = (previous_idx, last_idx)
            previous_idx = last_idx

    def set_goal(self, all_action):
        if not self.lite_physics:
            self.sim.forward()

        for part_name, controller in self.controllers.items():
            start_idx, end_idx = self._action_split_indexes[part_name]
            action = all_action[start_idx:end_idx]
            if part_name in self.grippers.keys():
                action = self.grippers[part_name].format_action(action)
            controller.set_goal(action)

    def reset(self):
        for part_name, controller in self.controllers.items():
            controller.reset_goal()

    def run_controller(self, enabled_parts):
        if not self.lite_physics:
            self.sim.forward()
        self.update_state()
        self._applied_action_dict.clear()
        for part_name, controller in self.controllers.items():
            if enabled_parts.get(part_name, False):
                self._applied_action_dict[part_name] = controller.run_controller()

        return self._applied_action_dict

    def get_control_dim(self, part_name):
        if part_name not in self.controllers:
            return 0
        else:
            return self.controllers[part_name].control_dim

    def get_controller_base_pose(self, controller_name):
        naming_prefix = self.controllers[controller_name].naming_prefix
        part_name = self.controllers[controller_name].part_name
        base_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(f"{naming_prefix}{part_name}_center")])
        base_ori = np.array(
            self.sim.data.site_xmat[self.sim.model.site_name2id(f"{naming_prefix}{part_name}_center")].reshape([3, 3])
        )
        return base_pos, base_ori

    def update_state(self):
        for arm in self.arms:
            base_pos, base_ori = self.get_controller_base_pose(controller_name=arm)
            self.controllers[arm].update_origin(base_pos, base_ori)

    def get_controller(self, part_name):
        return self.controllers[part_name]

    @property
    def action_limits(self):
        low, high = [], []
        for part_name, controller in self.controllers.items():
            if part_name not in self.arms:
                if part_name in self.grippers.keys():
                    low_g, high_g = ([-1] * self.grippers[part_name].dof, [1] * self.grippers[part_name].dof)
                    low, high = np.concatenate([low, low_g]), np.concatenate([high, high_g])
                else:
                    control_dim = controller.control_dim
                    low_c, high_c = ([-1] * control_dim, [1] * control_dim)
                    low, high = np.concatenate([low, low_c]), np.concatenate([high, high_c])
            else:
                low_c, high_c = controller.control_limits
                low, high = np.concatenate([low, low_c]), np.concatenate([high, high_c])
        return low, high

@register_composite_controller
class HybridMobileBaseCompositeController(CompositeController):
    name="HYBRID_MOBILE_BASE"
    def set_goal(self, all_action):
        if not self.lite_physics:
            self.sim.forward()

        action_mode = all_action[-1]
        if action_mode > 0:
            update_wrt_origin = True
        else:
            update_wrt_origin = False

        for part_name, controller in self.controllers.items():
            start_idx, end_idx = self._action_split_indexes[part_name]
            action = all_action[start_idx:end_idx]
            if part_name in self.grippers.keys():
                action = self.grippers[part_name].format_action(action)

            if part_name in self.arms:
                controller.set_goal(action, update_wrt_origin=update_wrt_origin)
            else:
                controller.set_goal(action)

    @property
    def action_limits(self):
        low, high = super().action_limits
        return np.concatenate((low, [-1])), np.concatenate((high, [1]))

@register_composite_controller
class WholeBodyCompositeController(CompositeController):
    name="WHOLE_BODY_COMPOSITE"
    def __init__(self, sim: MjSim, robot_model: RobotModel, grippers: Dict[str, GripperModel], lite_physics: bool = False):
        super().__init__(sim, robot_model, grippers, lite_physics)

        self.joint_action_policy: IKSolver = None  
        # TODO: handle different types of joint action policies; joint_action_policy maps 
        # task space actions (such as end effector poses) to joint actions (such as joint angles or joint torques)

        self._whole_body_controller_action_split_indexes: OrderedDict = OrderedDict()

    def _init_controllers(self):
        for part_name in self.controller_config.keys():
            controller_params = self.controller_config[part_name]
            self.controllers[part_name] = controller_factory(
                self.controller_config[part_name]["type"], 
                controller_params
            )

        self._init_joint_action_policy()


    def _init_joint_action_policy(self):
        """Joint action policy initialization.
        
        Joint action policy converts input targets (such as end-effector poses, head poses) to joint actions 
        (such as joint angles or joint torques).

        Examples of joint_action_policy could be an IK policy, a neural network policy, a model predictive controller, etc.
        """
        raise NotImplementedError("WholeBodyCompositeController requires a joint action policy")


    def setup_action_split_idx(self):
        """
        Action split indices for the underlying factorized controllers.

        WholeBody controller takes in a different action space from the
        underlying factorized controllers for individual body parts.
        """
        previous_idx = 0
        last_idx = 0
        # add joint_action_policy related body parts' action split index first
        for part_name in self.composite_controller_specific_config["individual_part_names"]:
            last_idx += self.controllers[part_name].control_dim
            self._action_split_indexes[part_name] = (previous_idx, last_idx)
            previous_idx = last_idx

        for part_name, controller in self.controllers.items():
            if part_name not in self.composite_controller_specific_config["individual_part_names"]:
                if part_name in self.grippers.keys():
                    last_idx += self.grippers[part_name].dof
                else:
                    last_idx += controller.control_dim
                self._action_split_indexes[part_name] = (previous_idx, last_idx)
                previous_idx = last_idx

        self.setup_whole_body_controller_action_split_idx()

    def setup_whole_body_controller_action_split_idx(self):
        """
        Action split indices for the composite controller's input action space.

        WholeBodyIK controller takes in a different action space from the
        underlying factorized controllers.
        """
        # add joint_action_policy's action split indexes first
        self._whole_body_controller_action_split_indexes.update(self.joint_action_policy.action_split_indexes())

        # prev and last index correspond to the IK solver indexes' last index
        previous_idx = last_idx = list(self._whole_body_controller_action_split_indexes.values())[-1][-1]
        for part_name, controller in self.controllers.items():
            if part_name in self.composite_controller_specific_config["individual_part_names"]:
                continue
            if part_name in self.grippers.keys():
                last_idx += self.grippers[part_name].dof
            else:
                last_idx += controller.control_dim
            self._whole_body_controller_action_split_indexes[part_name] = (previous_idx, last_idx)
            previous_idx = last_idx

    def set_goal(self, all_action):
        if not self.lite_physics:
            self.sim.forward()

        target_qpos = self.joint_action_policy.solve(all_action[:self.joint_action_policy.control_dim])
        # create new all_action vector with the IK solver's actions first
        all_action = np.concatenate([target_qpos, all_action[self.joint_action_policy.control_dim:]])
        for part_name, controller in self.controllers.items():
            start_idx, end_idx = self._action_split_indexes[part_name]
            action = all_action[start_idx:end_idx]
            if part_name in self.grippers.keys():
                action = self.grippers[part_name].format_action(action)
            controller.set_goal(action)

    def update_state(self):
        # no need for extra update state here, since Jacobians are computed inside the controllers of individual body parts
        return

    @property
    def action_limits(self):
        """
        Returns the action limits for the whole body controller.
        Corresponds to each term in the action vector passed to env.step().
        """
        low, high = [], []
        # assumption: IK solver's actions come first
        low_c, high_c = self.joint_action_policy.control_limits
        low, high = np.concatenate([low, low_c]), np.concatenate([high, high_c])
        for part_name, controller in self.controllers.items():
            # Exclude terms that the IK solver handles
            if part_name in self.composite_controller_specific_config["individual_part_names"]:
                continue
            if part_name not in self.arms:
                if part_name in self.grippers.keys():
                    low_g, high_g = ([-1] * self.grippers[part_name].dof, [1] * self.grippers[part_name].dof)
                    low, high = np.concatenate([low, low_g]), np.concatenate([high, high_g])
                else:
                    control_dim = controller.control_dim
                    low_c, high_c = ([-1] * control_dim, [1] * control_dim)
                    low, high = np.concatenate([low, low_c]), np.concatenate([high, high_c])
            else:
                low_c, high_c = controller.control_limits
                low, high = np.concatenate([low, low_c]), np.concatenate([high, high_c])
        return low, high

    def create_action_vector(self, action_dict: Dict[str, np.ndarray]) -> np.ndarray:
        full_action_vector = np.zeros(self.action_limits[0].shape)
        for (part_name, action_vector) in action_dict.items():
            if part_name not in self._whole_body_controller_action_split_indexes:
                ROBOSUITE_DEFAULT_LOGGER.debug(f"{part_name} is not specified in the action space")
                continue
            start_idx, end_idx = self._whole_body_controller_action_split_indexes[part_name]
            if end_idx - start_idx == 0:
                # skipping not controlling actions
                continue
            assert len(action_vector) == (end_idx - start_idx), ROBOSUITE_DEFAULT_LOGGER.error(
                f"Action vector for {part_name} is not the correct size. Expected {end_idx - start_idx} for {part_name}, got {len(action_vector)}"
            )
            full_action_vector[start_idx:end_idx] = action_vector
        return full_action_vector

    def print_action_info(self):
        action_index_info = []
        action_dim_info = []
        for part_name, (start_idx, end_idx) in self._whole_body_controller_action_split_indexes.items():
            action_dim_info.append(f"{part_name}: {(end_idx - start_idx)} dim")
            action_index_info.append(f"{part_name}: {start_idx}:{end_idx}")

        action_dim_info_str = ", ".join(action_dim_info)
        ROBOSUITE_DEFAULT_LOGGER.info(f"Action Dimensions: [{action_dim_info_str}]")

        action_index_info_str = ", ".join(action_index_info)
        ROBOSUITE_DEFAULT_LOGGER.info(f"Action Indices: [{action_index_info_str}]")

    def print_action_info_dict(self, name: str = ""):
        info_dict = {}
        info_dict["Action Dimension"] = self.action_limits[0].shape
        info_dict.update(dict(self._whole_body_controller_action_split_indexes))
        
        info_dict_str = f"\nAction Info for {name}:\n\n{json.dumps(dict(info_dict), indent=4)}"
        ROBOSUITE_DEFAULT_LOGGER.info(info_dict_str)

@register_composite_controller
class WholeBodyIKCompositeController(WholeBodyCompositeController):
    name="WHOLE_BODY_IK"
    def __init__(self, sim: MjSim, robot_model: RobotModel, grippers: Dict[str, GripperModel], lite_physics: bool = False):
        super().__init__(sim, robot_model, grippers, lite_physics)


    def _init_joint_action_policy(self):
        joint_names: str = []
        for part_name in self.composite_controller_specific_config["individual_part_names"]:
            joint_names += self.controllers[part_name].joint_names

        Kn = get_nullspace_gains(joint_names, self.composite_controller_specific_config["nullspace_joint_weights"])
        mocap_bodies = []
        robot_config = {
            'end_effector_sites': self.composite_controller_specific_config["ref_name"],
            'joint_names': joint_names,
            'mocap_bodies': mocap_bodies,
            'nullspace_gains': Kn
        }
        self.joint_action_policy = IKSolver(
            model=self.sim.model._model,
            data=self.sim.data._data, 
            robot_config=robot_config,
            damping=self.composite_controller_specific_config.get("ik_pseudo_inverse_damping", 5e-2),
            integration_dt=self.composite_controller_specific_config.get("ik_integration_dt", 0.1),
            max_dq=self.composite_controller_specific_config.get("ik_max_dq", 4),
            max_dq_torso=self.composite_controller_specific_config.get("ik_max_dq_torso", 0.2),
            input_rotation_repr=self.composite_controller_specific_config.get("ik_input_rotation_repr", "axis_angle"),
            input_action_repr=self.composite_controller_specific_config.get("ik_input_action_repr", "axis_angle"),
            debug=self.composite_controller_specific_config.get("ik_debug", False),
        )
