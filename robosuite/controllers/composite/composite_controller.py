from collections import OrderedDict
import json
from typing import Optional, Dict

import numpy as np

from robosuite.controllers import controller_factory
from robosuite.utils.ik_utils import IKSolver, get_Kn
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER



class CompositeController:
    """This is the basic class for composite controller. If you want to develop an advanced version of your controller, you should subclass from this composite controller."""

    def __init__(self, sim, robot_model, grippers, lite_physics=False):
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


class HybridMobileBaseCompositeController(CompositeController):
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


class WholeBodyIKCompositeController(CompositeController):
    def __init__(self, sim, robot_model, grippers, lite_physics=False):
        super().__init__(sim, robot_model, grippers, lite_physics)

        self.ik_solver: IKSolver = None

        self._whole_body_controller_action_split_indexes: OrderedDict = OrderedDict()

    def _init_controllers(self):
        for part_name in self.controller_config.keys():
            controller_params = self.controller_config[part_name]
            # default controller configs may have these values loaded
            if controller_params.get("sim", None) is None:
                controller_params["sim"] = self.sim
            if controller_params.get("joint_indexes", None) is None:
                controller_params["joint_indexes"] = {
                    "joints": [self.sim.model.joint(joint_name).id for joint_name in self.robot_model.joints],
                    "qpos": [self.sim.model.get_joint_qpos_addr(joint_name) for joint_name in self.robot_model.joints],
                    "qvel": [self.sim.model.get_joint_qvel_addr(joint_name) for joint_name in self.robot_model.joints],
                }
            self.controllers[part_name] = controller_factory(
                self.controller_config[part_name]["type"], 
                controller_params
            )

        joint_names = []
        for part_name in self.composite_controller_specific_config["individual_part_names"]:
            joint_names += self.controllers[part_name].joint_names

        Kn = get_Kn(joint_names, self.composite_controller_specific_config["nullspace_joint_weights"])
        mocap_bodies = []
        robot_config = {
            'end_effector_sites': self.composite_controller_specific_config["ref_name"],
            'joint_names': joint_names,
            'mocap_bodies': mocap_bodies,
            'nullspace_gains': Kn
        }
        self.ik_solver = IKSolver(
            model=self.sim.model._model, 
            data=self.sim.data._data, 
            robot_config=robot_config,
            damping=self.composite_controller_specific_config["ik_pseudo_inverse_damping"],
            integration_dt=self.composite_controller_specific_config["ik_integration_dt"],
            max_dq=self.composite_controller_specific_config["ik_max_dq"],
            input_rotation_repr=self.composite_controller_specific_config["ik_input_rotation_repr"],
        )

    def setup_action_split_idx(self):
        """
        Action split indices for the underlying factorized controllers.

        WholeBodyIK controller takes in a different action space from the
        underlying factorized controllers.
        """
        previous_idx = 0
        last_idx = 0
        # add the IK solver's action split index first -- outputs in the order of individual_part_names
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
        Action split indices for the composite controller's input ation space.

        WholeBodyIK controller takes in a different action space from the
        underlying factorized controllers.
        """
        # add ik solver action split indexes first
        self._whole_body_controller_action_split_indexes.update(self.ik_solver.action_split_indexes())

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

        target_qpos = self.ik_solver.solve_ik(all_action[:self.ik_solver.control_dim])
        # create new all_action vector with the IK solver's actions first
        all_action = np.concatenate([target_qpos, all_action[self.ik_solver.control_dim:]])
        for part_name, controller in self.controllers.items():
            start_idx, end_idx = self._action_split_indexes[part_name]
            action = all_action[start_idx:end_idx]
            if part_name in self.grippers.keys():
                action = self.grippers[part_name].format_action(action)
            controller.set_goal(action)

    def update_state(self):
        # no need for extra update state here, since Jacobians are computed inside the controller
        return

    def run_controller(self, enabled_parts):
        self._applied_action_dict.clear()
        for part_name, controller in self.controllers.items():
            # ignore the enabled_parts for now
            self._applied_action_dict[part_name] = controller.run_controller()
        return self._applied_action_dict

    @property
    def action_limits(self):
        """
        Returns the action limits for the whole body controller.
        Corresponds to each term in the action vector passed to env.step().
        """
        low, high = [], []
        # assumption: IK solver's actions come first
        low_c, high_c = self.ik_solver.control_limits
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
    
    def print_action_info_dict(self, name: str):
        info_dict = {}
        info_dict["Action Dimension"] = self.action_limits[0].shape
        info_dict.update(dict(self._whole_body_controller_action_split_indexes))
        
        info_dict_str = f"\nAction Info for {name}:\n\n{json.dumps(dict(info_dict), indent=4)}"
        ROBOSUITE_DEFAULT_LOGGER.info(info_dict_str)
