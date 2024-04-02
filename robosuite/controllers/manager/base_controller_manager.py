from collections import OrderedDict


class BaseControllerManager():
    def __init__(self) -> None: 
        self.sim = sim
        self.robot_model = robot_model

        self.controllers = OrderedDict()


        self._action_split_idices = {entity_name: None for entity_name in self.controllers.keys()}


    def setup_action_split_idx(self):
        split_idx = 0
        for entity_name, controller in self.controllers.items():
            self._action_split_idices[entity_name] = split_idx
            split_idx += controller.control_dim
        
    def set_goal(self, action):
        self.sim.forward()

        for entity_name, controller in self.controllers.items():
            controller.set_goal(action[self._action_split_idices[entity_name]])

    def reset(self):
        
    

    def compute_ctrl(self):
        ctrl_dict = {}

        for entity_name, controller in self.controllers.items():
            ctrl_dict[entity_name] = controller.run_controller()

        return self.ctrl_dict


    # def control_masking(self, entity_masking):
    #     """mask out the entities that a user does not want to control at all. Typically used when the user wants to switch between different control mode, e.g., arm control vs wheel-only control.

    #     Args:
    #         entity_masking (_type_): _description_
    #     """
    #     pass