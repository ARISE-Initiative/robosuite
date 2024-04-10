from .base_controller_manager import BaseControllerManager

def controller_manager_factory(name, sim, robot_model, grippers):
    if name == "BASE":
        return BaseControllerManager(sim, robot_model, grippers)