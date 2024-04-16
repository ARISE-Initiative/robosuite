from .composite_controller import CompositeController
from .floating_robot_controller import FloatingRobotController


def composite_controller_factory(name, sim, robot_model, grippers):
    if name == "BASE":
        return CompositeController(sim, robot_model, grippers)
    elif name == "FLOATING_ROBOT_CONTROLLER":
        return FloatingRobotController(sim, robot_model, grippers)
