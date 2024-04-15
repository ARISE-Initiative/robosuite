from .composite_controller import CompositeController
from .wheeled_robot_controller import WheeledRobotController


def composite_controller_factory(name, sim, robot_model, grippers):
    if name == "BASE":
        return CompositeController(sim, robot_model, grippers)
    elif name == "WHEELED_ROBOT_CONTROLLER":
        return WheeledRobotController(sim, robot_model, grippers)
