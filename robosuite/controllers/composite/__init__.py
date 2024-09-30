from .composite_controller import CompositeController, HybridMobileBase, WholeBodyIK
from .composite_controller import COMPOSITE_CONTROLLERS_DICT

ALL_COMPOSITE_CONTROLLERS = COMPOSITE_CONTROLLERS_DICT.keys()


def composite_controller_factory(type, sim, robot_model, grippers, lite_physics):
    assert type in COMPOSITE_CONTROLLERS_DICT, f"{type} controller is specified, but not imported or loaded"
    # Note: Currently we assume that the init arguments are same for all composite controllers. The situation might change given new controllers in the future, and we will adjust accodingly.

    # The default composite controllers are explicitly initialized without using the COMPOSITE_CONTORLLERS
    if type == "DEFAULT":
        return CompositeController(sim, robot_model, grippers, lite_physics)
    elif type == "HYBRID_MOBILE_BASE":
        return HybridMobileBase(sim, robot_model, grippers, lite_physics)
    elif type == "WHOLE_BODY_IK":
        return WholeBodyIK(sim, robot_model, grippers, lite_physics)
    else:
        return COMPOSITE_CONTROLLERS_DICT[type](sim, robot_model, grippers, lite_physics)
