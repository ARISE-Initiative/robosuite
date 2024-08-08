from .composite_controller import CompositeController, HybridMobileBaseCompositeController, WholeBodyIKCompositeController


CONTROLLER_INFO = {
    "BASE": "Base composite controller factory",
    "HYBRID_MOBILE_BASE": "Hybrid mobile base composite controller",
    "WHOLE_BODY_IK": "Whole Body Inverse Kinematics Composite Controller",
}

ALL_CONTROLLERS = CONTROLLER_INFO.keys()


def composite_controller_factory(type, sim, robot_model, grippers, lite_physics):
    if type == "BASE":
        return CompositeController(sim, robot_model, grippers, lite_physics)
    elif type == "HYBRID_MOBILE_BASE":
        return HybridMobileBaseCompositeController(sim, robot_model, grippers, lite_physics)
    elif type == "WHOLE_BODY_IK":
        return WholeBodyIKCompositeController(sim, robot_model, grippers, lite_physics)
    else:
        raise ValueError
