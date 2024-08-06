from .composite_controller import CompositeController, HybridMobileBaseCompositeController, WholeBodyIKCompositeController


def composite_controller_factory(type, sim, robot_model, grippers, lite_physics):
    if type == "BASE":
        return CompositeController(sim, robot_model, grippers, lite_physics)
    elif type == "HYBRID_MOBILE_BASE":
        return HybridMobileBaseCompositeController(sim, robot_model, grippers, lite_physics)
    elif type == "WHOLE_BODY_IK":
        return WholeBodyIKCompositeController(sim, robot_model, grippers, lite_physics)
    else:
        raise ValueError
