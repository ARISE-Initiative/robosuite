from .composite_controller import CompositeController, HybridMobileBaseCompositeController


def composite_controller_factory(type, sim, robot_model, grippers, lite_physics):
    if type == "BASE":
        return CompositeController(sim, robot_model, grippers, lite_physics)
    elif type == "HYBRID_MOBILE_BASE":
        return HybridMobileBaseCompositeController(sim, robot_model, grippers, lite_physics)
    else:
        raise ValueError
