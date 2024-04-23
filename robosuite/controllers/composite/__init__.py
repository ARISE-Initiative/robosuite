from .composite_controller import CompositeController


def composite_controller_factory(name, sim, robot_model, grippers, lite_physics):
    if name == "BASE":
        return CompositeController(sim, robot_model, grippers, lite_physics)
    else:
        raise ValueError
