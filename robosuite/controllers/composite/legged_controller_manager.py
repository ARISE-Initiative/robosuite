from .base_composite_controller import CompositeController


class LeggedControllerManager(CompositeController):
    def __init__(self, 
                 sim,
                 robot_model,
                 controller_config) -> None: 
        super().__init__(sim, robot_model, controller_config)
        
    def update_state(self):
        for arm in self.arms:
            self.controllers[arm].update_base_pose()

    def run_controller(self):
        return super().run_controller()