from robosuite.robots.mobile_robot import MobileRobot


class LeggedRobot(MobileRobot):
    def _load_controller(self):
        raise NotImplementedError

    def reset(self, deterministic=False):
        raise NotImplementedError

    def setup_references(self):
        raise NotImplementedError

    def control(self, action, policy_step=False):
        raise NotImplementedError

    @property
    def action_limits(self):
        raise NotImplementedError

    @property
    def _action_split_idx(self):
        raise NotImplementedError

    @property
    def _joint_split_idx(self):
        raise NotImplementedError
