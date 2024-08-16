from robosuite.robots.manipulator import Manipulator


class MobileManipulator(Manipulator):
    """
    Variant of Manipualtor with mobile base support. Currently serves as placeholder class.
    """

    pass

    @property
    def is_mobile(self):
        return True
