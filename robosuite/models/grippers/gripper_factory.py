"""
Defines a string based method of initializing grippers
"""


def gripper_factory(name, idn=0):
    """
    Generator for grippers

    Creates a GripperModel instance with the provided name.

    Args:
        name (str): the name of the gripper class
        idn (int or str): Number or some other unique identification string for this gripper instance

    Returns:
        GripperModel: requested gripper instance

    Raises:
        XMLError: [invalid XML]
    """
    # Import GRIPPER_MAPPING at runtime so we avoid circular imports
    from robosuite.models.grippers import GRIPPER_MAPPING
    return GRIPPER_MAPPING.get(name, "Unknown gripper name: {}".format(name))(idn=idn)
