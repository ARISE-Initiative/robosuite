"""
Defines a string based method of initializing grippers
"""


def gripper_factory(name, idn=0):
    """
    Generator for grippers

    Creates a GripperModel instance with the provided name.

    Args:
        name (None or str): the name of the gripper class
        idn (int or str): Number or some other unique identification string for this gripper instance

    Returns:
        GripperModel: requested gripper instance

    Raises:
        XMLError: [invalid XML]
    """
    # Import GRIPPER_MAPPING at runtime so we avoid circular imports
    from robosuite.models.grippers import ALL_EEFS, EEF_MAPPING

    # Make sure gripper is valid
    assert name in EEF_MAPPING, f"Unknown end effector name: {name}. Valid options are: {ALL_EEFS}"

    # Generate gripper
    return EEF_MAPPING[name](idn=idn)
