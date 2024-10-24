"""
Defines a string based method of initializing mounts
"""

from typing import Optional

from robosuite.models.bases.robot_base_model import RobotBaseModel


def robot_base_factory(name: Optional[str], idn=0) -> RobotBaseModel:
    """
    Generator for grippers

    Creates a RobotBaseModel instance with the provided name.

    Args:
        name (None or str): the name of the mount class
        idn (int or str): Number or some other unique identification string for this mount instance

    Returns:
        RobotBaseModel class: e.g. MobileBaseModel, LegBaseModel, or FixedBaseModel instance

    Raises:
        XMLError: [invalid XML]
    """
    # Import MOUNT_MAPPING at runtime so we avoid circular imports
    from robosuite.models.bases import BASE_MAPPING

    return BASE_MAPPING.get(name, "Unknown base name: {}".format(name))(idn=idn)
