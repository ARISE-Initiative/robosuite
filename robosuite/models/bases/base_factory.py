"""
Defines a string based method of initializing mounts
"""

from typing import Optional

from robosuite.models.bases.base_model import BaseModel


def base_factory(name: Optional[str], idn=0) -> BaseModel:
    """
    Generator for grippers

    Creates a BaseModel instance with the provided name.

    Args:
        name (None or str): the name of the mount class
        idn (int or str): Number or some other unique identification string for this mount instance

    Returns:
        Base Model class: e.g. MobileBaseModel, LegBaseModel, or MountModel instance

    Raises:
        XMLError: [invalid XML]
    """
    # Import MOUNT_MAPPING at runtime so we avoid circular imports
    from robosuite.models.bases import BASE_MAPPING

    return BASE_MAPPING.get(name, "Unknown base name: {}".format(name))(idn=idn)
