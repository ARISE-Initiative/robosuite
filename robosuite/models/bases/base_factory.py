"""
Defines a string based method of initializing mounts
"""

from typing import Optional

from robosuite.models.bases.mount_model import MountModel


def base_factory(name: Optional[str], idn=0) -> MountModel:
    """
    Generator for grippers

    Creates a MountModel instance with the provided name.

    Args:
        name (None or str): the name of the mount class
        idn (int or str): Number or some other unique identification string for this mount instance

    Returns:
        MountModel: requested mount instance

    Raises:
        XMLError: [invalid XML]
    """
    # Import MOUNT_MAPPING at runtime so we avoid circular imports
    from robosuite.models.bases import BASE_MAPPING

    return BASE_MAPPING.get(name, "Unknown base name: {}".format(name))(idn=idn)
