from .mount_model import MountModel
from .mount_factory import mount_factory

from .rethink_mount import RethinkMount
from .rethink_minimal_mount import RethinkMinimalMount
from .null_mount import NullMount
from .gr1_lowerbody_mount import GR1LowerBodyMount
from .humanoid_free_mount import HumanoidFreeMount


MOUNT_MAPPING = {
    "RethinkMount": RethinkMount,
    "RethinkMinimalMount": RethinkMinimalMount,
    "GR1LowerBodyMount": GR1LowerBodyMount,
    "HumanoidFreeMount": HumanoidFreeMount,
    None: NullMount,
}

ALL_MOUNTS = MOUNT_MAPPING.keys()
