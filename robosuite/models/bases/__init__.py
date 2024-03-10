from .mount_model import MountModel
from .base_factory import base_factory

from .rethink_mount import RethinkMount
from .rethink_minimal_mount import RethinkMinimalMount
from .null_mount import NullMount

from .mobile_base_model import MobileBaseModel
from .omron_mobile_base import OmronMobileBase
from .null_mobile_base import NullMobileBase


BASE_MAPPING = {
    "RethinkMount": RethinkMount,
    "RethinkMinimalMount": RethinkMinimalMount,
    None: NullMount,
    "OmronMobileBase": OmronMobileBase,
    "NullMobileBase": NullMobileBase,
    None: NullMobileBase,
}

ALL_BASES = BASE_MAPPING.keys()
