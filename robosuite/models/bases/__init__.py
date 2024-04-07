from .mount_model import MountModel
from .base_factory import base_factory
from .mobile_base_model import MobileBaseModel
from .leg_base_model import LegBaseModel

from .rethink_mount import RethinkMount
from .rethink_minimal_mount import RethinkMinimalMount
from .null_mount import NullMount

from .omron_mobile_base import OmronMobileBase
from .null_mobile_base import NullMobileBase
from .no_actuation_base import NoActuationBase
from .floating_legged_base import FloatingLeggedBase

from .aloha_mount import AlohaMount

from .b1_base import B1

BASE_MAPPING = {
    "RethinkMount": RethinkMount,
    "RethinkMinimalMount": RethinkMinimalMount,
    "AlohaMount": AlohaMount,
    "NullMount": NullMount,
    "OmronMobileBase": OmronMobileBase,
    "NullMobileBase": NullMobileBase,
    "NoActuationBase": NoActuationBase,
    "FloatingLeggedBase": FloatingLeggedBase,
    "B1": B1,
    # "Z1Base": Z1Base,
    # "SpotBase": SpotBase,
    # "NullLeggedBase": NullLeggedBase,
}

ALL_BASES = BASE_MAPPING.keys()
