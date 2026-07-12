from .mount_model import MountModel
from .robot_base_model import RobotBaseModel
from .robot_base_factory import robot_base_factory
from .mobile_base_model import MobileBaseModel
from .leg_base_model import LegBaseModel
from .null_base_model import NullBaseModel

from .rethink_mount import RethinkMount
from .rethink_minimal_mount import RethinkMinimalMount
from .null_mount import NullMount

from .omron_mobile_base import OmronMobileBase
from .null_mobile_base import NullMobileBase
from .no_actuation_base import NoActuationBase
from .floating_legged_base import FloatingLeggedBase
from .floating_legged_base_with_vertical import FloatingLeggedBaseWithVertical
from .null_base import NullBase
from .spot_base import Spot, SpotFloating

BASE_MAPPING = {
    "RethinkMount": RethinkMount,
    "RethinkMinimalMount": RethinkMinimalMount,
    "NullMount": NullMount,
    "OmronMobileBase": OmronMobileBase,
    "NullMobileBase": NullMobileBase,
    "NoActuationBase": NoActuationBase,
    "FloatingLeggedBase": FloatingLeggedBase,
    "FloatingLeggedBaseWithVertical": FloatingLeggedBaseWithVertical,
    "Spot": Spot,
    "SpotFloating": SpotFloating,
    "NullBase": NullBase,
}

ALL_BASES = BASE_MAPPING.keys()


def register_base(target_class):
    BASE_MAPPING[target_class.__name__] = target_class
    return target_class
