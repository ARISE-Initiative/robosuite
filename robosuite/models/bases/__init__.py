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

from .b1_base import B1, B1Floating
from .go2_base import Go2, Go2Floating
from .spot_base import Spot, SpotFloating

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
    "B1Floating": B1Floating,
    "Spot": Spot,
    "SpotFloating": SpotFloating,
    "Go2": Go2,
    "Go2Floating": Go2Floating,
}

ALL_BASES = BASE_MAPPING.keys()

# def register_base_class(target_type, **kwargs):
#     def decorator(target_class):
#         # Store the class in the registry with additional arguments
#         BASE_MAPPING.update({target_class.__name__: target_class})

#         return target_class  # Return the class itself
#     return decorator
