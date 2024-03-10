from .mobile_base_model import MobileBaseModel
from .mobile_base_factory import mobile_base_factory

from .omron_mobile_base import OmronMobileBase
from .null_mobile_base import NullMobileBase


MOBILE_BASE_MAPPING = {
    "OmronMobileBase": OmronMobileBase,
    "NullMobileBase": NullMobileBase,
    None: NullMobileBase,
}

ALL_MOBILE_BASES = MOBILE_BASE_MAPPING.keys()
