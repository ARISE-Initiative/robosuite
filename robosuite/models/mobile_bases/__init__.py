from .mobile_base_model import MobileBaseModel
from .mobile_base_factory import mobile_base_factory

from .omron_mobile_base import OmronMobileBase


MOBILE_BASE_MAPPING = {
    "OmronMobileBase": OmronMobileBase,
}

ALL_MOBILE_BASES = MOBILE_BASE_MAPPING.keys()
