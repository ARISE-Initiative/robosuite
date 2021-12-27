"""
Numba utils.
"""
import numba

import robosuite.utils.macros as macros


def jit_decorator(func):
    if macros.ENABLE_NUMBA:
        return numba.jit(nopython=True, cache=macros.CACHE_NUMBA)(func)
    return func
