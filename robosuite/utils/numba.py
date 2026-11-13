"""
Numba utils.
"""
import numba

import robosuite.macros as macros


# Registry of the functions wrapped by jit_decorator, so warmup_numba() is discoverable/extensible.
_JIT_WRAPPED = []


def jit_decorator(func):
    if macros.ENABLE_NUMBA:
        jitted = numba.jit(nopython=True, cache=macros.CACHE_NUMBA)(func)
        _JIT_WRAPPED.append(jitted)
        return jitted
    return func


_WARMED = False


def warmup_numba():
    """Eagerly compile robosuite's numba-JIT functions. Call once at startup, BEFORE importing torch.

    On aarch64 (e.g. NVIDIA Jetson), numba's lazy JIT compilation of a new signature segfaults once a
    library that bundles its own OpenMP runtime -- most notably PyTorch (``torch/lib/libgomp.so.1``)
    -- is resident (a static-TLS / OpenMP-runtime interaction during LLVM codegen). numba JIT itself
    works on aarch64, and already-compiled code runs fine after torch loads; only compilation AFTER
    the OpenMP runtime loads crashes. In a VLA pipeline (a robosuite/LIBERO env + a torch policy),
    robosuite's mat2quat / OSC-controller functions compile on first call during env.reset()/step()
    -- i.e. after ``import torch`` -- so the process dies with a bare "Fatal Python error:
    Segmentation fault" in transform_utils.mat2pose or controllers/osc.run_controller.

    Calling this first force-compiles every signature these functions are used with (including the
    non-contiguous 3x3 slice mat2quat receives) while no foreign OpenMP runtime is loaded; the cached
    machine code then runs fine after torch is imported. JIT stays fully enabled (no need for
    NUMBA_DISABLE_JIT). No-op when numba is disabled or on repeat calls.
    """
    global _WARMED
    if _WARMED or not macros.ENABLE_NUMBA:
        return
    import numpy as np
    import robosuite.utils.transform_utils as T
    import robosuite.utils.control_utils as C

    # transform_utils: mat2quat is called with a NON-contiguous 3x3 float64 (a [:3, :3] slice of a 4x4
    # homogeneous matrix) as well as contiguous arrays -- compile both layouts.
    hmat = np.eye(4, dtype=np.float64)
    hmat[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T.mat2quat(hmat[:3, :3])
    T.mat2quat(np.ascontiguousarray(hmat[:3, :3]))
    T.quat2mat(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
    if hasattr(T, "matrix_inverse"):
        T.matrix_inverse(np.eye(4, dtype=np.float64))

    # control_utils: arm controllers use a DOF x DOF mass matrix and task Jacobians; numba signatures
    # depend on dtype/ndim, not the exact dimensions, so canonical 7-DOF shapes compile the right ones.
    dof = 7
    mass = np.eye(dof, dtype=np.float64)
    J_full = np.ones((6, dof), dtype=np.float64)
    J_pos = np.ones((3, dof), dtype=np.float64)
    J_ori = np.ones((3, dof), dtype=np.float64)
    jp = np.zeros(dof, dtype=np.float64)
    jv = np.zeros(dof, dtype=np.float64)
    for call in (
        lambda: C.opspace_matrices(mass, J_full, J_pos, J_ori),
        lambda: C.nullspace_torques(mass, np.eye(dof, dtype=np.float64), jp, jp, jv),
        lambda: C.orientation_error(np.eye(3, dtype=np.float64), np.eye(3, dtype=np.float64)),
    ):
        try:
            call()
        except Exception:
            pass  # a signature mismatch here is harmless; the goal is only to trigger compilation
    _WARMED = True
