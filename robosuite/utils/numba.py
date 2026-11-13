"""
Numba utils.
"""
import numba

import robosuite.macros as macros


def jit_decorator(func):
    if macros.ENABLE_NUMBA:
        return numba.jit(nopython=True, cache=macros.CACHE_NUMBA)(func)
    return func


_WARMED = False


def warmup_numba():
    """Eagerly compile robosuite's numba-JIT functions. Call once at startup, BEFORE importing torch.

    Observed problem (reproducible on NVIDIA Jetson AGX Orin, aarch64, numba 0.68.1 / llvmlite 0.50.0
    / numpy 1.23.5): robosuite's ``@jit_decorator`` functions compile lazily, on first call. When a
    library that ships its own OpenMP runtime -- notably PyTorch, which bundles ``torch/lib/libgomp.so.1``
    -- has been imported first, numba's lazy JIT *compilation of a new signature* crashes the process
    with a bare ``Fatal Python error: Segmentation fault`` (seen in ``transform_utils.mat2pose`` ->
    ``mat2quat`` during ``env.reset()``, and ``controllers/osc.run_controller`` during ``env.step()``).

    What is established vs. hypothesised:
      * Established by bisection: numba JIT itself works on this aarch64 (a trivial ``@njit`` and an
        ``@njit`` using ``np.linalg.eigh`` both run); robosuite's ``mat2quat`` runs fine in isolation;
        the crash appears only when a *new* signature is compiled *after* ``import torch``; and
        compiling the same signatures *before* ``import torch`` makes the subsequent torch-resident
        calls run correctly. It is therefore an import-order / lazy-compilation dependency. Mitigations
        that do NOT help: ``LD_PRELOAD`` of libgomp, ``GLIBC_TUNABLES=glibc.rtld.optional_static_tls``,
        ``NUMBA_THREADING_LAYER=workqueue``, ``OMP_NUM_THREADS=1``, clearing the numba cache.
      * Hypothesised (not proven at the native-loader level): a static-TLS / OpenMP-runtime interaction
        during LLVM code generation once the second OpenMP runtime is resident.

    Fix: force-compile these functions (including the non-contiguous 3x3 slice ``mat2quat`` receives)
    while no foreign OpenMP runtime is loaded; the cached machine code then runs correctly after torch
    is imported. JIT stays fully enabled -- this is a compatibility fix, not ``NUMBA_DISABLE_JIT``.

    No-op when numba is disabled (``macros.ENABLE_NUMBA`` false) or on repeat calls. The core
    orientation functions are warmed eagerly and will raise if they cannot be compiled; the controller
    helpers are best-effort and any that fail to warm are reported via ``warnings.warn`` (so a residual
    crash can be diagnosed) rather than silently swallowed.

    Contributed by VEIZIK (https://github.com/veizikhq), found while running VLA policies on a
    NVIDIA Jetson AGX Orin (aarch64).
    """
    global _WARMED
    if _WARMED or not macros.ENABLE_NUMBA:
        return
    import numpy as np
    import robosuite.utils.transform_utils as T
    import robosuite.utils.control_utils as C

    # --- required: the rotation helpers that crash env.reset(). mat2quat is called with a
    # NON-contiguous 3x3 float64 (a [:3, :3] slice of a 4x4 homogeneous matrix) as well as contiguous
    # arrays -- compile both layouts. These raise on failure (a failure here means the fix is broken).
    hmat = np.eye(4, dtype=np.float64)
    hmat[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T.mat2quat(hmat[:3, :3])
    T.mat2quat(np.ascontiguousarray(hmat[:3, :3]))
    T.quat2mat(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
    if hasattr(T, "matrix_inverse"):
        T.matrix_inverse(np.eye(4, dtype=np.float64))

    # --- best-effort: arm-controller helpers that crash env.step(). numba signatures depend on
    # dtype/ndim, not the exact dimensions, so canonical 7-DOF shapes compile the signatures the OSC
    # controller uses. Failures are surfaced (not hidden) so a residual segfault can be diagnosed.
    dof = 7
    mass = np.eye(dof, dtype=np.float64)
    J_full = np.ones((6, dof), dtype=np.float64)
    J_pos = np.ones((3, dof), dtype=np.float64)
    J_ori = np.ones((3, dof), dtype=np.float64)
    jp = np.zeros(dof, dtype=np.float64)
    jv = np.zeros(dof, dtype=np.float64)
    failed = []
    for name, call in (
        ("opspace_matrices", lambda: C.opspace_matrices(mass, J_full, J_pos, J_ori)),
        ("nullspace_torques", lambda: C.nullspace_torques(mass, np.eye(dof, dtype=np.float64), jp, jp, jv)),
        ("orientation_error", lambda: C.orientation_error(np.eye(3, dtype=np.float64), np.eye(3, dtype=np.float64))),
    ):
        try:
            call()
        except Exception as exc:  # noqa: BLE001 -- surface, do not hide
            failed.append("%s (%s)" % (name, exc))
    if failed:
        import warnings
        warnings.warn(
            "warmup_numba: could not pre-compile %s; if you import torch before the first env.step() "
            "on aarch64 these may still trigger a numba-compile segfault. Compile them manually with "
            "representative inputs before importing torch." % ", ".join(failed),
            RuntimeWarning,
        )
    _WARMED = True
