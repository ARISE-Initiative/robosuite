"""Regression test for ``robosuite.utils.numba.warmup_numba()``.

On aarch64 with PyTorch co-imported, lazily JIT-compiling robosuite's ``@jit_decorator`` functions
AFTER ``import torch`` segfaults; ``warmup_numba()`` pre-compiles them beforehand. This test runs the
warm-then-use path in a SEPARATE process so a regression surfaces as a non-zero exit code instead of
killing the pytest process with a native segfault.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import robosuite.macros as macros
from robosuite.utils import numba as rsnumba


def test_warmup_numba_callable_and_idempotent():
    """warmup_numba() runs and a second call is a no-op (does not recompile or raise)."""
    rsnumba.warmup_numba()
    rsnumba.warmup_numba()


@pytest.mark.skipif(not macros.ENABLE_NUMBA, reason="numba disabled (macros.ENABLE_NUMBA is False)")
def test_warmup_then_torch_then_jit_in_subprocess():
    """warmup -> import torch -> call the JIT functions: must exit 0 with correct outputs.

    This is the regression for the aarch64 compile-after-torch segfault. torch is optional (the test
    still validates the warm path and output correctness without it); where torch is installed on
    aarch64 it also exercises the exact crash condition the fix addresses.
    """
    code = textwrap.dedent(
        """
        import numpy as np
        import robosuite.utils.numba as rn
        rn.warmup_numba()                      # compile JIT fns BEFORE torch loads its OpenMP runtime
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        import robosuite.utils.transform_utils as T
        # the non-contiguous 3x3 slice signature that crashed env.reset() -> mat2pose -> mat2quat
        hmat = np.eye(4)
        hmat[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        q = T.mat2quat(hmat[:3, :3])
        assert np.allclose(q, [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], atol=1e-5), q
        assert np.allclose(T.mat2quat(np.eye(3)), [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        print("WARMUP_OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        "warmup_numba + torch + JIT call failed (rc=%s).\nstdout:\n%s\nstderr:\n%s"
        % (r.returncode, r.stdout, r.stderr)
    )
    assert "WARMUP_OK" in r.stdout
