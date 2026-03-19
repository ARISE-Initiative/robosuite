import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from robosuite.utils.gl_backend import configure_mujoco_gl_backend, resolve_mujoco_gl_backend

REPO_ROOT = Path(__file__).resolve().parents[2]


def _macros(**kwargs):
    values = {
        "MUJOCO_GPU_RENDERING": True,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def _run_binding_utils_probe(*, env_updates=None, unset=None):
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else str(REPO_ROOT) + os.pathsep + existing_pythonpath
    )

    if unset is not None:
        for key in unset:
            env.pop(key, None)

    if env_updates is not None:
        env.update(env_updates)

    result = subprocess.run(
        [sys.executable, "-c", "import os, robosuite; print(os.environ.get('MUJOCO_GL', ''))"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def test_explicit_env_backend_is_preserved():
    environ = {"MUJOCO_GL": "glfw"}

    backend = configure_mujoco_gl_backend(_macros(), environ=environ, system="Linux")

    assert backend == "glfw"
    assert environ["MUJOCO_GL"] == "glfw"


def test_linux_gpu_default_remains_egl():
    backend = resolve_mujoco_gl_backend(
        _macros(MUJOCO_GPU_RENDERING=True),
        environ={},
        system="Linux",
    )

    assert backend == "egl"


def test_gpu_default_is_empty_when_gpu_rendering_is_disabled():
    backend = resolve_mujoco_gl_backend(
        _macros(MUJOCO_GPU_RENDERING=False),
        environ={},
        system="Linux",
    )

    assert backend == ""


def test_binding_utils_import_preserves_explicit_env():
    resolved_backend = _run_binding_utils_probe(env_updates={"MUJOCO_GL": "glfw"})

    assert resolved_backend == "glfw"


def test_binding_utils_import_defaults_to_egl_on_linux():
    if platform.system() != "Linux":
        return

    resolved_backend = _run_binding_utils_probe(unset={"MUJOCO_GL"})

    assert resolved_backend == "egl"
