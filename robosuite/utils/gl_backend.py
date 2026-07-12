from __future__ import annotations

import os
import platform
from typing import Any, MutableMapping


def _normalize_backend(value: str | None) -> str | None:
    if value is None:
        return None
    backend = value.strip().lower()
    return backend or None


def platform_default_backend(system: str | None = None) -> str:
    system_name = platform.system() if system is None else system
    if system_name == "Darwin":
        return "cgl"
    if system_name == "Windows":
        return "wgl"
    return "egl"


def resolve_mujoco_gl_backend(
    macros_module: Any,
    *,
    environ: MutableMapping[str, str] | None = None,
    system: str | None = None,
) -> str:
    environ = os.environ if environ is None else environ

    explicit_backend = _normalize_backend(environ.get("MUJOCO_GL"))
    if explicit_backend is not None:
        return explicit_backend

    if getattr(macros_module, "MUJOCO_GPU_RENDERING", False):
        return platform_default_backend(system=system)

    return ""


def configure_mujoco_gl_backend(
    macros_module: Any,
    *,
    environ: MutableMapping[str, str] | None = None,
    system: str | None = None,
) -> str:
    environ = os.environ if environ is None else environ
    backend = resolve_mujoco_gl_backend(
        macros_module,
        environ=environ,
        system=system,
    )
    if backend and _normalize_backend(environ.get("MUJOCO_GL")) is None:
        environ["MUJOCO_GL"] = backend
    return backend
