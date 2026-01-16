from __future__ import annotations

import logging
import os
import sys
from typing import Optional


_TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


def _parse_level(value, default_level: int) -> int:
    if value is None:
        return default_level
    if isinstance(value, int):
        return int(value)
    text = str(value).strip().upper()
    if not text:
        return default_level
    if text.isdigit():
        try:
            return int(text)
        except Exception:
            return default_level
    resolved = logging.getLevelName(text)
    if isinstance(resolved, int):
        return resolved
    return default_level


def is_root_rank(comm=None) -> bool:
    """
    Return True on rank 0 when MPI/PETSc is available; otherwise default True.

    This is safe before PETSc initialization: we try mpi4py first and only
    consult petsc4py if already imported.
    """
    if comm is not None:
        try:
            if hasattr(comm, "getRank"):
                return int(comm.getRank()) == 0
        except Exception:
            pass
        try:
            if hasattr(comm, "Get_rank"):
                return int(comm.Get_rank()) == 0
        except Exception:
            pass

    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank()) == 0
    except Exception:
        pass

    try:
        if "petsc4py" in sys.modules:
            from petsc4py import PETSc

            return int(PETSc.COMM_WORLD.getRank()) == 0
    except Exception:
        pass

    return True


def get_log_level_from_env(default: str | int = "INFO") -> int:
    """
    Resolve log level from env (DROPLET_LOG_LEVEL or DROPLET_PETSC_DEBUG).
    """
    default_level = _parse_level(default, logging.INFO)
    env_level = os.environ.get("DROPLET_LOG_LEVEL")
    if env_level:
        return _parse_level(env_level, default_level)
    if _is_truthy(os.environ.get("DROPLET_PETSC_DEBUG")):
        return logging.DEBUG
    return default_level


def setup_logging(rank: int, *, level: int, quiet_nonroot: bool = True) -> None:
    """
    Configure root logging once and quiet non-root console handlers by default.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    root.setLevel(level)

    if quiet_nonroot and rank != 0:
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                continue
            handler.setLevel(max(level, logging.WARNING))
    else:
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                continue
            handler.setLevel(level)
