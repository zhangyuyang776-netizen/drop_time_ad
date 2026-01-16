from __future__ import annotations

_BOOTSTRAPPED = False
_PETSC_BOOTSTRAPPED = False


def bootstrap_mpi() -> None:
    """
    Ensure mpi4py initializes before petsc4py.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True

    try:
        from mpi4py import MPI  # noqa: F401
    except Exception:
        return


def bootstrap_mpi_before_petsc() -> None:
    """
    Ensure mpi4py initializes before petsc4py, and pass argv to PETSc.
    """
    bootstrap_mpi()

    global _PETSC_BOOTSTRAPPED
    if _PETSC_BOOTSTRAPPED:
        return
    _PETSC_BOOTSTRAPPED = True

    try:
        import os
        import sys
        import petsc4py
    except Exception:
        return

    argv = [] if os.environ.get("PYTEST_CURRENT_TEST") else sys.argv
    try:
        petsc4py.init(argv)
    except Exception:
        # Ignore double-init or unsupported init paths; PETSc may already be initialized.
        pass
