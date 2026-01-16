from __future__ import annotations

from solvers.mpi_linear_support import validate_mpi_linear_support


def validate_mpi_before_petsc(cfg) -> None:
    """
    Entry-point MPI validation hook (fail-fast before PETSc objects are created).
    """
    validate_mpi_linear_support(cfg)
