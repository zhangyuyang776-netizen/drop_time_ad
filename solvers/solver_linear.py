"""
Linear solve dispatcher for SciPy/PETSc backends.

This module routes A,b to the selected backend without mixing assembly logic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.types import CaseConfig
from solvers.linear_types import LinearSolveResult
from solvers.scipy_linear import solve_linear_system_scipy
from solvers.petsc_linear import solve_linear_system_petsc


def solve_linear_system(
    A,
    b,
    cfg: CaseConfig,
    x0: Optional[np.ndarray] = None,
    comm=None,
    layout=None,
    method: Optional[str] = None,
) -> LinearSolveResult:
    """Dispatch to SciPy or PETSc linear solver based on cfg.solver.linear.backend."""
    solver_cfg = getattr(cfg, "solver", None)
    linear_cfg = getattr(solver_cfg, "linear", None)
    backend = getattr(linear_cfg, "backend", None)
    if backend is None:
        backend = getattr(getattr(cfg, "nonlinear", None), "backend", "scipy")
    backend = str(backend).lower()

    if backend == "scipy":
        if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError("SciPy backend expects numpy arrays for A and b.")
        if method is None:
            return solve_linear_system_scipy(A=A, b=b, cfg=cfg, x0=x0)
        return solve_linear_system_scipy(A=A, b=b, cfg=cfg, x0=x0, method=method)

    if backend == "petsc":
        try:
            from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

            bootstrap_mpi_before_petsc()
            from petsc4py import PETSc
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("petsc4py is required for PETSc backend.") from exc

        if isinstance(A, PETSc.Mat) and isinstance(b, PETSc.Vec):
            A_p, b_p = A, b
        else:
            if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
                raise TypeError("PETSc backend expects PETSc Mat/Vec or numpy arrays for A and b.")
            from assembly.build_system_petsc import numpy_dense_to_petsc_aij
            A_p, b_p = numpy_dense_to_petsc_aij(A, b, comm=comm)

        if method is None:
            return solve_linear_system_petsc(A=A_p, b=b_p, cfg=cfg, x0=x0, layout=layout)
        return solve_linear_system_petsc(A=A_p, b=b_p, cfg=cfg, x0=x0, layout=layout, method=method)

    raise ValueError(f"Unknown backend '{backend}' for linear solver (expected 'scipy' or 'petsc').")
