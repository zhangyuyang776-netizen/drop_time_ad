"""
SciPy-based linear solver backend for small systems (Windows/SciPy workflow, Step 6).

Design goals:
- Pure SciPy/NumPy (no PETSc dependency).
- API mirrors future PETSc backend: returns LinearSolveResult.
- Strict shape checks; no state/layout mutations.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from core.types import CaseConfig
from solvers.linear_types import LinearSolveResult

logger = logging.getLogger(__name__)

def _as_csr(A) -> sp.csr_matrix:
    """Ensure matrix is CSR sparse format."""
    if sp.isspmatrix_csr(A):
        return A
    if sp.isspmatrix(A):
        return A.tocsr()
    if isinstance(A, np.ndarray):
        if A.ndim != 2:
            raise TypeError(f"Expected 2D array for A, got ndim={A.ndim}")
        return sp.csr_matrix(A)
    raise TypeError(f"Unsupported matrix type for A: {type(A)}")


def solve_linear_system_scipy(
    A,
    b: np.ndarray,
    cfg: CaseConfig,
    x0: Optional[np.ndarray] = None,
    method: str = "direct",
) -> LinearSolveResult:
    """Solve Ax=b using SciPy; default direct sparse solve."""
    A_csr = _as_csr(A)
    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError(f"A must be square, got shape {A_csr.shape}")
    N = A_csr.shape[0]

    b = np.asarray(b, dtype=np.float64)
    if b.shape != (N,):
        raise ValueError(f"b shape {b.shape} does not match A dimension {N}")
    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != (N,):
            raise ValueError(f"x0 shape {x0.shape} does not match A dimension {N}")

    logger.debug(
        "solve_linear_system_scipy: case=%s size=%s method=%s",
        getattr(cfg.case, "id", "unknown"),
        A_csr.shape,
        method,
    )

    rtol = float(getattr(cfg.petsc, "rtol", 1e-8))
    atol = float(getattr(cfg.petsc, "atol", 1e-12))

    if method == "direct":
        try:
            x_raw = spla.spsolve(A_csr, b)
        except Exception as exc:
            msg = f"spsolve failed: {exc}"
            logger.error(msg)
            raise RuntimeError(msg) from exc
        n_iter = 1
        solve_method = "direct_spsolve"
        info = 0
    elif method == "cg":
        raise NotImplementedError("CG solver not implemented in SciPy backend yet.")
    elif method == "gmres":
        raise NotImplementedError("GMRES solver not implemented in SciPy backend yet.")
    else:
        raise ValueError(f"Unknown method: {method}")

    r = b - A_csr.dot(x_raw)
    res_norm = float(np.linalg.norm(r))
    b_norm = float(np.linalg.norm(b))
    rel = res_norm / (b_norm + 1e-30)
    converged = res_norm <= max(atol, rtol * b_norm)

    if not converged:
        logger.warning(
            "Linear solve not converged: residual=%.3e rel=%.3e method=%s rtol=%.3e atol=%.3e",
            res_norm,
            rel,
            solve_method if method == "direct" else method,
            rtol,
            atol,
        )
    else:
        logger.debug(
            "Linear solve converged: residual=%.3e rel=%.3e method=%s",
            res_norm,
            rel,
            solve_method if method == "direct" else method,
        )

    return LinearSolveResult(
        x=np.asarray(x_raw, dtype=np.float64),
        converged=converged,
        n_iter=n_iter,
        residual_norm=res_norm,
        rel_residual=rel,
        method=solve_method if method == "direct" else method,
        message=None if converged else "Residual above tolerance",
    )
