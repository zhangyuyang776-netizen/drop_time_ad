"""
Shared nonlinear solver result types.

Goal:
- Backend-agnostic: SciPy and PETSc (SNES) return the same structure.
- Keep timestepper/tests stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class NonlinearBackend(str, Enum):
    SCIPY = "scipy"
    PETSC = "petsc"
    PETSC_MPI = "petsc_mpi"


@dataclass(slots=True)
class NonlinearDiagnostics:
    converged: bool
    method: str
    n_iter: int
    res_norm_2: float
    res_norm_inf: float
    history_res_inf: List[float] = field(default_factory=list)
    message: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NonlinearSolveResult:
    u: np.ndarray
    diag: NonlinearDiagnostics
