"""
Helpers for converting Jacobian sparsity patterns into PETSc AIJ preallocation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from assembly.jacobian_pattern import JacobianPattern


def build_petsc_prealloc_from_pattern(
    pattern: JacobianPattern,
) -> Tuple[int, List[int], List[int]]:
    """
    Convert a global CSR JacobianPattern into per-row nnz estimates for PETSc AIJ.
    """
    n_rows, n_cols = pattern.shape
    if n_rows != n_cols:
        raise ValueError(f"JacobianPattern must be square, got shape={pattern.shape}.")

    indptr = pattern.indptr
    if indptr.shape[0] != n_rows + 1:
        raise ValueError(
            "JacobianPattern.indptr length mismatch: "
            f"len(indptr)={indptr.shape[0]}, expected {n_rows + 1}."
        )

    d_nz = np.diff(indptr).astype(int).tolist()
    o_nz = [0] * n_rows

    nnz_from_indptr = int(indptr[-1])
    nnz_from_indices = int(pattern.indices.size)
    if nnz_from_indptr != nnz_from_indices:
        raise ValueError(
            "JacobianPattern nnz mismatch: "
            f"indptr[-1]={nnz_from_indptr}, len(indices)={nnz_from_indices}."
        )

    return n_rows, d_nz, o_nz
