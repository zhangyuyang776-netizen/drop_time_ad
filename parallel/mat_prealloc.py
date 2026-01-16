# -*- coding: utf-8 -*-
"""
MPI-aware matrix preallocation utilities.

Stage 5.4B Task 2: distributed preallocation helpers (owner / pattern-local).

Features:
- Read PETSc Vec/Mat ownership range(s);
- Build an owner map for global index -> rank;
- Split per-row columns into diag/off blocks and count per-row nnz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


def _get_petsc():
    """
    Import petsc4py.PETSc with MPI bootstrap.

    Note: do not import petsc4py at module import time to avoid failures in
    environments without PETSc.
    """
    try:
        from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

        bootstrap_mpi_before_petsc()
        from petsc4py import PETSc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("petsc4py is required for mat_prealloc utilities.") from exc
    return PETSc


def get_global_ownership_ranges_from_vec(vec) -> np.ndarray:
    """
    Read global ownership ranges from a PETSc Vec.

    Returns a shape=(size+1,) int array where ranges[p] <= j < ranges[p+1]
    is owned by rank p.
    """
    PETSc = _get_petsc()
    if not hasattr(vec, "getOwnershipRanges"):
        raise TypeError(f"Object {type(vec)} does not support getOwnershipRanges().")
    ranges = vec.getOwnershipRanges()
    arr = np.asarray(ranges, dtype=PETSc.IntType)
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = np.concatenate([arr[:1, 0], arr[:, 1]])
    elif arr.ndim != 1:
        arr = arr.ravel()
    return arr


def get_ownership_range_from_mat_or_vec(obj) -> Tuple[int, int]:
    """
    Read the local ownership range [rstart, rend) from a PETSc Vec or Mat.
    """
    if hasattr(obj, "getOwnershipRange"):
        rstart, rend = obj.getOwnershipRange()
        return int(rstart), int(rend)
    raise TypeError(f"Object {type(obj)} does not support getOwnershipRange().")


@dataclass
class OwnerMap:
    """
    Owner map built from ownership ranges.

    ranges: 1D array of length size+1 with non-decreasing entries.
    """

    ranges: np.ndarray

    def __post_init__(self) -> None:
        self.ranges = np.asarray(self.ranges, dtype=int)
        if self.ranges.ndim != 1:
            self.ranges = self.ranges.ravel()
        if self.ranges.size < 2:
            raise ValueError("Ownership ranges must have length >= 2.")
        if not np.all(self.ranges[1:] >= self.ranges[:-1]):
            raise ValueError("Ownership ranges must be non-decreasing.")

    @property
    def size(self) -> int:
        return int(self.ranges.size - 1)

    def owner_of(self, j: int) -> int:
        """
        Return the owner rank for global index j.
        """
        j = int(j)
        if j < int(self.ranges[0]) or j >= int(self.ranges[-1]):
            raise ValueError(
                f"Global index {j} outside ownership ranges "
                f"[{int(self.ranges[0])}, {int(self.ranges[-1])})."
            )

        idx = int(np.searchsorted(self.ranges, j, side="right")) - 1
        if idx < 0 or idx >= self.size:
            raise RuntimeError(f"Failed to locate owner for index {j} with ranges={self.ranges}.")
        return idx


def build_owner_map_from_ownership_ranges(ranges: Sequence[int]) -> OwnerMap:
    """
    Build an OwnerMap from ownership ranges.
    """
    arr = np.asarray(ranges, dtype=int)
    return OwnerMap(arr)


def split_row_cols_by_owner(
    row_cols: Sequence[int],
    owner_map: OwnerMap,
    myrank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split one row's global column indices into diag/off columns by owner.
    """
    myrank = int(myrank)
    cols = np.asarray(row_cols, dtype=int).ravel()
    if cols.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    owners = np.fromiter(
        (owner_map.owner_of(int(j)) for j in cols),
        dtype=int,
        count=cols.size,
    )
    mask_diag = owners == myrank
    diag_cols = cols[mask_diag]
    off_cols = cols[~mask_diag]
    return diag_cols, off_cols


def count_diag_off_nnz_for_local_rows(
    local_rows_global: Sequence[int],
    local_row_cols: Sequence[Sequence[int]],
    owner_map: OwnerMap,
    myrank: int,
    ownership_range: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Count diag/off nnz per owned row using global column indices.

    Returns arrays of length (rend - rstart), aligned with local row indices.
    """
    myrank = int(myrank)
    rows = list(local_rows_global)
    if len(rows) != len(local_row_cols):
        raise ValueError(
            f"local_rows_global length {len(rows)} does not match "
            f"local_row_cols length {len(local_row_cols)}."
        )

    rstart, rend = ownership_range
    rstart = int(rstart)
    rend = int(rend)
    if rstart < 0 or rend < 0 or rstart > rend:
        raise ValueError(f"Invalid ownership_range={ownership_range}.")
    n_owned = int(rend - rstart)
    d_nz = np.zeros(n_owned, dtype=int)
    o_nz = np.zeros(n_owned, dtype=int)

    for i in range(len(rows)):
        gi = int(rows[i])
        if gi < rstart or gi >= rend:
            continue
        i_local = gi - rstart
        cols_i = local_row_cols[i]
        diag_cols, off_cols = split_row_cols_by_owner(cols_i, owner_map, myrank)
        d_nz[i_local] = int(diag_cols.size)
        o_nz[i_local] = int(off_cols.size)

    return d_nz, o_nz
