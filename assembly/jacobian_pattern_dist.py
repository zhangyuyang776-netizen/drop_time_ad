# -*- coding: utf-8 -*-
"""
Distributed (per-rank) Jacobian sparsity pattern construction.

Stage 5.4B Task 3:
- Build a CSR sparsity pattern only for rows owned by this rank;
- Columns remain in global index space, suitable for MPIAIJ preallocation;
- No PETSc dependency (ownership_range provided by caller).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from core.types import CaseConfig, Grid1D
from core.layout import UnknownLayout


@dataclass(slots=True)
class LocalJacPattern:
    """
    Local (per-rank) Jacobian sparsity pattern in CSR form.

    Contract:
    - rows_global contains global row indices for this rank's pattern rows.
    - Each rows_global entry must satisfy rstart <= row < rend (DM ownership).
    - Order need not be contiguous or sorted.
    - Any DM-owned row not present here is treated as a zero row for FD preconditioning.
    - This pattern must not be used to define ownership sizes or Mat/Vec ranges.
    """

    rows_global: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    shape: Tuple[int, int]
    meta: Dict[str, float]


def build_jacobian_pattern_local(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    ownership_range: Tuple[int, int],
) -> LocalJacPattern:
    """
    Build a conservative CSR sparsity pattern for rows owned by this rank.

    The result is intended for FD preconditioning only. It does not define
    ownership ranges or local row counts; use the DM ownership range for that.
    """
    blocks = getattr(layout, "blocks", None)
    if not blocks:
        raise ValueError("layout.blocks is required for LocalJacPattern.")

    try:
        N = max(sl.stop for sl in blocks.values())
    except ValueError as exc:
        raise ValueError("layout.blocks must contain at least one non-empty slice.") from exc

    rstart, rend = ownership_range
    rstart = int(rstart)
    rend = int(rend)
    if rstart < 0 or rend < 0 or rstart > rend:
        raise ValueError(f"Invalid ownership_range={ownership_range}.")
    if rend > N:
        raise ValueError(f"ownership_range end {rend} exceeds global size N={N}.")

    nloc = rend - rstart
    rows_global = np.arange(rstart, rend, dtype=np.int32)

    if nloc == 0:
        indptr = np.zeros(1, dtype=np.int32)
        indices = np.zeros(0, dtype=np.int32)
        meta: Dict[str, float] = {
            "nnz_total": 0.0,
            "nnz_avg": 0.0,
            "nnz_max_row": 0.0,
            "n_local_rows": 0.0,
        }
        return LocalJacPattern(
            rows_global=rows_global,
            indptr=indptr,
            indices=indices,
            shape=(N, N),
            meta=meta,
        )

    row_sets = [set() for _ in range(nloc)]

    def add_coupling(i: int, j: int) -> None:
        if 0 <= i < N and 0 <= j < N and rstart <= i < rend:
            row_sets[i - rstart].add(int(j))

    for gi in range(rstart, rend):
        add_coupling(gi, gi)

    sl_Tg = blocks.get("Tg")
    sl_Tl = blocks.get("Tl")
    sl_Yg = blocks.get("Yg")
    sl_Yl = blocks.get("Yl")
    sl_Ts = blocks.get("Ts")
    sl_mpp = blocks.get("mpp")
    sl_Rd = blocks.get("Rd")

    physics = getattr(cfg, "physics", cfg)
    Ng = int(getattr(layout, "Ng", 0))
    Nl = int(getattr(layout, "Nl", 0))
    Ns_g_eff = int(getattr(layout, "Ns_g_eff", 0))
    Ns_l_eff = int(getattr(layout, "Ns_l_eff", 0))

    if sl_Tg is not None and Ng > 0:
        for ig in range(Ng):
            gi = sl_Tg.start + ig
            if ig > 0:
                add_coupling(gi, sl_Tg.start + ig - 1)
            if ig < Ng - 1:
                add_coupling(gi, sl_Tg.start + ig + 1)

    if sl_Tl is not None and Nl > 0:
        for il in range(Nl):
            gi = sl_Tl.start + il
            if il > 0:
                add_coupling(gi, sl_Tl.start + il - 1)
            if il < Nl - 1:
                add_coupling(gi, sl_Tl.start + il + 1)

    if sl_Yg is not None and getattr(physics, "solve_Yg", False) and Ng > 0 and Ns_g_eff > 0:
        n_spec = Ns_g_eff
        for ig in range(Ng):
            cell_start = sl_Yg.start + ig * n_spec
            cell_end = cell_start + n_spec
            for j in range(cell_start, cell_end):
                for k in range(cell_start, cell_end):
                    add_coupling(j, k)
                if ig > 0:
                    add_coupling(j, j - n_spec)
                if ig < Ng - 1:
                    add_coupling(j, j + n_spec)

    if sl_Yl is not None and getattr(physics, "solve_Yl", False) and Nl > 0 and Ns_l_eff > 0:
        n_spec = Ns_l_eff
        for il in range(Nl):
            cell_start = sl_Yl.start + il * n_spec
            cell_end = cell_start + n_spec
            for j in range(cell_start, cell_end):
                for k in range(cell_start, cell_end):
                    add_coupling(j, k)
                if il > 0:
                    add_coupling(j, j - n_spec)
                if il < Nl - 1:
                    add_coupling(j, j + n_spec)

    if sl_Tg is not None and sl_Yg is not None and Ng > 0 and Ns_g_eff > 0:
        n_spec = Ns_g_eff
        for ig in range(Ng):
            tg_idx = sl_Tg.start + ig
            cell_start = sl_Yg.start + ig * n_spec
            cell_end = cell_start + n_spec
            for j in range(cell_start, cell_end):
                add_coupling(tg_idx, j)
                add_coupling(j, tg_idx)

    if sl_Tl is not None and sl_Yl is not None and Nl > 0 and Ns_l_eff > 0:
        n_spec = Ns_l_eff
        for il in range(Nl):
            tl_idx = sl_Tl.start + il
            cell_start = sl_Yl.start + il * n_spec
            cell_end = cell_start + n_spec
            for j in range(cell_start, cell_end):
                add_coupling(tl_idx, j)
                add_coupling(j, tl_idx)

    interface_ids = []
    for sl in (sl_Ts, sl_mpp, sl_Rd):
        if sl is not None:
            interface_ids.extend(range(sl.start, sl.stop))

    for i in interface_ids:
        for j in interface_ids:
            add_coupling(i, j)

    if interface_ids:
        if sl_Tg is not None and Ng > 0:
            idx_Tg0 = sl_Tg.start
            for i in interface_ids:
                add_coupling(i, idx_Tg0)
                add_coupling(idx_Tg0, i)
        if sl_Tl is not None and Nl > 0:
            idx_Tl_last = sl_Tl.start + (Nl - 1)
            for i in interface_ids:
                add_coupling(i, idx_Tl_last)
                add_coupling(idx_Tl_last, i)
        if sl_Yg is not None and getattr(physics, "solve_Yg", False) and Ng > 0 and Ns_g_eff > 0:
            n_spec = Ns_g_eff
            cell_start = sl_Yg.start
            cell_end = cell_start + n_spec
            for i in interface_ids:
                for j in range(cell_start, cell_end):
                    add_coupling(i, j)
                    add_coupling(j, i)
        if sl_Yl is not None and getattr(physics, "solve_Yl", False) and Nl > 0 and Ns_l_eff > 0:
            n_spec = Ns_l_eff
            cell_start = sl_Yl.start + (Nl - 1) * n_spec
            cell_end = cell_start + n_spec
            for i in interface_ids:
                for j in range(cell_start, cell_end):
                    add_coupling(i, j)
                    add_coupling(j, i)

    indptr = np.zeros(nloc + 1, dtype=np.int32)
    indices_list = []
    nnz = 0
    max_row = 0
    for iloc in range(nloc):
        cols = sorted(row_sets[iloc])
        nnz += len(cols)
        if len(cols) > max_row:
            max_row = len(cols)
        indptr[iloc + 1] = nnz
        indices_list.extend(cols)

    indices = np.asarray(indices_list, dtype=np.int32)
    meta = {
        "nnz_total": float(nnz),
        "nnz_avg": float(nnz) / float(nloc) if nloc > 0 else 0.0,
        "nnz_max_row": float(max_row),
        "n_local_rows": float(nloc),
    }
    return LocalJacPattern(
        rows_global=rows_global,
        indptr=indptr,
        indices=indices,
        shape=(N, N),
        meta=meta,
    )
