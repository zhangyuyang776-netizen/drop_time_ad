"""
Conservative sparsity pattern for the global Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from core.types import CaseConfig, Grid1D
from core.layout import UnknownLayout


@dataclass(slots=True)
class JacobianPattern:
    """CSR pattern for global Jacobian / preconditioner."""

    indptr: np.ndarray
    indices: np.ndarray
    shape: Tuple[int, int]
    meta: Dict[str, float]


def build_jacobian_pattern(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
) -> JacobianPattern:
    """
    Build a conservative CSR sparsity pattern for the global Jacobian dF/du.
    """
    blocks = getattr(layout, "blocks", None)
    if not blocks:
        raise ValueError("layout.blocks is required for Jacobian pattern.")

    N = max(sl.stop for sl in blocks.values())
    row_sets = [set() for _ in range(N)]

    def add_coupling(i: int, j: int) -> None:
        if 0 <= i < N and 0 <= j < N:
            row_sets[i].add(j)

    for i in range(N):
        row_sets[i].add(i)

    sl_Tg = blocks.get("Tg")
    sl_Tl = blocks.get("Tl")
    sl_Yg = blocks.get("Yg")
    sl_Yl = blocks.get("Yl")
    sl_Ts = blocks.get("Ts")
    sl_mpp = blocks.get("mpp")
    sl_Rd = blocks.get("Rd")

    if sl_Tg is not None and layout.Ng > 0:
        for ig in range(layout.Ng):
            gi = sl_Tg.start + ig
            if ig > 0:
                add_coupling(gi, sl_Tg.start + ig - 1)
            if ig < layout.Ng - 1:
                add_coupling(gi, sl_Tg.start + ig + 1)

    if sl_Tl is not None and layout.Nl > 0:
        for il in range(layout.Nl):
            gi = sl_Tl.start + il
            if il > 0:
                add_coupling(gi, sl_Tl.start + il - 1)
            if il < layout.Nl - 1:
                add_coupling(gi, sl_Tl.start + il + 1)

    if sl_Yg is not None and cfg.physics.solve_Yg and layout.Ng > 0 and layout.Ns_g_eff > 0:
        n_spec = layout.Ns_g_eff
        for ig in range(layout.Ng):
            cell_start = sl_Yg.start + ig * n_spec
            cell_end = cell_start + n_spec
            for j in range(cell_start, cell_end):
                for k in range(cell_start, cell_end):
                    add_coupling(j, k)
                if ig > 0:
                    add_coupling(j, j - n_spec)
                if ig < layout.Ng - 1:
                    add_coupling(j, j + n_spec)

    if sl_Yl is not None and cfg.physics.solve_Yl and layout.Nl > 0 and layout.Ns_l_eff > 0:
        n_spec = layout.Ns_l_eff
        for il in range(layout.Nl):
            cell_start = sl_Yl.start + il * n_spec
            cell_end = cell_start + n_spec
            for j in range(cell_start, cell_end):
                for k in range(cell_start, cell_end):
                    add_coupling(j, k)
                if il > 0:
                    add_coupling(j, j - n_spec)
                if il < layout.Nl - 1:
                    add_coupling(j, j + n_spec)

    if sl_Tg is not None and sl_Yg is not None and layout.Ng > 0 and layout.Ns_g_eff > 0:
        n_spec = layout.Ns_g_eff
        for ig in range(layout.Ng):
            tg_idx = sl_Tg.start + ig
            cell_start = sl_Yg.start + ig * n_spec
            cell_end = cell_start + n_spec
            for j in range(cell_start, cell_end):
                add_coupling(tg_idx, j)
                add_coupling(j, tg_idx)

    if sl_Tl is not None and sl_Yl is not None and layout.Nl > 0 and layout.Ns_l_eff > 0:
        n_spec = layout.Ns_l_eff
        for il in range(layout.Nl):
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
        if sl_Tg is not None and layout.Ng > 0:
            idx_Tg0 = sl_Tg.start
            for i in interface_ids:
                add_coupling(i, idx_Tg0)
                add_coupling(idx_Tg0, i)
        if sl_Tl is not None and layout.Nl > 0:
            idx_Tl_last = sl_Tl.start + (layout.Nl - 1)
            for i in interface_ids:
                add_coupling(i, idx_Tl_last)
                add_coupling(idx_Tl_last, i)
        if sl_Yg is not None and cfg.physics.solve_Yg and layout.Ng > 0 and layout.Ns_g_eff > 0:
            n_spec = layout.Ns_g_eff
            cell_start = sl_Yg.start
            cell_end = cell_start + n_spec
            for i in interface_ids:
                for j in range(cell_start, cell_end):
                    add_coupling(i, j)
                    add_coupling(j, i)
        if sl_Yl is not None and cfg.physics.solve_Yl and layout.Nl > 0 and layout.Ns_l_eff > 0:
            n_spec = layout.Ns_l_eff
            cell_start = sl_Yl.start + (layout.Nl - 1) * n_spec
            cell_end = cell_start + n_spec
            for i in interface_ids:
                for j in range(cell_start, cell_end):
                    add_coupling(i, j)
                    add_coupling(j, i)

    indptr = np.zeros(N + 1, dtype=np.int32)
    indices_list = []
    nnz = 0
    max_row = 0
    for i in range(N):
        cols = sorted(row_sets[i])
        nnz += len(cols)
        if len(cols) > max_row:
            max_row = len(cols)
        indptr[i + 1] = nnz
        indices_list.extend(cols)

    indices = np.asarray(indices_list, dtype=np.int32)
    meta = {
        "nnz_total": float(nnz),
        "nnz_avg": float(nnz) / float(N) if N > 0 else 0.0,
        "nnz_max_row": float(max_row),
    }
    return JacobianPattern(indptr=indptr, indices=indices, shape=(N, N), meta=meta)
