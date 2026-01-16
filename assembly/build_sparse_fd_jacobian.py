"""
Build sparse FD Jacobian in PETSc AIJ using a conservative pattern and coloring.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core import layout as layout_mod
from properties.gas import get_gas_eval_phase, set_gas_eval_phase
from solvers.nonlinear_context import NonlinearContext
from assembly.jacobian_pattern import JacobianPattern, build_jacobian_pattern
from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc
from parallel.mat_prealloc import (
    get_global_ownership_ranges_from_vec,
    build_owner_map_from_ownership_ranges,
    count_diag_off_nnz_for_local_rows,
)

logger = logging.getLogger(__name__)


def _get_petsc():
    try:
        bootstrap_mpi_before_petsc()
        from petsc4py import PETSc  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PETSc not available") from exc
    return PETSc


def _mat_setvalues_local(mat, rows, cols, values, *, addv: bool = False) -> None:
    """
    Thin wrapper around PETSc.Mat.setValues.

    Intended for test-time monkeypatching to validate local-row writes.
    """
    mat.setValues(rows, cols, values, addv=addv)


def _res_phys_to_eval_rows(
    res_phys_rows: np.ndarray,
    scale_F_safe: np.ndarray,
    row_ids_global: np.ndarray,
) -> np.ndarray:
    """
    Convert residual rows in physical space to evaluation space.
    """
    if scale_F_safe is None:
        return res_phys_rows
    if scale_F_safe.ndim != 1:
        scale_F_safe = scale_F_safe.ravel()
    scale_rows = scale_F_safe[row_ids_global]
    return res_phys_rows / scale_rows


def _build_sparse_fd_jacobian_serial(
    ctx: NonlinearContext,
    x0: np.ndarray,
    eps: float = 1.0e-8,
    drop_tol: float = 0.0,
    pattern: Optional[JacobianPattern] = None,
    mat=None,
    *,
    PETSc=None,
    comm=None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build sparse FD Jacobian using a coloring of the pattern.

    The x/residual spaces follow ctx.meta["petsc_x_space"] and ["petsc_f_space"].
    Defaults are physical/physical to align PETSc/FD semantics.
    """
    if PETSc is None:
        PETSc = _get_petsc()
    if comm is None:
        comm = PETSc.COMM_WORLD

    x0 = np.asarray(x0, dtype=np.float64)
    N = int(x0.size)

    scale_u = np.asarray(ctx.scale_u, dtype=np.float64)
    if scale_u.shape != x0.shape:
        raise ValueError(f"scale_u shape {scale_u.shape} does not match x0 {x0.shape}")
    scale_u_safe = np.where(scale_u > 0.0, scale_u, 1.0)

    meta = getattr(ctx, "meta", None)
    if meta is None or not isinstance(meta, dict):
        meta = {}

    scale_F = np.asarray(meta.get("residual_scale_F", np.ones_like(x0)), dtype=np.float64)
    if scale_F.shape != x0.shape:
        raise ValueError(f"residual_scale_F shape {scale_F.shape} does not match x0 {x0.shape}")
    scale_F_safe = np.where(scale_F > 0.0, scale_F, 1.0)
    x_space = str(meta.get("petsc_x_space", "physical")).lower()
    f_space = str(meta.get("petsc_f_space", "physical")).lower()
    petsc_cfg = getattr(ctx.cfg, "petsc", None)
    fd_jac_min_col_norm = float(getattr(petsc_cfg, "fd_jac_min_col_norm", 0.0)) if petsc_cfg else 0.0
    fd_jac_diag_damping = float(getattr(petsc_cfg, "fd_jac_diag_damping", 1.0)) if petsc_cfg else 1.0
    debug_fd_col = os.environ.get("DROPLET_FD_J_DEBUG_COL", "0") == "1"

    from assembly.residual_global import residual_only

    def _x_to_u_phys(x_arr: np.ndarray) -> np.ndarray:
        if x_space in ("physical", "phys", "unscaled"):
            return x_arr
        if x_space in ("scaled", "eval"):
            return x_arr * scale_u_safe
        raise ValueError(f"Unknown petsc_x_space={x_space!r}")

    def _res_phys_to_eval(res_phys: np.ndarray) -> np.ndarray:
        if f_space in ("physical", "phys", "unscaled"):
            return res_phys
        if f_space in ("scaled", "eval"):
            return res_phys / scale_F_safe
        raise ValueError(f"Unknown petsc_f_space={f_space!r}")

    def _residual_with_phase(u_phys: np.ndarray, phase: str) -> np.ndarray:
        prev_phase = get_gas_eval_phase()
        prev_tag = layout_mod.get_apply_u_tag()
        set_gas_eval_phase(phase)
        layout_mod.set_apply_u_tag(phase)
        try:
            return residual_only(u_phys, ctx)
        finally:
            layout_mod.set_apply_u_tag(prev_tag)
            set_gas_eval_phase(prev_phase)

    u0_phys = _x_to_u_phys(x0)
    r0_phys = _residual_with_phase(u0_phys, "MFPC_FD_base")
    if not np.all(np.isfinite(r0_phys)):
        r0_phys = np.where(np.isfinite(r0_phys), r0_phys, 1.0e20)
    r0_eval = _res_phys_to_eval(r0_phys)
    if r0_eval.shape != x0.shape:
        raise ValueError(f"residual shape {r0_eval.shape} does not match x0 {x0.shape}")

    if pattern is None:
        pattern = build_jacobian_pattern(ctx.cfg, ctx.grid_ref, ctx.layout)
    indptr = np.asarray(pattern.indptr, dtype=np.int32)
    indices = np.asarray(pattern.indices, dtype=np.int32)
    if indptr.size != N + 1:
        raise ValueError(f"pattern indptr size {indptr.size} does not match N+1 {N+1}")

    col_adj: List[set[int]] = [set() for _ in range(N)]
    col_rows: List[List[int]] = [[] for _ in range(N)]
    for i in range(N):
        row_cols = indices[indptr[i] : indptr[i + 1]]
        for c in row_cols:
            col_rows[int(c)].append(i)
        for a in range(row_cols.size):
            ca = int(row_cols[a])
            for b in range(a + 1, row_cols.size):
                cb = int(row_cols[b])
                if ca == cb:
                    continue
                col_adj[ca].add(cb)
                col_adj[cb].add(ca)

    order = sorted(range(N), key=lambda j: len(col_adj[j]), reverse=True)
    colors = [-1] * N
    ncolors = 0
    for j in order:
        used = {colors[nbr] for nbr in col_adj[j] if colors[nbr] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[j] = c
        if c + 1 > ncolors:
            ncolors = c + 1

    groups: List[List[int]] = [[] for _ in range(ncolors)]
    for j, c in enumerate(colors):
        groups[c].append(j)

    if mat is None:
        row_nnz = indptr[1:] - indptr[:-1]
        max_nnz_row = int(row_nnz.max()) if N > 0 else 0
        P = PETSc.Mat().createAIJ(size=(N, N), nnz=max_nnz_row, comm=comm)
        P.setUp()
    else:
        P = mat
        P.zeroEntries()

    n_fd_calls = 0
    x_work = x0.copy()
    col_norm_min = np.inf
    col_norm_max = 0.0
    col_count = 0
    suspicious_cols: List[Dict[str, Any]] = []

    for color_idx, cols_in_color in enumerate(groups):
        if not cols_in_color:
            continue

        dx_by_col: Dict[int, float] = {}
        for j in cols_in_color:
            dx = eps * (1.0 + abs(x0[j]))
            if dx == 0.0:
                dx = eps
            x_work[j] = x0[j] + dx
            dx_by_col[j] = dx

        u_phys = _x_to_u_phys(x_work)
        if len(cols_in_color) == 1:
            phase = f"MFPC_FD_col_{cols_in_color[0]}"
        else:
            phase = "MFPC_FD_cols_" + "_".join(str(c) for c in cols_in_color)
        r_phys = _residual_with_phase(u_phys, phase)
        if not np.all(np.isfinite(r_phys)):
            r_phys = np.where(np.isfinite(r_phys), r_phys, 1.0e20)
        r_eval = _res_phys_to_eval(r_phys)
        diff = r_eval - r0_eval
        n_fd_calls += 1

        for j in cols_in_color:
            rows = col_rows[j]
            if not rows:
                continue
            dx = dx_by_col[j]
            rows_arr = np.asarray(rows, dtype=np.int64)
            vals = diff[rows_arr] / dx
            if drop_tol > 0.0:
                mask = np.abs(vals) >= drop_tol
                rows_use = rows_arr[mask]
                vals_use = vals[mask]
            else:
                rows_use = rows_arr
                vals_use = vals

            nnz_col = int(vals_use.size)
            col_norm_inf = float(np.max(np.abs(vals_use))) if nnz_col else 0.0
            col_count += 1
            col_norm_min = min(col_norm_min, col_norm_inf)
            col_norm_max = max(col_norm_max, col_norm_inf)

            if fd_jac_min_col_norm > 0.0 and col_norm_inf < fd_jac_min_col_norm:
                suspicious_cols.append({"col": int(j), "norm_inf": col_norm_inf, "nnz": nnz_col})
                if debug_fd_col:
                    logger.warning(
                        "[FD_J_DEBUG][rank=%d] column i=%d looks nearly zero: ||col||_inf=%.3e nnz=%d",
                        int(comm.getRank()),
                        int(j),
                        col_norm_inf,
                        nnz_col,
                    )
                if fd_jac_diag_damping > 0.0:
                    P.setValue(int(j), int(j), float(fd_jac_diag_damping), addv=True)

            for row, val in zip(rows_use, vals_use):
                P.setValue(int(row), int(j), float(val), addv=False)

        for j in cols_in_color:
            x_work[j] = x0[j]

    P.assemblyBegin()
    P.assemblyEnd()

    ia, _ja, a_vals = P.getValuesCSR()
    nnz_total = int(a_vals.size)
    nnz_max = int((ia[1:] - ia[:-1]).max()) if N > 0 else 0
    nnz_avg = float(nnz_total) / float(N) if N > 0 else 0.0

    stats: Dict[str, Any] = {
        "ncolors": int(ncolors),
        "n_fd_calls": int(n_fd_calls),
        "nnz_total": nnz_total,
        "nnz_max_row": nnz_max,
        "nnz_avg": nnz_avg,
        "shape": (N, N),
        "eps": float(eps),
        "drop_tol": float(drop_tol),
        "pattern_nnz": int(indices.size),
    }
    stats["fd_jacobian_stats"] = {
        "n_cols": int(col_count),
        "col_norm_min": float(col_norm_min) if col_count else None,
        "col_norm_max": float(col_norm_max) if col_count else None,
        "n_suspicious_cols": int(len(suspicious_cols)),
        "suspicious_cols": list(suspicious_cols),
        "min_col_norm": float(fd_jac_min_col_norm),
        "diag_damping": float(fd_jac_diag_damping),
    }
    return P, stats


def _build_sparse_fd_jacobian_mpi(
    ctx: NonlinearContext,
    x0: np.ndarray,
    eps: float = 1.0e-8,
    drop_tol: float = 0.0,
    pattern: Optional[JacobianPattern] = None,
    mat=None,
    *,
    PETSc=None,
    comm=None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build sparse FD Jacobian in MPI mode (local rows only, no coloring/prealloc).
    """
    if PETSc is None:
        PETSc = _get_petsc()
    if comm is None:
        comm = PETSc.COMM_WORLD

    x0 = np.asarray(x0, dtype=np.float64)
    N = int(x0.size)
    if N == 0:
        raise ValueError("x0 must be non-empty for MPI Jacobian build.")

    scale_u = np.asarray(ctx.scale_u, dtype=np.float64)
    if scale_u.shape != x0.shape:
        raise ValueError(f"scale_u shape {scale_u.shape} does not match x0 {x0.shape}")
    scale_u_safe = np.where(scale_u > 0.0, scale_u, 1.0)

    meta = getattr(ctx, "meta", None)
    if meta is None or not isinstance(meta, dict):
        meta = {}
    scale_F = np.asarray(meta.get("residual_scale_F", np.ones_like(x0)), dtype=np.float64)
    if scale_F.shape != x0.shape:
        raise ValueError(f"residual_scale_F shape {scale_F.shape} does not match x0 {x0.shape}")
    scale_F_safe = np.where(scale_F > 0.0, scale_F, 1.0)
    x_space = str(meta.get("petsc_x_space", "physical")).lower()
    f_space = str(meta.get("petsc_f_space", "physical")).lower()

    from assembly.jacobian_pattern_dist import LocalJacPattern
    from assembly.residual_global import residual_only
    from assembly.residual_local import ResidualLocalCtx, pack_local_to_layout
    from parallel.dm_manager import global_to_local

    dm_mgr = meta.get("dm_manager") or meta.get("dm_mgr") or getattr(ctx, "dm_manager", None)
    ld = meta.get("layout_dist") or meta.get("ld") or getattr(ctx, "layout_dist", None)
    if dm_mgr is None or ld is None:
        raise RuntimeError(
            "[FD Jacobian] MPI mapping requires ctx.meta['dm_manager'] and ctx.meta['layout_dist']."
        )

    def _build_perm_l2d_d2l() -> Tuple[np.ndarray, np.ndarray]:
        Xg_gid = dm_mgr.dm.createGlobalVec()
        try:
            dm_rstart, dm_rend = Xg_gid.getOwnershipRange()
            dm_rstart = int(dm_rstart)
            dm_rend = int(dm_rend)
            try:
                x_view = Xg_gid.getArray()
                x_view[:] = np.arange(dm_rstart, dm_rend, dtype=np.float64) + 1.0
            finally:
                try:
                    Xg_gid.restoreArray()
                except Exception:
                    pass

            Xl_liq, Xl_gas, Xl_if = global_to_local(dm_mgr, Xg_gid)
            aXl_liq = dm_mgr.dm_liq.getVecArray(Xl_liq)
            aXl_gas = dm_mgr.dm_gas.getVecArray(Xl_gas)
            aXl_if = Xl_if.getArray()

            ctx_local = ResidualLocalCtx(layout=ctx.layout, ld=ld)
            u_layout_gid = pack_local_to_layout(
                ctx_local,
                aXl_liq,
                aXl_gas,
                aXl_if,
                rank=int(comm.getRank()),
            )
        finally:
            try:
                Xg_gid.destroy()
            except Exception:
                pass
            try:
                Xl_liq.destroy()
            except Exception:
                pass
            try:
                Xl_gas.destroy()
            except Exception:
                pass
            try:
                Xl_if.destroy()
            except Exception:
                pass

        mask = u_layout_gid != 0.0
        layout_idx_local = np.nonzero(mask)[0].astype(np.int64)
        dm_gid_local = (u_layout_gid[mask] - 1.0).astype(np.int64)

        try:
            from mpi4py import MPI  # type: ignore

            mpicomm = comm.tompi4py()
        except Exception as exc:
            raise RuntimeError("mpi4py is required to build MPI layout/DM permutation.") from exc

        all_layout = mpicomm.allgather(layout_idx_local)
        all_dm = mpicomm.allgather(dm_gid_local)

        perm_l2d = None
        perm_d2l = None
        if mpicomm.rank == 0:
            perm_l2d = np.full(N, -1, dtype=np.int64)
            for li, di in zip(all_layout, all_dm):
                if li.size:
                    perm_l2d[li] = di
            if np.any(perm_l2d < 0):
                raise RuntimeError("[FD Jacobian] failed to build full layout->DM permutation.")
            perm_d2l = np.empty_like(perm_l2d)
            perm_d2l[perm_l2d] = np.arange(N, dtype=np.int64)
            if not np.array_equal(perm_d2l[perm_l2d], np.arange(N, dtype=np.int64)):
                raise RuntimeError("[FD Jacobian] perm_d2l is not the inverse of perm_l2d.")
            if not np.array_equal(perm_l2d[perm_d2l], np.arange(N, dtype=np.int64)):
                raise RuntimeError("[FD Jacobian] perm_l2d is not the inverse of perm_d2l.")

        perm_l2d = mpicomm.bcast(perm_l2d, root=0)
        perm_d2l = mpicomm.bcast(perm_d2l, root=0)
        return perm_l2d, perm_d2l

    def _x_to_u_phys(x_arr: np.ndarray) -> np.ndarray:
        if x_space in ("physical", "phys", "unscaled"):
            return x_arr
        if x_space in ("scaled", "eval"):
            return x_arr * scale_u_safe
        raise ValueError(f"Unknown petsc_x_space={x_space!r}")

    def _res_phys_to_eval_local(res_phys_rows: np.ndarray, row_ids_global: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(res_phys_rows)):
            res_phys_rows = np.where(np.isfinite(res_phys_rows), res_phys_rows, 1.0e20)
        if f_space in ("physical", "phys", "unscaled"):
            return res_phys_rows
        if f_space in ("scaled", "eval"):
            return _res_phys_to_eval_rows(res_phys_rows, scale_F_safe, row_ids_global)
        raise ValueError(f"Unknown petsc_f_space={f_space!r}")

    def _get_ownership_info(mat_local=None) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        Resolve ownership range from LayoutDistributed or DM only.

        Do not fallback to a default Vec partition, which can mask partition/mapping bugs.
        """
        meta = getattr(ctx, "meta", None)
        if meta is None or not isinstance(meta, dict):
            meta = {}
        ld = meta.get("layout_dist") or meta.get("ld") or getattr(ctx, "layout_dist", None)
        dm_ctx = meta.get("dm") or getattr(ctx, "dm", None)

        rstart = None
        rend = None
        ranges = None

        if ld is not None and getattr(ld, "ownership_range", None) is not None:
            rstart, rend = (int(x) for x in ld.ownership_range)
            ranges = getattr(ld, "ownership_ranges", None)
            ld_size = getattr(ld, "global_size", getattr(ld, "N_total", N))
            if int(ld_size) != N:
                raise RuntimeError(
                    f"[FD Jacobian] global_size mismatch: ld={int(ld_size)} N={N}"
                )
            if ranges is None and dm_ctx is not None:
                vec = dm_ctx.createGlobalVec()
                try:
                    ranges = get_global_ownership_ranges_from_vec(vec)
                finally:
                    try:
                        vec.destroy()
                    except Exception:
                        pass
        elif dm_ctx is not None:
            vec = dm_ctx.createGlobalVec()
            try:
                rstart, rend = vec.getOwnershipRange()
                if int(vec.getSize()) != N:
                    raise RuntimeError(f"[FD Jacobian] DM Vec size mismatch: vec={vec.getSize()} N={N}")
                ranges = get_global_ownership_ranges_from_vec(vec)
            finally:
                try:
                    vec.destroy()
                except Exception:
                    pass
        else:
            raise RuntimeError(
                "[FD Jacobian] cannot determine ownership_range: ctx.meta lacks 'layout_dist' and 'dm'. "
                "Refusing default partition fallback."
            )

        if rstart is None or rend is None:
            raise RuntimeError("[FD Jacobian] failed to resolve ownership_range from DM/LD.")
        if ranges is None:
            raise RuntimeError("[FD Jacobian] failed to resolve ownership_ranges from DM/LD.")

        if mat_local is not None:
            mrstart, mrend = mat_local.getOwnershipRange()
            if (int(mrstart), int(mrend)) != (int(rstart), int(rend)):
                raise RuntimeError(
                    f"[FD Jacobian] Mat ownership_range {(int(mrstart), int(mrend))} "
                    f"!= DM/LD ownership_range {(int(rstart), int(rend))}. "
                    "Create matrices with the DM partition."
                )

        return (int(rstart), int(rend)), ranges

    ownership_range, ownership_ranges = _get_ownership_info(mat)
    rstart, rend = ownership_range
    n_owned = int(rend - rstart)

    perm_l2d, perm_d2l = _build_perm_l2d_d2l()
    rows_layout = perm_d2l[rstart:rend].astype(np.int64, copy=False)
    rows_global_dm = np.arange(rstart, rend, dtype=PETSc.IntType)
    if rows_layout.size != n_owned:
        raise RuntimeError(
            f"[FD Jacobian] layout row count {rows_layout.size} != owned rows {n_owned}."
        )
    if rows_layout.size and (rows_layout.min() < 0 or rows_layout.max() >= N):
        raise RuntimeError("[FD Jacobian] layout rows out of bounds for permutation.")

    def _residual_owned_with_phase(u_phys: np.ndarray, phase: str) -> np.ndarray:
        prev_phase = get_gas_eval_phase()
        prev_tag = layout_mod.get_apply_u_tag()
        set_gas_eval_phase(phase)
        layout_mod.set_apply_u_tag(phase)
        try:
            r_full = residual_only(u_phys, ctx)
            return r_full[rows_layout]
        finally:
            layout_mod.set_apply_u_tag(prev_tag)
            set_gas_eval_phase(prev_phase)

    pattern_global = build_jacobian_pattern(ctx.cfg, ctx.grid_ref, ctx.layout)
    indptr_global = np.asarray(pattern_global.indptr, dtype=np.int64)
    indices_global = np.asarray(pattern_global.indices, dtype=np.int64)
    nloc = int(rows_layout.size)

    if nloc == 0:
        indptr = np.zeros(1, dtype=np.int64)
        indices_layout = np.zeros(0, dtype=np.int64)
    else:
        indptr = np.zeros(nloc + 1, dtype=np.int64)
        indices_list: List[int] = []
        for k, row_l in enumerate(rows_layout):
            start = int(indptr_global[row_l])
            end = int(indptr_global[row_l + 1])
            cols = indices_global[start:end]
            cols_list = [int(c) for c in cols]
            if int(row_l) not in cols_list:
                cols_list.append(int(row_l))
            cols_list = sorted(set(cols_list))
            indices_list.extend(cols_list)
            indptr[k + 1] = len(indices_list)
        indices_layout = np.asarray(indices_list, dtype=np.int64)

    indices_dm = perm_l2d[indices_layout] if indices_layout.size else np.zeros(0, dtype=np.int64)
    nnz_total_local = int(indices_layout.size)
    nnz_max_row = int((indptr[1:] - indptr[:-1]).max()) if nloc > 0 else 0
    nnz_avg = float(nnz_total_local) / float(nloc) if nloc > 0 else 0.0
    pattern_local = LocalJacPattern(
        rows_global=rows_global_dm.astype(PETSc.IntType, copy=False),
        indptr=indptr.astype(np.int64, copy=False),
        indices=indices_dm.astype(PETSc.IntType, copy=False),
        shape=(N, N),
        meta={
            "nnz_total": float(nnz_total_local),
            "nnz_avg": float(nnz_avg),
            "nnz_max_row": float(nnz_max_row),
            "n_local_rows": float(nloc),
        },
    )

    if nloc == 0 or indices_layout.size == 0:
        if mat is None:
            P = PETSc.Mat().createAIJ(size=((n_owned, N), (n_owned, N)), comm=comm)
            try:
                P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
            except Exception:
                pass
            P.setUp()
        else:
            P = mat
            try:
                P.zeroEntries()
            except Exception:
                pass
            try:
                P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
            except Exception:
                pass
        stats: Dict[str, Any] = {
            "ncolors": 0,
            "n_fd_calls": 0,
            "nnz_total_local": 0,
            "n_local_rows": int(nloc),
            "shape": (N, N),
            "eps": float(eps),
            "drop_tol": float(drop_tol),
            "pattern_nnz_local": int(indices_layout.size),
            "prealloc_nnz_local": 0,
            "cols_to_perturb": [],
            "ownership_range": ownership_range,
            "mpi_size": int(comm.getSize()),
            "max_diag_abs_local": 0.0,
            "pattern_local": pattern_local,
        }
        return P, stats

    owner_map = build_owner_map_from_ownership_ranges(ownership_ranges)
    myrank = int(comm.getRank())
    debug_fd = os.environ.get("DROPLET_PETSC_DEBUG", "0") == "1"
    debug_fd_once = os.environ.get("DROPLET_PETSC_DEBUG_ONCE", "1") == "1"
    debug_fd_logged = False
    debug_fd_col = os.environ.get("DROPLET_FD_J_DEBUG_COL", "0") == "1"
    petsc_cfg = getattr(ctx.cfg, "petsc", None)
    fd_jac_min_col_norm = float(getattr(petsc_cfg, "fd_jac_min_col_norm", 0.0)) if petsc_cfg else 0.0
    fd_jac_diag_damping = float(getattr(petsc_cfg, "fd_jac_diag_damping", 1.0)) if petsc_cfg else 1.0

    local_row_cols_layout: List[List[int]] = []
    local_row_cols_dm: List[List[int]] = []
    col_to_rows: Dict[int, List[int]] = {}
    for k in range(nloc):
        start = int(indptr[k])
        end = int(indptr[k + 1])
        cols = indices_layout[start:end]
        row_l = int(rows_layout[k])
        cols_list = [int(c) for c in cols]
        if row_l not in cols_list:
            cols_list.append(row_l)
        cols_list = sorted(set(cols_list))
        local_row_cols_layout.append(cols_list)
        for c in cols_list:
            col_to_rows.setdefault(c, []).append(k)
        cols_dm = perm_l2d[np.asarray(cols_list, dtype=np.int64)]
        local_row_cols_dm.append([int(c) for c in cols_dm])

    d_nz, o_nz = count_diag_off_nnz_for_local_rows(
        local_rows_global=rows_global_dm,
        local_row_cols=local_row_cols_dm,
        owner_map=owner_map,
        myrank=myrank,
        ownership_range=ownership_range,
    )
    d_nz = np.asarray(d_nz, dtype=PETSc.IntType)
    o_nz = np.asarray(o_nz, dtype=PETSc.IntType)
    prealloc_nnz_local = int(d_nz.sum() + o_nz.sum())

    if mat is None:
        P = PETSc.Mat().createAIJ(size=((n_owned, N), (n_owned, N)), nnz=(d_nz, o_nz), comm=comm)
        P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        P.setUp()
    else:
        P = mat
        try:
            P.zeroEntries()
        except Exception:
            pass
        try:
            P.setPreallocationNNZ((d_nz, o_nz))
        except Exception:
            pass
        try:
            P.setUp()
        except Exception:
            pass
        try:
            P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        except Exception:
            pass

    cols_to_perturb = np.asarray(sorted(col_to_rows.keys()), dtype=PETSc.IntType)
    rows_global_list = rows_global_dm
    u0_phys = _x_to_u_phys(x0)
    r0_phys = _residual_owned_with_phase(u0_phys, "MFPC_FD_base")
    if debug_fd:
        try:
            meta = getattr(ctx, "meta", None)
            if meta is None or not isinstance(meta, dict):
                meta = {}
                try:
                    ctx.meta = meta
                except Exception:
                    pass
            key = "_dbg_mfpc_fd_base"
            if (not debug_fd_once) or (not meta.get(key)):
                meta[key] = True
                r0_inf = float(np.linalg.norm(r0_phys, ord=np.inf)) if r0_phys.size else 0.0
                r0_2 = float(np.linalg.norm(r0_phys)) if r0_phys.size else 0.0
                logger.warning(
                    "[DBG][rank=%d] MFPC_FD_base residual: ||F||_inf=%.3e ||F||_2=%.3e nloc=%d",
                    myrank,
                    r0_inf,
                    r0_2,
                    int(r0_phys.size),
                )
        except Exception:
            logger.debug("MFPC_FD_base debug logging failed.", exc_info=True)
    if not np.all(np.isfinite(r0_phys)):
        r0_phys = np.where(np.isfinite(r0_phys), r0_phys, 1.0e20)
    r0_eval = _res_phys_to_eval_local(r0_phys, rows_layout)
    if r0_eval.shape != (nloc,):
        raise ValueError(f"local residual shape {r0_eval.shape} does not match nloc {nloc}")

    x_work = x0.copy()
    def _assert_local_rows(rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        rmin = int(rows.min())
        rmax = int(rows.max())
        if rmin < rstart or rmax >= rend:
            raise RuntimeError(
                f"[FD Jacobian] rank {myrank} writing non-local rows "
                f"[{rmin}, {rmax}] with ownership [{rstart}, {rend})"
            )
    n_fd_calls = 0
    # Enhanced debug: track which column caused Y anomaly
    debug_fd_track_Y_anomaly = os.environ.get("DROPLET_FD_TRACK_Y_ANOMALY", "0") == "1"
    debug_fd_max_logs = int(os.environ.get("DROPLET_FD_MAX_LOGS", "50"))
    debug_fd_log_count = 0
    col_norm_min = np.inf
    col_norm_max = 0.0
    col_count = 0
    suspicious_cols: List[Dict[str, Any]] = []

    def _get_var_info(col_idx: int) -> str:
        """Get variable type and cell/species info for a column index."""
        layout = ctx.layout
        entries = getattr(layout, "entries", [])
        if col_idx < len(entries):
            e = entries[col_idx]
            return f"{e.kind}[cell={e.cell},spec={e.spec}]"
        return f"unknown[{col_idx}]"

    for j in cols_to_perturb:
        j = int(j)
        iloc_list = col_to_rows.get(j)
        if not iloc_list:
            continue
        col_dm = int(perm_l2d[j])
        dx = eps * (1.0 + abs(x0[j]))
        if dx == 0.0:
            dx = eps

        x_work[j] = x0[j] + dx

        # Enhanced debug logging
        should_log = debug_fd and (not debug_fd_once or not debug_fd_logged)
        if should_log or debug_fd_track_Y_anomaly:
            try:
                u_dbg = _x_to_u_phys(x_work)
                state_dbg = ctx.make_state_from_u(u_dbg)
                Yg = np.asarray(state_dbg.Yg, dtype=np.float64)
                cell_idx = 7
                Y_sum = 0.0
                Y_anomaly = False
                if Yg.ndim == 2 and Yg.shape[1] > cell_idx:
                    Y = Yg[:, cell_idx]
                    Y_sum = float(np.sum(Y)) if Y.size else 0.0
                    Y_anomaly = abs(Y_sum - 1.0) > 0.01  # Y sum deviates from 1.0

                if should_log or (debug_fd_track_Y_anomaly and Y_anomaly and debug_fd_log_count < debug_fd_max_logs):
                    var_info = _get_var_info(j)
                    logger.warning(
                        "[DBG][rank=%d] MFPC_FD col=%d (%s): x0=%.6e dx=%.3e x_pert=%.6e",
                        myrank, j, var_info, float(x0[j]), dx, float(x_work[j]),
                    )
                    if Yg.ndim == 2 and Yg.shape[1] > cell_idx:
                        Y = Yg[:, cell_idx]
                        mn = float(np.min(Y)) if Y.size else 0.0
                        mx = float(np.max(Y)) if Y.size else 0.0
                        idx = np.argsort(-Y)[:3] if Y.size else np.array([], dtype=np.int64)
                        top = [(int(i), float(Y[i])) for i in idx]
                        logger.warning(
                            "[DBG][rank=%d] MFPC_FD_EVAL Y@cell%d sum=%.6g min=%.3e max=%.3e top=%s",
                            myrank, cell_idx, Y_sum, mn, mx, top,
                        )
                        if Y_anomaly:
                            # Print more detailed info for anomaly
                            logger.warning(
                                "[DBG][rank=%d] Y_ANOMALY at col=%d! Printing full Y: %s",
                                myrank, j, Y.tolist(),
                            )
                    debug_fd_log_count += 1
            except Exception as exc:
                if should_log:
                    logger.warning("[DBG][rank=%d] MFPC_FD eval debug failed: %r", myrank, exc)
            if should_log:
                debug_fd_logged = True
        u1_phys = _x_to_u_phys(x_work)
        phase = f"MFPC_FD_col_{j}"
        r1_phys = _residual_owned_with_phase(u1_phys, phase)
        if not np.all(np.isfinite(r1_phys)):
            r1_phys = np.where(np.isfinite(r1_phys), r1_phys, 1.0e20)
        r1_eval = _res_phys_to_eval_local(r1_phys, rows_layout)

        dF_local = (r1_eval - r0_eval) / dx
        rows_sel = rows_global_list[iloc_list]
        vals_sel = dF_local[iloc_list]

        if drop_tol > 0.0:
            mask = np.abs(vals_sel) >= drop_tol
            rows_use = rows_sel[mask]
            vals_use = vals_sel[mask]
        else:
            rows_use = rows_sel
            vals_use = vals_sel

        nnz_col = int(vals_use.size)
        col_norm_inf = float(np.max(np.abs(vals_use))) if nnz_col else 0.0
        col_count += 1
        col_norm_min = min(col_norm_min, col_norm_inf)
        col_norm_max = max(col_norm_max, col_norm_inf)

        if fd_jac_min_col_norm > 0.0 and col_norm_inf < fd_jac_min_col_norm:
            suspicious_cols.append({"col": int(j), "norm_inf": col_norm_inf, "nnz": nnz_col})
            if debug_fd_col:
                logger.warning(
                    "[FD_J_DEBUG][rank=%d] column i=%d looks nearly zero: ||col||_inf=%.3e nnz=%d",
                    myrank,
                    int(j),
                    col_norm_inf,
                    nnz_col,
                )
            if fd_jac_diag_damping > 0.0 and (rstart <= col_dm < rend):
                P.setValue(col_dm, col_dm, float(fd_jac_diag_damping), addv=True)

        if rows_use.size:
            _assert_local_rows(rows_use)
            _mat_setvalues_local(
                P,
                rows_use,
                np.asarray([col_dm], dtype=PETSc.IntType),
                vals_use.reshape(-1, 1),
                addv=False,
            )

        x_work[j] = x0[j]
        n_fd_calls += 1

    P.assemblyBegin()
    P.assemblyEnd()

    nnz_local = 0
    max_diag_abs_local = 0.0
    for gi in rows_global_list:
        cols, vals = P.getRow(int(gi))
        nnz_local += int(len(cols))
        mask = cols == gi
        if np.any(mask):
            diag_val = float(vals[mask][0])
            max_diag_abs_local = max(max_diag_abs_local, abs(diag_val))
        if hasattr(P, "restoreRow"):
            P.restoreRow(int(gi), cols, vals)

    stats = {
        "ncolors": 0,
        "n_fd_calls": int(n_fd_calls),
        "nnz_total_local": int(nnz_local),
        "n_local_rows": int(nloc),
        "shape": (N, N),
        "eps": float(eps),
        "drop_tol": float(drop_tol),
        "pattern_nnz_local": int(indices_layout.size),
        "prealloc_nnz_local": int(prealloc_nnz_local),
        "cols_to_perturb": [int(c) for c in cols_to_perturb],
        "ownership_range": ownership_range,
        "mpi_size": int(comm.getSize()),
        "max_diag_abs_local": float(max_diag_abs_local),
        "pattern_local": pattern_local,
    }
    stats["fd_jacobian_stats"] = {
        "n_cols": int(col_count),
        "col_norm_min": float(col_norm_min) if col_count else None,
        "col_norm_max": float(col_norm_max) if col_count else None,
        "n_suspicious_cols": int(len(suspicious_cols)),
        "suspicious_cols": list(suspicious_cols),
        "min_col_norm": float(fd_jac_min_col_norm),
        "diag_damping": float(fd_jac_diag_damping),
    }
    return P, stats


def build_sparse_fd_jacobian(
    ctx: NonlinearContext,
    x0: np.ndarray,
    eps: float = 1.0e-8,
    drop_tol: float = 0.0,
    pattern: Optional[JacobianPattern] = None,
    mat=None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build sparse FD Jacobian for the current PETSc communicator.

    Defaults to physical space for both serial and MPI; can be overridden via
    ctx.meta["petsc_x_space"] and ["petsc_f_space"].

    If mat is provided, its ownership range defines local rows.
    """
    PETSc = _get_petsc()
    comm = PETSc.COMM_WORLD
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D array.")

    if comm.getSize() == 1:
        return _build_sparse_fd_jacobian_serial(
            ctx,
            x0,
            eps=eps,
            drop_tol=drop_tol,
            pattern=pattern,
            mat=mat,
            PETSc=PETSc,
            comm=comm,
        )

    return _build_sparse_fd_jacobian_mpi(
        ctx,
        x0,
        eps=eps,
        drop_tol=drop_tol,
        pattern=pattern,
        mat=mat,
        PETSc=PETSc,
        comm=comm,
    )
