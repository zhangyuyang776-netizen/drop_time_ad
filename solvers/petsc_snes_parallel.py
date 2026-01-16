# -*- coding: utf-8 -*-
"""
PETSc SNES nonlinear solver wrapper (parallel-only).

Scope:
- Always uses PETSc.COMM_WORLD as the communicator.
- Auto-builds DM + dm_manager + layout_dist if missing from ctx.meta.
- Matrix-free Jacobian (SNES MFFD); supports mf and mfpc_sparse_fd modes.
- Supports fieldsplit additive and schur PC variants.
- This module does NOT provide serial fallback; use solvers.petsc_snes for serial or hybrid modes.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Mapping

import numpy as np

from core import layout as layout_mod
from core.logging_utils import is_root_rank
from assembly.residual_global import residual_only, residual_petsc
from properties.gas import set_gas_eval_phase
from solvers.linear_types import JacobianMode, LinearSolverConfig as LinearSolverConfigTyped
from solvers.mpi_entry_validate import validate_mpi_before_petsc
from solvers.nonlinear_context import NonlinearContext
from solvers.nonlinear_types import NonlinearDiagnostics, NonlinearSolveResult
from solvers.petsc_linear import apply_fieldsplit_subksp_defaults, apply_structured_pc, _normalize_pc_type
from solvers.petsc_snes import _enable_snes_matrix_free, _finalize_ksp_config, _get_petsc

logger = logging.getLogger(__name__)

DEBUG_DOF_GID = os.getenv("DROPLET_DEBUG_DOF_GID")
DEBUG_DOF_GID = int(DEBUG_DOF_GID) if DEBUG_DOF_GID is not None else None


def _debug_enabled() -> bool:
    return os.environ.get("DROPLET_PETSC_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_options_prefix(prefix: str | None) -> str:
    if not prefix:
        return ""
    p = str(prefix)
    if not p.endswith("_"):
        p += "_"
    return p


def _normalize_cfg_for_snes_smoke(cfg):
    """
    P3.5-5-1: Normalize config for SNES smoke mode.

    Smoke mode adjustments:
    1. Limit SNES outer iterations to 1 (or min(max_outer_iter, 1))
    2. Force fieldsplit to use mfpc_sparse_fd (never pure mf, which gives identity P)

    Args:
        cfg: Case configuration object

    Returns:
        dict: Diagnostic info about normalization applied
            {
                "smoke_enabled": bool,
                "max_outer_iter_original": int,
                "max_outer_iter_effective": int,
                "jacobian_mode_original": str,
                "jacobian_mode_effective": str,
            }
    """
    diag_info = {
        "smoke_enabled": False,
        "max_outer_iter_original": None,
        "max_outer_iter_effective": None,
        "jacobian_mode_original": None,
        "jacobian_mode_effective": None,
    }

    # Check smoke mode
    nl_cfg = getattr(cfg, "nonlinear", None)

    debug_enabled = _debug_enabled()
    is_root = is_root_rank()
    if debug_enabled and is_root:
        logger.debug("[P3.5-5-1 DEBUG] _normalize_cfg_for_snes_smoke called")
        logger.debug("  nl_cfg type: %s", type(nl_cfg))
        logger.debug("  nl_cfg: %s", nl_cfg)

    if nl_cfg is None:
        if debug_enabled and is_root:
            logger.debug("  nl_cfg is None, returning early")
        return diag_info

    smoke = bool(getattr(nl_cfg, "smoke", False))
    if debug_enabled and is_root:
        logger.debug("  smoke field value: %s", smoke)
    diag_info["smoke_enabled"] = smoke

    # Record original max_outer_iter
    max_outer_iter_orig = int(getattr(nl_cfg, "max_outer_iter", 20))
    if debug_enabled and is_root:
        logger.debug("  max_outer_iter_original: %d", max_outer_iter_orig)
    diag_info["max_outer_iter_original"] = max_outer_iter_orig

    # If smoke mode, limit to 1 iteration
    if smoke:
        max_outer_iter_eff = min(max_outer_iter_orig, 1)
        if debug_enabled and is_root:
            logger.debug("  smoke=True, setting nl_cfg.max_outer_iter to %d", max_outer_iter_eff)
        nl_cfg.max_outer_iter = max_outer_iter_eff
        if debug_enabled and is_root:
            logger.debug(
                "  After mutation: nl_cfg.max_outer_iter = %s",
                getattr(nl_cfg, "max_outer_iter", "MISSING"),
            )
        diag_info["max_outer_iter_effective"] = max_outer_iter_eff
    else:
        if debug_enabled and is_root:
            logger.debug("  smoke=False, no iteration limiting")
        diag_info["max_outer_iter_effective"] = max_outer_iter_orig

    # Record original jacobian_mode
    petsc_cfg = getattr(cfg, "petsc", None)
    if petsc_cfg is not None:
        jac_mode_orig = getattr(petsc_cfg, "jacobian_mode", "mf")
        if debug_enabled and is_root:
            logger.debug("  jacobian_mode_original: %s", jac_mode_orig)
        diag_info["jacobian_mode_original"] = jac_mode_orig

        # Check if fieldsplit is being used
        linear_cfg = getattr(getattr(cfg, "solver", None), "linear", None)
        pc_type = None
        if linear_cfg is not None:
            pc_type = getattr(linear_cfg, "pc_type", None)
        if debug_enabled and is_root:
            logger.debug("  pc_type: %s", pc_type)

        # Force fieldsplit + mf -> mfpc_sparse_fd (ensures P is assemblable AIJ)
        if pc_type == "fieldsplit" and jac_mode_orig == "mf":
            if is_root:
                logger.warning("Promoting jacobian_mode: mf -> mfpc_sparse_fd")
            petsc_cfg.jacobian_mode = "mfpc_sparse_fd"
            diag_info["jacobian_mode_effective"] = "mfpc_sparse_fd"
            if debug_enabled and is_root:
                logger.debug(
                    "  After mutation: petsc_cfg.jacobian_mode = %s",
                    getattr(petsc_cfg, "jacobian_mode", "MISSING"),
                )
        else:
            diag_info["jacobian_mode_effective"] = jac_mode_orig
    else:
        diag_info["jacobian_mode_original"] = "mf"
        diag_info["jacobian_mode_effective"] = "mf"

    if debug_enabled and is_root:
        logger.debug("  Returning diag_info: %s", diag_info)
    return diag_info


def _create_identity_precond_mat(comm, dm):
    """
    Stage P1.5: DM-driven MPIAIJ identity preconditioner matrix.
    Ownership ranges follow dm.createGlobalVec().
    """
    PETSc = _get_petsc()
    x_template = dm.createGlobalVec()
    nloc = int(x_template.getLocalSize())
    n_glob = int(x_template.getSize())
    r0, r1 = x_template.getOwnershipRange()

    P = PETSc.Mat().create(comm=comm)
    P.setSizes(((nloc, n_glob), (nloc, n_glob)))
    try:
        P.setType(PETSc.Mat.Type.AIJ)
    except Exception:
        P.setType("aij")
    try:
        P.setPreallocationNNZ(1)
    except Exception:
        pass
    try:
        P.setUp()
    except Exception:
        pass

    for row in range(int(r0), int(r1)):
        P.setValue(row, row, 1.0)

    P.assemblyBegin()
    P.assemblyEnd()
    return P


def _apply_linesearch_options_from_env(PETSc, prefix: str, default_type: str) -> str:
    """
    Inject snes_linesearch_* options from env into PETSc.Options, without
    relying on petsc4py SNES.getLineSearch().
    """
    linesearch_env = os.environ.get("DROPLET_PETSC_LINESEARCH_TYPE", "").strip()
    desired = linesearch_env or default_type

    try:
        opts = PETSc.Options(prefix)
        try:
            current = opts.getString("snes_linesearch_type", default="")
        except Exception:
            current = ""

        if not current and desired:
            try:
                opts.setValue("snes_linesearch_type", desired)
            except Exception:
                opts["snes_linesearch_type"] = desired

        ls_damping = os.environ.get("DROPLET_PETSC_LINESEARCH_DAMPING", "").strip()
        if ls_damping:
            try:
                opts.setValue("snes_linesearch_damping", ls_damping)
            except Exception:
                opts["snes_linesearch_damping"] = ls_damping
    except Exception:
        current = ""

    return current or desired


def _vec_get_array_read(vec):
    """
    Return a read-only array view and a mode flag for safe restore.
    """
    try:
        return vec.getArrayRead(), "read"
    except Exception:
        try:
            return vec.getArray(readonly=True), "read"
        except Exception:
            return vec.getArray(), "write"


def _vec_restore_array(vec, arr, mode: str) -> None:
    try:
        if mode == "read" and hasattr(vec, "restoreArrayRead"):
            vec.restoreArrayRead(arr)
        else:
            vec.restoreArray()
    except Exception:
        pass


def _log_linesearch_options(PETSc, prefix: str, rank: int) -> None:
    """
    Debug helper to report whether PETSc options picked up line search flags.
    """
    try:
        opts_global = PETSc.Options()
        ls_global = opts_global.getString("snes_linesearch_type", default="(unset)")
        mon_global = opts_global.getBool("snes_linesearch_monitor", default=False)
        if prefix:
            opts_pref = PETSc.Options(prefix)
            ls_pref = opts_pref.getString("snes_linesearch_type", default="(unset)")
            mon_pref = opts_pref.getBool("snes_linesearch_monitor", default=False)
            logger.debug(
                "[DBG][rank=%d] PETSc opts: snes_linesearch_type=%s monitor=%s "
                "(prefix=%s -> %s/%s)",
                rank,
                ls_global,
                bool(mon_global),
                prefix,
                ls_pref,
                bool(mon_pref),
            )
        else:
            logger.debug(
                "[DBG][rank=%d] PETSc opts: snes_linesearch_type=%s monitor=%s",
                rank,
                ls_global,
                bool(mon_global),
            )
    except Exception:
        logger.debug("[DBG][rank=%d] PETSc opts check failed", rank)


def solve_nonlinear_petsc_parallel(
    ctx: NonlinearContext,
    u0: np.ndarray,
) -> NonlinearSolveResult:
    """
    Solve F(u) = 0 using PETSc SNES (parallel-only, DM + MFFD).
    """
    cfg = ctx.cfg
    nl_cfg = getattr(cfg, "nonlinear", None)
    if nl_cfg is None or not getattr(nl_cfg, "enabled", False):
        raise ValueError("Nonlinear solver requested but cfg.nonlinear.enabled is False or missing.")

    # P3.5-5-1: Normalize config for smoke mode (limits iterations, ensures AIJ P for fieldsplit)
    smoke_diag = _normalize_cfg_for_snes_smoke(cfg)

    # P3.5-5-1 DEBUG: Log smoke_diag returned value
    if _debug_enabled() and is_root_rank():
        logger.debug("[P3.5-5-1 DEBUG] After _normalize_cfg_for_snes_smoke:")
        logger.debug("  smoke_diag = %s", smoke_diag)
        logger.debug("  nl_cfg.max_outer_iter = %s", getattr(nl_cfg, "max_outer_iter", "MISSING"))

    LinearSolverConfigTyped.from_cfg(cfg)
    validate_mpi_before_petsc(cfg)

    PETSc = _get_petsc()
    comm = PETSc.COMM_WORLD
    world_rank = int(comm.getRank())
    world_size = int(comm.getSize())

    if world_rank == 0:
        logger.info(
            "PETSc parallel SNES (COMM_WORLD) starting: size=%d (no serial fallback in this module).",
            world_size,
        )

    petsc_cfg = getattr(cfg, "petsc", None)

    meta = getattr(ctx, "meta", None)
    if meta is None:
        meta = {}
        ctx.meta = meta
    elif not isinstance(meta, dict):
        meta = dict(meta)
        ctx.meta = meta
    # Dedicated MPI backend uses physical x and residual spaces.
    meta["petsc_x_space"] = "physical"
    meta["petsc_f_space"] = "physical"

    dm_mgr = meta.get("dm_manager") or meta.get("dm_mgr") or getattr(ctx, "dm_manager", None)
    dm = meta.get("dm") or getattr(ctx, "dm", None)
    ld = meta.get("layout_dist") or meta.get("ld") or getattr(ctx, "layout_dist", None)

    if dm_mgr is None:
        from parallel.dm_manager import build_dm
        from core.layout_dist import LayoutDistributed

        if world_rank == 0:
            logger.info("Parallel SNES auto-building DM/layout_dist for COMM_WORLD.")

        dm_mgr = build_dm(cfg, ctx.layout, comm=comm)
        dm = dm_mgr.dm
        ld = LayoutDistributed.build(comm, dm_mgr, ctx.layout)

        meta["dm_manager"] = dm_mgr
        meta["dm_mgr"] = dm_mgr
        meta["dm"] = dm
        meta["layout_dist"] = ld
        meta["ld"] = ld
        try:
            ctx.dm_manager = dm_mgr
        except Exception:
            pass
        try:
            ctx.dm = dm
        except Exception:
            pass
        try:
            ctx.layout_dist = ld
        except Exception:
            pass
    else:
        if dm is None:
            dm = getattr(dm_mgr, "dm", None)
        if ld is None:
            from core.layout_dist import LayoutDistributed

            ld = LayoutDistributed.build(comm, dm_mgr, ctx.layout)
            meta["layout_dist"] = ld
            meta["ld"] = ld
            try:
                ctx.layout_dist = ld
            except Exception:
                pass
        if dm is not None:
            meta.setdefault("dm", dm)
        meta.setdefault("dm_manager", dm_mgr)
        meta.setdefault("dm_mgr", dm_mgr)

    if dm is None or ld is None:
        raise RuntimeError("Parallel SNES failed to initialize DM/layout_dist; check configuration.")

    v_check = None
    try:
        v_check = dm.createGlobalVec()
        v_lo, v_hi = (int(x) for x in v_check.getOwnershipRange())
        v_size = int(v_check.getSize())
    finally:
        if v_check is not None:
            try:
                v_check.destroy()
            except Exception:
                pass

    if getattr(ld, "ownership_range", None) is None:
        raise ValueError("LayoutDistributed.ownership_range is None (Stage1 requires it).")
    if tuple(int(x) for x in ld.ownership_range) != (v_lo, v_hi):
        raise ValueError(
            f"Ownership mismatch: DM={(v_lo, v_hi)} vs LD={ld.ownership_range}"
        )
    ld_global = getattr(ld, "global_size", getattr(ld, "N_total", None))
    if ld_global is not None and int(ld_global) != v_size:
        raise ValueError(
            f"Global size mismatch: DM={v_size} vs LD.global_size={ld_global}"
        )
    layout_size = int(ctx.layout.n_dof())
    if v_size != layout_size:
        raise ValueError(
            f"Global size mismatch: DM={v_size} vs layout.size={layout_size}"
        )
    if hasattr(ld, "local_size") and int(ld.local_size) != int(v_hi - v_lo):
        raise ValueError(
            f"Local size mismatch: LD.local_size={ld.local_size} vs ownership_range_len={v_hi - v_lo}"
        )

    max_outer_iter = int(getattr(nl_cfg, "max_outer_iter", 20))

    # P3.5-5-1 DEBUG: Log max_outer_iter read from nl_cfg
    if _debug_enabled() and world_rank == 0:
        logger.debug("[P3.5-5-1 DEBUG] Reading max_outer_iter from nl_cfg:")
        logger.debug("  max_outer_iter = %s", max_outer_iter)
        logger.debug("  nl_cfg.max_outer_iter = %s", getattr(nl_cfg, "max_outer_iter", "MISSING"))
        logger.debug("  Expected from smoke_diag: %s", smoke_diag.get("max_outer_iter_effective", "N/A"))

    # Fixed tolerances for the dedicated MPI backend.
    f_rtol = 1.0e-6
    f_atol = 1.0e-8
    verbose = bool(getattr(nl_cfg, "verbose", False))
    log_every = int(getattr(nl_cfg, "log_every", 5))
    if log_every < 1:
        log_every = 1

    jacobian_mode_raw = "mf"
    if petsc_cfg is not None:
        jacobian_mode_raw = getattr(petsc_cfg, "jacobian_mode", "mf")

    # P3.5-5-1 DEBUG: Log jacobian_mode read from petsc_cfg
    if _debug_enabled() and world_rank == 0:
        logger.debug("[P3.5-5-1 DEBUG] Reading jacobian_mode from petsc_cfg:")
        logger.debug("  jacobian_mode_raw = %s", jacobian_mode_raw)
        logger.debug("  Expected from smoke_diag: %s", smoke_diag.get("jacobian_mode_effective", "N/A"))
    jacobian_mode = JacobianMode.normalize(jacobian_mode_raw)
    if jacobian_mode not in (JacobianMode.MF, JacobianMode.MFPC_SPARSE_FD):
        raise ValueError(
            "Parallel SNES backend supports jacobian_mode='mf' or 'mfpc_sparse_fd' only. "
            f"Got: {jacobian_mode!r}."
        )
    linear_cfg = getattr(getattr(cfg, "solver", None), "linear", None)
    pc_type_override_raw = os.environ.get("DROPLET_PETSC_PC_TYPE_OVERRIDE", "").strip()
    pc_type_override_req = _normalize_pc_type(pc_type_override_raw) if pc_type_override_raw else None
    pc_type_cfg_req = _normalize_pc_type(getattr(linear_cfg, "pc_type", None)) if linear_cfg is not None else None
    pc_type_req = pc_type_override_req if pc_type_override_req is not None else pc_type_cfg_req
    if pc_type_req == "fieldsplit" and jacobian_mode == JacobianMode.MF:
        if world_rank == 0:
            logger.warning(
                "pc_type=fieldsplit with jacobian_mode=mf uses identity P; "
                "promoting to mfpc_sparse_fd for a usable preconditioner."
            )
        jacobian_mode = JacobianMode.MFPC_SPARSE_FD
    use_mfpc_sparse_fd = (jacobian_mode == JacobianMode.MFPC_SPARSE_FD)
    fd_eps = float(getattr(petsc_cfg, "fd_eps", 1.0e-8)) if petsc_cfg is not None else 1.0e-8
    precond_drop_tol = float(getattr(petsc_cfg, "precond_drop_tol", 0.0)) if petsc_cfg is not None else 0.0
    precond_max_nnz_row = None
    if petsc_cfg is not None and hasattr(petsc_cfg, "precond_max_nnz_row"):
        try:
            precond_max_nnz_row = int(getattr(petsc_cfg, "precond_max_nnz_row"))
        except Exception:
            precond_max_nnz_row = None

    # Dedicated MPI backend uses physical unknowns/residuals (see meta flags above).

    prefix = _normalize_options_prefix(
        getattr(petsc_cfg, "options_prefix", None) if petsc_cfg is not None else None
    )

    snes_type = "newtonls"
    linesearch_type = os.environ.get("DROPLET_PETSC_LINESEARCH_TYPE", "").strip() or "l2"
    snes_monitor = bool(getattr(petsc_cfg, "snes_monitor", False)) if petsc_cfg is not None else False

    ksp_type = "gmres"
    ksp_rtol = 1.0e-6
    ksp_atol = 1.0e-12
    ksp_max_it = 200
    ksp_restart = 30

    if world_rank == 0:
        logger.info(
            "Parallel SNES fixed config: snes=%s/%s, ksp=%s",
            snes_type,
            linesearch_type,
            ksp_type,
        )

    u0 = np.asarray(u0, dtype=np.float64)
    n_layout = int(ctx.layout.n_dof())
    if u0.shape != (n_layout,):
        raise ValueError(f"u0 shape {u0.shape} incompatible with layout size {n_layout}.")

    from assembly.residual_local import ResidualLocalCtx, pack_local_to_layout, scatter_layout_to_local
    from parallel.dm_manager import global_to_local, local_state_to_global

    ctx_local = ResidualLocalCtx(layout=ctx.layout, ld=ld)

    def _layout_to_dm_vec(u_layout: np.ndarray):
        if comm.getSize() == 1:
            Xg = dm_mgr.dm.createGlobalVec()
            u_arr = np.asarray(u_layout, dtype=np.float64)
            try:
                x_view = Xg.getArray()
                if u_arr.shape[0] != x_view.shape[0]:
                    raise ValueError(
                        f"serial fastpath: len(u)={u_arr.shape[0]} != len(x)={x_view.shape[0]}"
                    )
                x_view[:] = u_arr
            finally:
                try:
                    Xg.restoreArray()
                except Exception:
                    pass
            return Xg
        Xl_liq = dm_mgr.dm_liq.createLocalVec()
        Xl_gas = dm_mgr.dm_gas.createLocalVec()
        Xl_if = dm_mgr.dm_if.createLocalVec()
        Xl_liq.set(0.0)
        Xl_gas.set(0.0)
        Xl_if.set(0.0)
        aXl_liq = dm_mgr.dm_liq.getVecArray(Xl_liq)
        aXl_gas = dm_mgr.dm_gas.getVecArray(Xl_gas)
        aXl_if = Xl_if.getArray()
        scatter_layout_to_local(
            ctx_local,
            u_layout,
            aXl_liq,
            aXl_gas,
            aXl_if,
            rank=world_rank,
            owned_only=True,
        )
        return local_state_to_global(dm_mgr, Xl_liq, Xl_gas, Xl_if)

    def _dm_vec_to_layout(Xg_vec) -> np.ndarray:
        if comm.getSize() == 1:
            try:
                try:
                    x_view = Xg_vec.getArray(readonly=True)
                except TypeError:
                    x_view = Xg_vec.getArray()
                return np.array(x_view, dtype=np.float64, copy=True)
            finally:
                try:
                    Xg_vec.restoreArray()
                except Exception:
                    pass
        Xl_liq, Xl_gas, Xl_if = global_to_local(dm_mgr, Xg_vec)
        aXl_liq = dm_mgr.dm_liq.getVecArray(Xl_liq)
        aXl_gas = dm_mgr.dm_gas.getVecArray(Xl_gas)
        aXl_if = Xl_if.getArray()
        u_local = pack_local_to_layout(ctx_local, aXl_liq, aXl_gas, aXl_if, rank=world_rank)
        if comm.getSize() == 1:
            return u_local
        try:
            from mpi4py import MPI

            mpicomm = comm.tompi4py()
            return mpicomm.allreduce(u_local, op=MPI.SUM)
        except Exception as exc:
            raise RuntimeError("mpi4py is required for DM to layout gather in MPI mode.") from exc

    def _get_or_build_perm():
        perm_l2d = meta.get("perm_l2d")
        perm_d2l = meta.get("perm_d2l")
        perm_layout_mask = meta.get("perm_layout_mask")
        if perm_l2d is not None and perm_d2l is not None and perm_layout_mask is not None:
            return perm_l2d, perm_d2l, perm_layout_mask

        if comm.getSize() == 1:
            perm_l2d = np.arange(n_layout, dtype=np.int64)
            perm_d2l = perm_l2d.copy()
            perm_layout_mask = np.ones(n_layout, dtype=bool)
            meta["perm_l2d"] = perm_l2d
            meta["perm_d2l"] = perm_d2l
            meta["perm_layout_mask"] = perm_layout_mask
            return perm_l2d, perm_d2l, perm_layout_mask

        Xg_gid = dm_mgr.dm.createGlobalVec()
        Xl_liq = None
        Xl_gas = None
        Xl_if = None
        try:
            lo, hi = Xg_gid.getOwnershipRange()
            lo = int(lo)
            hi = int(hi)
            try:
                x_view = Xg_gid.getArray()
                x_view[:] = np.arange(lo, hi, dtype=np.float64) + 1.0
            finally:
                try:
                    Xg_gid.restoreArray()
                except Exception:
                    pass

            Xl_liq, Xl_gas, Xl_if = global_to_local(dm_mgr, Xg_gid)
            aXl_liq = dm_mgr.dm_liq.getVecArray(Xl_liq)
            aXl_gas = dm_mgr.dm_gas.getVecArray(Xl_gas)
            aXl_if = Xl_if.getArray()
            u_layout_local = pack_local_to_layout(
                ctx_local,
                aXl_liq,
                aXl_gas,
                aXl_if,
                rank=world_rank,
            )
            local_idx = np.nonzero(u_layout_local)[0].astype(np.int64)
            perm_layout_mask = np.zeros(n_layout, dtype=bool)
            perm_layout_mask[local_idx] = True

            u_layout_global = _dm_vec_to_layout(Xg_gid)
        finally:
            try:
                Xg_gid.destroy()
            except Exception:
                pass
            for vec in (Xl_liq, Xl_gas, Xl_if):
                if vec is None:
                    continue
                try:
                    vec.destroy()
                except Exception:
                    pass

        perm_l2d = np.rint(u_layout_global).astype(np.int64) - 1
        if perm_l2d.size != n_layout:
            raise RuntimeError(
                f"[perm] layout size mismatch: perm={perm_l2d.size} layout={n_layout}"
            )
        if np.any(perm_l2d < 0):
            raise RuntimeError("[perm] invalid layout->DM permutation (negative entries).")
        perm_d2l = np.empty_like(perm_l2d)
        perm_d2l[perm_l2d] = np.arange(n_layout, dtype=np.int64)
        if not np.array_equal(perm_d2l[perm_l2d], np.arange(n_layout, dtype=np.int64)):
            raise RuntimeError("[perm] perm_d2l is not the inverse of perm_l2d.")
        if not np.array_equal(perm_l2d[perm_d2l], np.arange(n_layout, dtype=np.int64)):
            raise RuntimeError("[perm] perm_l2d is not the inverse of perm_d2l.")

        meta["perm_l2d"] = perm_l2d
        meta["perm_d2l"] = perm_d2l
        meta["perm_layout_mask"] = perm_layout_mask
        return perm_l2d, perm_d2l, perm_layout_mask

    def residual_petsc_owned_rows(dm_mgr, ctx, X, Fg):
        """
        MPI: each rank writes only its owned rows of residual into Fg.
        """
        PETSc = _get_petsc()
        comm = PETSc.COMM_WORLD
        rank = int(comm.getRank())

        u_layout = _dm_vec_to_layout(X)

        rstart, rend = Fg.getOwnershipRange()
        rstart = int(rstart)
        rend = int(rend)
        nloc = int(rend - rstart)

        perm_l2d, perm_d2l, perm_layout_mask = _get_or_build_perm()
        idx_dm = np.arange(rstart, rend, dtype=np.int64)
        idx_layout = perm_d2l[idx_dm]

        if not np.all(perm_layout_mask[idx_layout]):
            if _debug_enabled() and rank == 0:
                bad = np.nonzero(~perm_layout_mask[idx_layout])[0]
                sample = bad[:10].tolist()
                logger.debug(
                    "[DBG][rank=%d] residual_petsc_owned_rows fallback: "
                    "DM-owned rows map to nonlocal layout rows. bad_count=%d sample_local_idx=%s",
                    rank,
                    int(bad.size),
                    sample,
                )
            residual_petsc(dm_mgr, ld, ctx, X, Fg=Fg)
            try:
                f_view, f_mode = _vec_get_array_read(Fg)
                r_owned = np.array(f_view, dtype=np.float64, copy=True)
            finally:
                _vec_restore_array(Fg, f_view, f_mode)
        else:
            res_full = residual_only(u_layout, ctx)
            r_owned = np.asarray(res_full[idx_layout], dtype=np.float64)
            if _debug_enabled() and rank == 0:
                Ftmp = dm_mgr.dm.createGlobalVec()
                try:
                    residual_petsc(dm_mgr, ld, ctx, X, Fg=Ftmp)
                    try:
                        f_view, f_mode = _vec_get_array_read(Ftmp)
                        f_loc = np.array(f_view, dtype=np.float64, copy=True)
                    finally:
                        _vec_restore_array(Ftmp, f_view, f_mode)
                    err = float(np.max(np.abs(f_loc - r_owned))) if f_loc.size else 0.0
                    logger.debug(
                        "[DBG][rank=%d] OwnedRows fastpath vs full residual: ||diff||_inf = %.3e",
                        rank,
                        err,
                    )
                finally:
                    try:
                        Ftmp.destroy()
                    except Exception:
                        pass

        if not np.all(np.isfinite(r_owned)):
            r_owned = np.where(np.isfinite(r_owned), r_owned, 1.0e20)

        if r_owned.size != nloc:
            raise RuntimeError(
                f"[residual_petsc_owned_rows][rank={rank}] size mismatch: "
                f"len(r_owned)={r_owned.size} vs nloc={nloc} range=[{rstart},{rend})"
            )

        Fg.set(0.0)
        idx = np.arange(rstart, rend, dtype=PETSc.IntType)
        Fg.setValues(idx, r_owned, addv=PETSc.InsertMode.INSERT_VALUES)
        Fg.assemblyBegin()
        Fg.assemblyEnd()

        if _debug_enabled() and rank == 0:
            try:
                f_view, f_mode = _vec_get_array_read(Fg)
                f_loc = np.array(f_view, dtype=np.float64, copy=True)
            finally:
                _vec_restore_array(Fg, f_view, f_mode)
            err = float(np.max(np.abs(f_loc - r_owned))) if f_loc.size else 0.0
            logger.debug(
                "[DBG][rank=%d] OwnedRows writeback check: ||Fg_local - r_owned||_inf = %.3e",
                rank,
                err,
            )

        return r_owned

    def _dbg_log_Y(tag: str, state_dbg) -> None:
        if not DEBUG:
            return
        try:
            Yg = np.asarray(state_dbg.Yg, dtype=np.float64)
            if Yg.ndim != 2:
                _dbg_rank("%s Yg shape=%r", tag, getattr(Yg, "shape", None))
                return
            cell_idx = 7
            if Yg.shape[1] <= cell_idx:
                _dbg_rank("%s Yg cell=%d out of range (ncell=%d)", tag, cell_idx, int(Yg.shape[1]))
                return
            Y = Yg[:, cell_idx]
            s = float(np.sum(Y)) if Y.size else 0.0
            mn = float(np.min(Y)) if Y.size else 0.0
            mx = float(np.max(Y)) if Y.size else 0.0
            idx = np.argsort(-Y)[:3] if Y.size else np.array([], dtype=np.int64)
            top = [(int(i), float(Y[i])) for i in idx]
            _dbg_rank(
                "%s Y@cell%d sum=%.6g min=%.3e max=%.3e top=%s",
                tag,
                cell_idx,
                s,
                mn,
                mx,
                top,
            )
        except Exception as exc:
            _dbg_rank("%s Y debug failed: %r", tag, exc)

    def _debug_check_xsig(state_dbg, rank: int) -> None:
        if rank != 0:
            return
        ok = True
        try:
            Y = np.asarray(state_dbg.Yg, dtype=np.float64)
            if Y.ndim != 2:
                logger.debug("[XSIG_DEBUG][rank=%d] Yg shape=%r", rank, getattr(Y, "shape", None))
                return
            Y_sum = np.sum(Y, axis=0)
        except Exception as exc:
            logger.debug("[XSIG_DEBUG][rank=%d] failed to access Yg: %r", rank, exc)
            return

        if not np.all(np.isfinite(Y)):
            logger.debug("[XSIG_DEBUG][rank=%d] non-finite values in Yg", rank)
            ok = False

        neg = np.where(Y < -1.0e-8)
        if neg[0].size:
            sample = (int(neg[0][0]), int(neg[1][0]), float(Y[neg[0][0], neg[1][0]]))
            logger.debug(
                "[XSIG_DEBUG][rank=%d] negative Yg entries: count=%d sample=%s",
                rank,
                int(neg[0].size),
                str([sample]),
            )
            ok = False

        too_large = np.where(Y > 0.79 + 1.0e-2)
        if too_large[0].size:
            sample = (
                int(too_large[0][0]),
                int(too_large[1][0]),
                float(Y[too_large[0][0], too_large[1][0]]),
            )
            logger.debug(
                "[XSIG_DEBUG][rank=%d] too large Yg entries: count=%d sample=%s",
                rank,
                int(too_large[0].size),
                str([sample]),
            )
            ok = False

        if not np.allclose(Y_sum, 1.0, atol=1.0e-6):
            logger.debug(
                "[XSIG_DEBUG][rank=%d] sum(Yg)!=1 for some cells: max|sum-1|=%e",
                rank,
                float(np.max(np.abs(Y_sum - 1.0))),
            )
            ok = False

        if not ok:
            logger.debug("[DBG][rank=%d] Xsig debug failed", rank)

    history_inf: List[float] = []
    last_inf = {"val": np.nan}
    t_solve0 = time.perf_counter()
    ctr = {"n_func_eval": 0, "n_jac_eval": 0, "ksp_its_total": 0}
    tim = {"time_func": 0.0, "time_jac": 0.0, "time_linear_total": 0.0}

    def snes_func(snes_obj, X, F):
        t0 = time.perf_counter()
        try:
            iter_no = int(snes_obj.getIterationNumber())
        except Exception:
            iter_no = -1
        phase_label = f"SNES_F_iter{iter_no}"
        sig_changed = False
        set_gas_eval_phase(phase_label)
        layout_mod.set_apply_u_tag(phase_label)
        try:
            if DEBUG_DOF_GID is not None:
                try:
                    try:
                        lo, hi = dm.getOwnershipRange()
                    except Exception:
                        lo, hi = X.getOwnershipRange()
                    if lo <= DEBUG_DOF_GID < hi:
                        local_i = int(DEBUG_DOF_GID - lo)
                        try:
                            x_view, x_mode = _vec_get_array_read(X)
                            x_arr = np.array(x_view, dtype=np.float64, copy=True)
                        finally:
                            _vec_restore_array(X, x_view, x_mode)
                        x_val = float(x_arr[local_i])
                        debug_key = "_debug_x0_local"
                        if debug_key not in ctx.meta:
                            ctx.meta[debug_key] = X.copy()
                            if world_rank == 0:
                                logger.debug(
                                    "[SNES_FUNC_DEBUG][rank=%d] init x0[gid=%d, loc=%d]=%e (||x0||_inf=%e)",
                                    world_rank,
                                    int(DEBUG_DOF_GID),
                                    local_i,
                                    x_val,
                                    float(np.linalg.norm(x_arr, ord=np.inf)) if x_arr.size else 0.0,
                                )
                        else:
                            x0_vec = ctx.meta[debug_key]
                            try:
                                x0_view, x0_mode = _vec_get_array_read(x0_vec)
                                x0_arr = np.array(x0_view, dtype=np.float64, copy=True)
                            finally:
                                _vec_restore_array(x0_vec, x0_view, x0_mode)
                            dx_val = float(x_val - x0_arr[local_i])
                            call_count = int(ctx.meta.get("_debug_call_count", 0))
                            if world_rank == 0:
                                logger.debug(
                                    "[SNES_FUNC_DEBUG][rank=%d] call=%d gid=%d loc=%d x=%e dx=%e (||x||_inf=%e)",
                                    world_rank,
                                    call_count,
                                    int(DEBUG_DOF_GID),
                                    local_i,
                                    x_val,
                                    dx_val,
                                    float(np.linalg.norm(x_arr, ord=np.inf)) if x_arr.size else 0.0,
                                )
                            ctx.meta["_debug_call_count"] = call_count + 1
                except Exception as exc:
                    logger.debug("DEBUG_DOF_GID logging failed: %r", exc)
            if DEBUG:
                try:
                    x_inf = float(X.norm(PETSc.NormType.NORM_INFINITY))
                    x_2 = float(X.norm(PETSc.NormType.NORM_2))
                    try:
                        x_view, x_mode = _vec_get_array_read(X)
                        x_loc = np.array(x_view, dtype=np.float64, copy=True)
                    finally:
                        _vec_restore_array(X, x_view, x_mode)
                    stride = max(1, int(x_loc.size // 16)) if x_loc.size else 1
                    x_chk = float(x_loc[::stride].sum()) if x_loc.size else 0.0
                    sig = (x_inf, x_2, x_chk, int(x_loc.size))
                    prev = ctx.meta.get("_dbg_prev_x_sig")
                    if prev != sig:
                        ctx.meta["_dbg_prev_x_sig"] = sig
                        sig_changed = True
                        _dbg_rank(
                            "Xsig inf=%.3e 2=%.3e chk=%.6e nloc=%d",
                            x_inf,
                            x_2,
                            x_chk,
                            int(x_loc.size),
                        )
                except Exception as exc:
                    _dbg_rank(
                        "Xsig debug failed: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
            if DEBUG:
                try:
                    u_layout_dbg = _dm_vec_to_layout(X)
                    state_dbg = ctx.make_state_from_u(u_layout_dbg)
                    _dbg_log_Y("AFTER_VEC2STATE_BEFORE_PROPS", state_dbg)
                    _debug_check_xsig(state_dbg, world_rank)
                except Exception as exc:
                    _dbg_rank("Y debug before props failed: %r", exc)
            use_owned = bool(int(os.environ.get("DROPLET_PETSC_RESID_OWNED", "1")))
            if use_owned:
                residual_petsc_owned_rows(dm_mgr, ctx, X, F)
            else:
                residual_petsc(dm_mgr, ld, ctx, X, Fg=F)
        finally:
            layout_mod.set_apply_u_tag(None)
            set_gas_eval_phase(None)
        try:
            res_inf = float(F.norm(PETSc.NormType.NORM_INFINITY))
        except Exception:
            res_inf = float(F.norm())
        last_inf["val"] = res_inf
        ctr["n_func_eval"] += 1
        tim["time_func"] += (time.perf_counter() - t0)
        if DEBUG:
            if sig_changed:
                _dbg_rank("Fsig inf=%.3e", res_inf)
            try:
                rstart_f, rend_f = F.getOwnershipRange()
                try:
                    f_view, f_mode = _vec_get_array_read(F)
                    f_loc = np.array(f_view, dtype=np.float64, copy=True)
                finally:
                    _vec_restore_array(F, f_view, f_mode)
                _dbg_rank(
                    "Fg(local) ||F||_inf=%.3e  local_n=%d  range=[%d,%d)",
                    float(np.max(np.abs(f_loc))),
                    int(f_loc.size),
                    int(rstart_f),
                    int(rend_f),
                )
            except Exception:
                _dbg_rank("Fg(local) debug failed")

    def snes_monitor_fn(snes_obj, its, fnorm):
        v = float(last_inf["val"])
        if np.isfinite(v):
            history_inf.append(v)
        if snes_monitor or verbose:
            if (its + 1) % log_every == 0:
                logger.info("snes iter=%d fnorm=%.3e res_inf=%.3e", its + 1, float(fnorm), v)

    snes = PETSc.SNES().create(comm=comm)
    if prefix:
        snes.setOptionsPrefix(prefix)

    try:
        snes.setDM(dm)
    except Exception:
        pass

    F = dm.createGlobalVec()
    try:
        snes.setFunction(snes_func, F)
    except TypeError:
        snes.setFunction(F, snes_func)
    DEBUG = os.environ.get("DROPLET_PETSC_DEBUG", "0") == "1"
    DEBUG_ONCE = os.environ.get("DROPLET_PETSC_DEBUG_ONCE", "1") == "1"

    def _dbg(msg, *args):
        if DEBUG and world_rank == 0:
            logger.debug("[DBG] " + msg, *args)

    def _dbg_rank(msg, *args):
        if DEBUG and world_rank == 0:
            logger.debug("[DBG][rank=%d] " + msg, world_rank, *args)

    if DEBUG:
        X0 = None
        F0 = None
        try:
            X0 = _layout_to_dm_vec(u0)
            F0 = dm.createGlobalVec()
            layout_mod.set_apply_u_tag("XSIG_TEST")
            set_gas_eval_phase("XSIG_TEST")
            try:
                residual_petsc(dm_mgr, ld, ctx, X0, Fg=F0)
            finally:
                layout_mod.set_apply_u_tag(None)
                set_gas_eval_phase(None)

            rstart_dm, rend_dm = F0.getOwnershipRange()
            _dbg_rank("DM ownership_range = [%d, %d)", int(rstart_dm), int(rend_dm))

            ld_range = None
            for key in ("ownership_range", "owned_range", "range_owned", "owned_dof_range"):
                if hasattr(ld, key):
                    ld_range = getattr(ld, key)
                    break
            if ld_range is not None:
                try:
                    _dbg_rank("LD ownership_range = %s", str(tuple(int(x) for x in ld_range)))
                except Exception:
                    _dbg_rank("LD ownership_range(raw) = %r", ld_range)
            else:
                _dbg_rank("LD ownership_range = <missing>")

            try:
                x0_view, x0_mode = _vec_get_array_read(X0)
                x0_loc = np.array(x0_view, dtype=np.float64, copy=True)
            finally:
                _vec_restore_array(X0, x0_view, x0_mode)
            _dbg_rank(
                "X0 local ||x||_inf=%.3e  local_size=%d",
                float(np.max(np.abs(x0_loc))),
                int(x0_loc.size),
            )

            try:
                u_back = _dm_vec_to_layout(X0)
                du = u_back - u0
                _dbg_rank("Roundtrip ||u_back-u0||_inf=%.3e", float(np.max(np.abs(du))))
                imax = int(np.argmax(np.abs(du)))
                _dbg_rank(
                    "Roundtrip max@i=%d: u0=%.6e u_back=%.6e du=%.6e",
                    imax,
                    float(u0[imax]),
                    float(u_back[imax]),
                    float(du[imax]),
                )
            except Exception as exc:
                _dbg_rank("Roundtrip failed: %r", exc)
        finally:
            if X0 is not None:
                try:
                    X0.destroy()
                except Exception:
                    pass
            if F0 is not None:
                try:
                    F0.destroy()
                except Exception:
                    pass
        if DEBUG_ONCE:
            os.environ["DROPLET_PETSC_DEBUG"] = "0"

    try:
        snes.setType(snes_type)
    except Exception:
        logger.warning("Unknown snes_type='%s', falling back to newtonls", snes_type)
        snes.setType("newtonls")

    # P3.5-5-1 DEBUG: Log SNES tolerance parameters before setting
    if _debug_enabled() and world_rank == 0:
        logger.debug("[P3.5-5-1 DEBUG] About to call snes.setTolerances:")
        logger.debug("  max_it = %s", max_outer_iter)
        logger.debug("  f_rtol = %s", f_rtol)
        logger.debug("  f_atol = %s", f_atol)

    snes.setTolerances(rtol=f_rtol, atol=f_atol, max_it=max_outer_iter)

    linesearch_type_eff = _apply_linesearch_options_from_env(PETSc, prefix, linesearch_type)
    if DEBUG and world_rank == 0:
        _log_linesearch_options(PETSc, prefix, world_rank)

    try:
        snes.setFromOptions()
    except Exception:
        pass

    try:
        opts_eff = PETSc.Options(prefix)
        linesearch_type_eff = opts_eff.getString(
            "snes_linesearch_type",
            default=linesearch_type_eff,
        )
    except Exception:
        pass

    snes_mf_enabled = _enable_snes_matrix_free(snes, prefix)

    X = dm.createGlobalVec()
    P = None
    fd_stats: Dict[str, Any] = {}
    if use_mfpc_sparse_fd:
        from assembly.build_sparse_fd_jacobian import build_sparse_fd_jacobian

        x_template = None
        try:
            x_template = dm.createGlobalVec()
            nloc = int(x_template.getLocalSize())
            n_glob = int(x_template.getSize())
        finally:
            if x_template is not None:
                try:
                    x_template.destroy()
                except Exception:
                    pass

        P_dm = PETSc.Mat().create(comm=comm)
        P_dm.setSizes(((nloc, n_glob), (nloc, n_glob)))
        try:
            P_dm.setType(PETSc.Mat.Type.AIJ)
        except Exception:
            P_dm.setType("aij")

        v_check = None
        try:
            v_check = dm.createGlobalVec()
            v_lo, v_hi = (int(x) for x in v_check.getOwnershipRange())
            if hasattr(ld, "ownership_range"):
                assert tuple(int(x) for x in ld.ownership_range) == (v_lo, v_hi), (
                    f"LD range {getattr(ld, 'ownership_range', None)} "
                    f"!= DM Vec range {(v_lo, v_hi)}"
                )
            p_lo, p_hi = (int(x) for x in P_dm.getOwnershipRange())
            assert (p_lo, p_hi) == (v_lo, v_hi), (
                f"P_dm range {(p_lo, p_hi)} != DM Vec range {(v_lo, v_hi)}"
            )
        finally:
            if v_check is not None:
                try:
                    v_check.destroy()
                except Exception:
                    pass

        t0 = time.perf_counter()
        try:
            P, fd_stats = build_sparse_fd_jacobian(
                ctx,
                np.asarray(u0, dtype=np.float64),
                eps=fd_eps,
                drop_tol=precond_drop_tol,
                pattern=None,
                mat=P_dm,
            )
        except Exception as exc:
            _dbg("build_sparse_fd_jacobian failed in parallel SNES: %r", exc)
            raise RuntimeError("Failed to build mfpc_sparse_fd preconditioner.") from exc
        finally:
            tim["time_jac"] += (time.perf_counter() - t0)
        ctr["n_jac_eval"] += 1
        try:
            p_range = tuple(int(v) for v in P.getOwnershipRange())
            x_range = tuple(int(v) for v in X.getOwnershipRange())
        except Exception:
            p_range = None
            x_range = None
        if p_range is not None and x_range is not None and p_range != x_range:
            raise RuntimeError(
                "mfpc_sparse_fd ownership range mismatch: "
                f"P={p_range} vs X={x_range}. "
                "Use jacobian_mode='mf' or update build_sparse_fd_jacobian to use DM ownership."
            )
        p_type_check = str(P.getType()).lower() if P is not None else ""
        if "aij" not in p_type_check:
            raise RuntimeError(f"mfpc_sparse_fd requires AIJ/MPIAIJ, got {p_type_check}")
    else:
        P = _create_identity_precond_mat(comm, dm)

    J = None
    try:
        J = PETSc.Mat().createSNESMF(snes)
    except Exception:
        try:
            J = PETSc.Mat().create(comm=comm)
            J.setType("mffd")
            J.setUp()
        except Exception:
            J = None

    try:
        snes.setJacobian(J=J, P=P, func=None)
    except TypeError:
        try:
            snes.setJacobian(J, P, None)
        except TypeError:
            try:
                snes.setJacobian(J, P)
            except TypeError:
                snes.setJacobian(P=P)

    ksp = snes.getKSP()
    if prefix:
        ksp.setOptionsPrefix(prefix)
    Aop = J
    Pop = P if P is not None else J
    if Aop is not None and Pop is not None:
        try:
            ksp.setOperators(Aop, Pop)
        except Exception:
            pass

    try:
        ksp.setFromOptions()
    except Exception:
        pass

    try:
        ksp.setType(ksp_type)
    except Exception:
        logger.warning("Unknown ksp_type='%s', falling back to gmres", ksp_type)
        ksp.setType("gmres")

    ksp.setInitialGuessNonzero(False)
    ksp.setTolerances(rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it)

    try:
        ksp_type_eff = str(ksp.getType()).lower()
        if ksp_type_eff in ("gmres", "fgmres"):
            ksp.setGMRESRestart(ksp_restart)
        elif ksp_type_eff == "lgmres" and hasattr(ksp, "setLGMRESRestart"):
            ksp.setLGMRESRestart(ksp_restart)
    except Exception:
        logger.debug("Unable to set restart for ksp_type='%s'", ksp.getType())

    pc_type_override_raw = os.environ.get("DROPLET_PETSC_PC_TYPE_OVERRIDE", "").strip()
    pc_type_override = _normalize_pc_type(pc_type_override_raw) if pc_type_override_raw else None
    pc_overridden = pc_type_override is not None

    linear_cfg = getattr(getattr(cfg, "solver", None), "linear", None)
    pc_type_cfg = _normalize_pc_type(getattr(linear_cfg, "pc_type", None)) if linear_cfg is not None else None
    pc_type_eff = pc_type_override if pc_type_override is not None else pc_type_cfg
    if pc_type_eff == "fieldsplit":
        fs_cfg = getattr(linear_cfg, "fieldsplit", None) if linear_cfg is not None else None
        if isinstance(fs_cfg, Mapping):
            fs_type = fs_cfg.get("type", "additive")
        else:
            fs_type = getattr(fs_cfg, "type", "additive") if fs_cfg is not None else "additive"
        fs_type = str(fs_type).strip().lower() or "additive"
        if fs_type not in ("additive", "schur"):
            raise ValueError(
                "Parallel SNES supports fieldsplit types 'additive' and 'schur' only "
                f"(got '{fs_type}')."
            )

    Aop_call = Aop
    Pop_call = Pop

    diag_pc = apply_structured_pc(
        ksp=ksp,
        cfg=cfg,
        layout=ctx.layout,
        A=Aop_call,
        P=Pop_call,
        pc_type_override=pc_type_override,
    )
    snes.setMonitor(snes_monitor_fn)
    _finalize_ksp_config(ksp, diag_pc, from_options=False)

    ksp_state = {"monitor_enabled": False, "last_its": 0}
    ksp_t = {"in_solve": False, "t0": 0.0}

    def _ksp_monitor(ksp_obj, its, rnorm):
        if its == 0:
            if ksp_t["in_solve"]:
                tim["time_linear_total"] += (time.perf_counter() - ksp_t["t0"])
            ksp_t["in_solve"] = True
            ksp_t["t0"] = time.perf_counter()
            ksp_state["last_its"] = 0
            return
        if its > ksp_state["last_its"]:
            ctr["ksp_its_total"] += int(its - ksp_state["last_its"])
            ksp_state["last_its"] = int(its)

    try:
        ksp.setMonitor(_ksp_monitor)
        ksp_state["monitor_enabled"] = True
    except Exception:
        pass

    X_init = _layout_to_dm_vec(u0)
    try:
        X_init.copy(X)
    except Exception:
        X.set(0.0)
        X.axpy(1.0, X_init)

    snes.solve(None, X)
    if ksp_t.get("in_solve", False):
        tim["time_linear_total"] += (time.perf_counter() - ksp_t["t0"])
        ksp_t["in_solve"] = False

    try:
        apply_fieldsplit_subksp_defaults(ksp, diag_pc)
    except Exception as exc:
        diag_pc.setdefault("fieldsplit", {})["subksp_ops_error"] = f"{type(exc).__name__}:{exc}"

    u_final = _dm_vec_to_layout(X)

    reason = int(snes.getConvergedReason())
    ksp_reason = int(ksp.getConvergedReason())
    ksp_it = int(ksp.getIterationNumber())
    converged = reason > 0
    n_iter = int(snes.getIterationNumber())

    res_final = residual_only(u_final, ctx)
    res_norm_2 = float(np.linalg.norm(res_final))
    res_norm_inf = float(np.linalg.norm(res_final, ord=np.inf))

    if not converged and world_rank == 0:
        logger.warning("SNES not converged: reason=%d res_inf=%.3e", reason, res_norm_inf)

    j_type = "none"
    p_type = "none"
    x_type = ""
    f_type = ""
    try:
        x_type = str(X.getType())
        f_type = str(F.getType())
    except Exception:
        pass
    try:
        if J is not None:
            j_type = str(J.getType()).lower()
    except Exception:
        j_type = "none"
    try:
        if P is not None:
            p_type = str(P.getType()).lower()
    except Exception:
        p_type = "none"

    ksp_a_type = ""
    ksp_p_type = ""
    ksp_a_is_p = False
    ksp_a = None
    ksp_p = None
    try:
        ksp_a, ksp_p = ksp.getOperators()
        if ksp_a is not None:
            ksp_a_type = str(ksp_a.getType()).lower()
        if ksp_p is not None:
            ksp_p_type = str(ksp_p.getType()).lower()
        if ksp_a is not None and ksp_p is not None:
            try:
                ksp_a_is_p = bool(ksp_a.handle == ksp_p.handle)
            except Exception:
                ksp_a_is_p = bool(ksp_a is ksp_p)
    except Exception:
        pass

    if not j_type or j_type == "none":
        j_type = "mffd"

    extra: Dict[str, Any] = {
        "snes_reason": reason,
        "snes_reason_str": str(reason),
        "snes_type": snes.getType(),
        "linesearch_type": linesearch_type_eff,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "ksp_reason": ksp_reason,
        "ksp_reason_str": str(ksp_reason),
        "ksp_it": ksp_it,
        "snes_iter": n_iter,
        "jacobian_mode": jacobian_mode,
        "n_func_eval": int(ctr["n_func_eval"]),
        "n_jac_eval": int(ctr["n_jac_eval"]),
        "ksp_its_total": int(ctr["ksp_its_total"]),
        "time_func": float(tim["time_func"]),
        "time_jac": float(tim["time_jac"]),
        "time_linear_total": float(tim["time_linear_total"]),
        "time_total": float(time.perf_counter() - t_solve0),
        "snes_mf_enabled": bool(snes_mf_enabled),
        "pc_type_overridden": bool(pc_overridden),
        "J_mat_type": j_type,
        "P_mat_type": p_type,
    }
    extra["petsc_options_prefix"] = prefix
    try:
        extra["snes_options_prefix"] = str(snes.getOptionsPrefix() or "")
    except Exception:
        extra["snes_options_prefix"] = ""
    try:
        extra["ksp_options_prefix"] = str(ksp.getOptionsPrefix() or "")
    except Exception:
        extra["ksp_options_prefix"] = ""
    # P3.5-5-1: Record smoke mode diagnostics
    if smoke_diag:
        extra["snes_smoke"] = smoke_diag
    if ksp_a_type:
        extra["KSP_A_type"] = ksp_a_type
    if ksp_p_type:
        extra["KSP_P_type"] = ksp_p_type
    if ksp_a_type or ksp_p_type:
        ops_info: Dict[str, Any] = {}
        if ksp_a_type:
            ops_info["A_type"] = ksp_a_type
        if ksp_p_type:
            ops_info["P_type"] = ksp_p_type
        extra["ksp_operators"] = ops_info
    extra["KSP_A_is_P"] = bool(ksp_a_is_p)
    if ksp_p is not None:
        try:
            PETSc = _get_petsc()
            info = ksp_p.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
            pmat_info: Dict[str, Any] = {}
            if isinstance(info, dict):
                if "nz_used" in info:
                    pmat_info["nz_used"] = float(info.get("nz_used", 0.0))
                if "nz_allocated" in info:
                    pmat_info["nz_allocated"] = float(info.get("nz_allocated", 0.0))
            else:
                try:
                    pmat_info["nz_used"] = float(getattr(info, "nz_used"))
                except Exception:
                    pass
                try:
                    pmat_info["nz_allocated"] = float(getattr(info, "nz_allocated"))
                except Exception:
                    pass
            if pmat_info:
                extra["pmat_info"] = pmat_info
        except Exception:
            pass
    if use_mfpc_sparse_fd:
        stats = dict(fd_stats) if isinstance(fd_stats, dict) else {}
        stats.setdefault("eps", float(fd_eps))
        stats.setdefault("drop_tol", float(precond_drop_tol))
        if precond_max_nnz_row is not None:
            stats.setdefault("precond_max_nnz_row", int(precond_max_nnz_row))
        extra["mfpc_sparse_fd"] = stats
        fd_col_stats = stats.get("fd_jacobian_stats")
        if isinstance(fd_col_stats, dict):
            extra["fd_jacobian_stats"] = dict(fd_col_stats)
    extra["parallel_backend"] = "petsc_snes_parallel"
    extra["comm_kind"] = "world"

    try:
        extra["comm_size"] = int(comm.getSize())
    except Exception:
        pass
    if x_type:
        extra["X_vec_type"] = x_type
    if f_type:
        extra["F_vec_type"] = f_type
    try:
        extra["X_local_size"] = int(X.getLocalSize())
        extra["X_ownership_range"] = tuple(int(v) for v in X.getOwnershipRange())
    except Exception:
        pass
    try:
        extra["dm_type"] = str(dm.getType())
    except Exception:
        pass
    if diag_pc:
        extra["pc_structured"] = dict(diag_pc)
    if history_inf:
        extra["history_res_inf"] = list(history_inf)

    diag = NonlinearDiagnostics(
        converged=bool(converged),
        method=f"snes:{snes.getType()}",
        n_iter=int(n_iter),
        res_norm_2=res_norm_2,
        res_norm_inf=res_norm_inf,
        history_res_inf=list(history_inf),
        message=None if converged else f"SNES diverged (reason={reason})",
        extra=extra,
    )

    if world_rank == 0 and verbose:
        logger.info(
            "PETSc parallel SNES done: conv=%s reason=%d its=%d |F|_inf=%.3e ksp_its_total=%d",
            converged,
            reason,
            n_iter,
            res_norm_inf,
            int(ctr["ksp_its_total"]),
        )

    # Clean up PETSc objects to prevent memory leaks
    try:
        if X_init is not None:
            X_init.destroy()
    except Exception:
        pass
    try:
        if X is not None:
            X.destroy()
    except Exception:
        pass
    try:
        if F is not None:
            F.destroy()
    except Exception:
        pass
    try:
        if J is not None:
            J.destroy()
    except Exception:
        pass
    try:
        if P is not None:
            P.destroy()
    except Exception:
        pass
    try:
        if snes is not None:
            snes.destroy()
    except Exception:
        pass

    return NonlinearSolveResult(u=np.asarray(u_final, dtype=np.float64), diag=diag)
