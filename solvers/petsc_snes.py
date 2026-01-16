# -*- coding: utf-8 -*-
"""
PETSc SNES nonlinear solver wrapper (serial + MPI).

Stage 5.3 base + 5.4A:
- Supports different Jacobian modes via cfg.petsc.jacobian_mode:
  * "fd"            : dense finite-difference Jacobian (serial only or simple use)
  * "mf"            : PETSc matrix-free SNES (SNES-level MFFD)
  * "mfpc_sparse_fd": matrix-free operator + sparse FD preconditioner (serial-only for now)
  * "mfpc_aija"     : matrix-free operator + Aij preconditioner (serial-only for now)
- In MPI mode (COMM_WORLD size > 1) we currently only support jacobian_mode="mf".
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np

from assembly.build_precond_aij import build_precond_mat_aij_from_A, fill_precond_mat_aij_from_A
from assembly.build_sparse_fd_jacobian import build_sparse_fd_jacobian
from assembly.jacobian_pattern import build_jacobian_pattern
from assembly.residual_global import build_global_residual, residual_only
from solvers.linear_types import JacobianMode, LinearSolverConfig as LinearSolverConfigTyped
from solvers.nonlinear_context import NonlinearContext
from solvers.nonlinear_types import NonlinearDiagnostics, NonlinearSolveResult
from solvers.petsc_linear import apply_fieldsplit_subksp_defaults, apply_structured_pc, _normalize_pc_type

logger = logging.getLogger(__name__)


def _get_petsc():
    """
    Import petsc4py.PETSc with MPI bootstrap.
    """
    try:
        from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

        bootstrap_mpi_before_petsc()
        from petsc4py import PETSc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("petsc4py is required for PETSc SNES backend.") from exc
    return PETSc


def _enable_snes_matrix_free(snes, prefix: str) -> bool:
    """
    Enable SNES matrix-free Jacobian in a petsc4py-version tolerant way.

    Priority:
    1) Use snes.setUseMF(True) if available;
    2) Otherwise, set the -<prefix>snes_mf option via PETSc.Options.
    """
    PETSc = _get_petsc()
    # Preferred API in recent petsc4py versions
    if hasattr(snes, "setUseMF"):
        try:
            snes.setUseMF(True)
            return True
        except Exception:
            pass

    # Fallback: use Options
    try:
        opts = PETSc.Options(prefix)
        try:
            # Some builds allow setting boolean options with None
            opts.setValue("snes_mf", None)
        except Exception:
            # Others require a string
            opts["snes_mf"] = "1"
        return True
    except Exception as exc:
        raise RuntimeError(
            "Requested SNES matrix-free Jacobian (mf) but cannot enable it "
            "on this PETSc/petsc4py build."
        ) from exc


def _create_mffd_mat(comm, nloc: int, N: int):
    """
    Helper (currently unused) to create a Mat of type MFFD explicitly.
    Kept for possible future extensions.
    """
    PETSc = _get_petsc()
    try:
        J = PETSc.Mat().create(comm=comm)
        J.setType("mffd")
    except Exception:
        try:
            J = PETSc.Mat().create(comm=comm)
            J.setType(PETSc.Mat.Type.MFFD)
        except Exception as exc:
            raise RuntimeError("Failed to create MFFD matrix for mf Jacobian.") from exc
    J.setSizes(((nloc, N), (nloc, N)))
    J.setUp()
    return J


def _cfg_get(obj, name: str, default=None):
    """
    Safe getattr / dict-get helper for config-like objects.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _finalize_ksp_config(ksp, diag_pc, *, from_options: bool = True) -> None:
    """
    Apply PETSc options and structured PC tweaks, with safety guards.
    """
    if from_options:
        try:
            ksp.setFromOptions()
        except Exception:
            pass
    try:
        ksp.setUp()
    except Exception:
        pass
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)
    try:
        ksp.setUp()
    except Exception:
        pass


def _snes_set_jacobian_compat(snes, J, P, jac_func) -> None:
    """
    Wrapper around SNES.setJacobian to handle petsc4py API variations.

    Tries several common signatures:
      1) snes.setJacobian(J, P, func)
      2) snes.setJacobian(J, func)       # for older 2-arg API
      3) snes.setJacobian(func, J, P)
      4) snes.setJacobian(J=J, P=P, func=func)

    Raises TypeError if all forms fail.
    """
    errors: List[Exception] = []
    # (J, P, func)
    try:
        snes.setJacobian(J, P, jac_func)
        return
    except TypeError as exc:
        errors.append(exc)

    # (J, func) -- some older versions
    try:
        snes.setJacobian(J, jac_func)
        return
    except TypeError as exc:
        errors.append(exc)

    # (func, J, P)
    try:
        snes.setJacobian(jac_func, J, P)
        return
    except TypeError as exc:
        errors.append(exc)

    # Keyword form
    try:
        snes.setJacobian(J=J, P=P, func=jac_func)
        return
    except TypeError as exc:
        errors.append(exc)

    raise TypeError(
        "SNES.setJacobian signature mismatch for this petsc4py build; "
        f"tried (J,P,func), (J,func), (func,J,P), and keyword(J=,P=,func=). "
        f"errors={errors}"
    )


def _apply_linesearch_options(PETSc, prefix: str, linesearch_type: str) -> str:
    """
    Inject snes_linesearch_type into PETSc.Options to avoid SNES.getLineSearch().
    """
    desired = (linesearch_type or "").strip()
    if not desired:
        return ""
    try:
        opts = PETSc.Options(prefix)
        try:
            current = opts.getString("snes_linesearch_type", default="")
        except Exception:
            current = ""
        if not current:
            try:
                opts.setValue("snes_linesearch_type", desired)
            except Exception:
                opts["snes_linesearch_type"] = desired
        return current or desired
    except Exception:
        return desired


def solve_nonlinear_petsc(
    ctx: NonlinearContext,
    u0: np.ndarray,
) -> NonlinearSolveResult:
    """
    Solve F(u) = 0 using PETSc SNES with optional scaling and several Jacobian modes.
    """
    PETSc = _get_petsc()
    world = PETSc.COMM_WORLD
    world_size = world.getSize()
    world_rank = world.getRank()

    cfg = ctx.cfg
    petsc_cfg = cfg.petsc

    LinearSolverConfigTyped.from_cfg(cfg)

    # ------------------------------------------------------------------
    # MPI mode selection
    # ------------------------------------------------------------------
    mpi_mode = "world"
    try:
        mpi_mode = str(getattr(petsc_cfg, "mpi_mode", "world")).strip().lower()
    except Exception:
        mpi_mode = "world"

    comm = world
    serial_emulation = False   # "world but only rank 0 actually solves"
    do_solve = True

    if world_size > 1:
        if mpi_mode in ("self", "serial"):
            # Emulate serial SNES on rank 0 only; other ranks skip solve.
            serial_emulation = True
            do_solve = world_rank == 0
            comm = PETSc.COMM_SELF
            if world_rank == 0:
                logger.info("MPI mode '%s': using COMM_SELF on rank 0 only.", mpi_mode)
        elif mpi_mode == "redundant":
            # Every rank runs an independent serial SNES with COMM_SELF.
            comm = PETSc.COMM_SELF
            if world_rank == 0:
                logger.info("MPI mode '%s': using COMM_SELF on all ranks.", mpi_mode)
        else:
            # Normal fully-distributed mode
            comm = world

    # ------------------------------------------------------------------
    # DM / layout metadata (for MPI mode)
    # ------------------------------------------------------------------
    meta = getattr(ctx, "meta", {}) or {}
    dm = meta.get("dm", None)
    if dm is None:
        dm = getattr(ctx, "dm", None)

    dm_mgr = meta.get("dm_manager", None) or meta.get("dm_mgr", None) or getattr(ctx, "dm_manager", None)
    ld = meta.get("layout_dist", None) or meta.get("ld", None) or getattr(ctx, "layout_dist", None)

    parallel_active = comm.getSize() > 1
    if world_size > 1 and not parallel_active:
        # We are emulating serial (COMM_SELF) on some ranks; DM for world is not relevant.
        dm = None
        dm_mgr = None
        ld = None

    if parallel_active:
        if dm_mgr is None or ld is None:
            raise RuntimeError(
                "Parallel SNES requires dm_manager and layout_dist; "
                "ctx.meta['dm_manager'] / ['layout_dist'] are missing."
            )
        if dm is None and dm_mgr is not None:
            dm = getattr(dm_mgr, "dm", None)
        if dm is None:
            raise RuntimeError("Parallel SNES requires DM (DMComposite). ctx.meta['dm'] or ctx.dm is missing.")

    # ------------------------------------------------------------------
    # Nonlinear solver configuration
    # ------------------------------------------------------------------
    nl = getattr(cfg, "nonlinear", None)
    if nl is None or not getattr(nl, "enabled", False):
        raise ValueError("Nonlinear solver requested but cfg.nonlinear.enabled is False or missing.")

    max_outer_iter = int(getattr(nl, "max_outer_iter", 20))
    f_rtol = float(getattr(nl, "f_rtol", 1.0e-6))
    f_atol = float(getattr(nl, "f_atol", 1.0e-10))
    use_scaled_u = bool(getattr(nl, "use_scaled_unknowns", True))
    use_scaled_res = bool(getattr(nl, "use_scaled_residual", True))
    residual_scale_floor = float(getattr(nl, "residual_scale_floor", 1.0e-12))
    verbose = bool(getattr(nl, "verbose", False))
    log_every = int(getattr(nl, "log_every", 5))
    if log_every < 1:
        log_every = 1

    # Stage 5.4A rule: in true parallel mode, we disable scaling for now
    if parallel_active:
        if use_scaled_u:
            logger.warning("Disable use_scaled_unknowns in parallel for Stage 5.4A.")
            use_scaled_u = False
        if use_scaled_res:
            logger.warning("Disable use_scaled_residual in parallel for Stage 5.4A.")
            use_scaled_res = False

    # ------------------------------------------------------------------
    # PETSc / SNES configuration
    # ------------------------------------------------------------------
    prefix = getattr(petsc_cfg, "options_prefix", "")
    if prefix is None:
        prefix = ""
    prefix = str(prefix)
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    snes_type = str(getattr(petsc_cfg, "snes_type", "newtonls"))
    linesearch_type = str(getattr(petsc_cfg, "linesearch_type", "bt"))
    jacobian_mode_raw = getattr(petsc_cfg, "jacobian_mode", JacobianMode.FD)
    jacobian_mode_cfg = JacobianMode.normalize(jacobian_mode_raw)
    jacobian_mode = jacobian_mode_cfg

    snes_monitor = bool(getattr(petsc_cfg, "snes_monitor", False))

    # Serial mapping: allow jacobian_mode="mf" to use mfpc_aija implementation
    if (not parallel_active) and jacobian_mode_cfg == JacobianMode.MF:
        jacobian_mode = JacobianMode.MFPC_AIJA

    # Stage 5.4A: MPI mode only supports pure matrix-free operator
    if parallel_active and jacobian_mode_cfg != JacobianMode.MF:
        raise NotImplementedError("Parallel SNES currently supports jacobian_mode='mf' only (Stage 5.4A).")

    # Linear solver / PC config
    ksp_type = str(getattr(petsc_cfg, "ksp_type", "gmres"))
    linear_cfg = getattr(getattr(cfg, "solver", None), "linear", None)
    pc_type = _cfg_get(linear_cfg, "pc_type", None)
    if pc_type is None:
        pc_type = str(getattr(petsc_cfg, "pc_type", "ilu"))
    else:
        pc_type = str(pc_type)
    pc_type = _normalize_pc_type(pc_type) or "ilu"
    pc_overridden = False
    pc_overridden_dense = False

    # In parallel + (non-mf) we would override to ASM, but for mf we will override below.
    if parallel_active and pc_type in ("lu", "ilu"):
        # This path is not active for jacobian_mode="mf" due to check above,
        # but keep it for safety if future stages relax the constraint.
        try:
            opts = PETSc.Options(prefix)
            try:
                opts.setValue("pc_type", "asm")
                opts.setValue("pc_asm_overlap", "1")
                opts.setValue("sub_pc_type", "ilu")
                opts.setValue("sub_pc_factor_levels", "0")
                opts.setValue("sub_ksp_type", "preonly")
            except Exception:
                opts["pc_type"] = "asm"
                opts["pc_asm_overlap"] = "1"
                opts["sub_pc_type"] = "ilu"
                opts["sub_pc_factor_levels"] = "0"
                opts["sub_ksp_type"] = "preonly"
        except Exception:
            pass
        pc_type = "asm"
        pc_overridden = True

    petsc_rtol = float(getattr(petsc_cfg, "rtol", 1e-8))
    petsc_atol = float(getattr(petsc_cfg, "atol", 1e-12))
    petsc_max_it = int(getattr(petsc_cfg, "max_it", 200))
    restart = int(getattr(petsc_cfg, "restart", 30))
    precond_drop_tol = float(getattr(petsc_cfg, "precond_drop_tol", 0.0))
    if precond_drop_tol < 0.0:
        precond_drop_tol = 0.0
    precond_max_nnz_row = getattr(petsc_cfg, "precond_max_nnz_row", None)
    if precond_max_nnz_row is not None:
        try:
            precond_max_nnz_row = int(precond_max_nnz_row)
        except Exception:
            precond_max_nnz_row = None

    # ------------------------------------------------------------------
    # Unknown scaling
    # ------------------------------------------------------------------
    u0 = np.asarray(u0, dtype=np.float64)
    scale = np.asarray(ctx.scale_u, dtype=np.float64)
    if u0.shape != scale.shape:
        raise ValueError(f"u0 shape {u0.shape} incompatible with scale_u {scale.shape}")
    scale_safe = np.where(scale > 0.0, scale, 1.0)

    if use_scaled_u:
        x0 = u0 / scale_safe
    else:
        x0 = u0

    # Residual scaling
    scale_F = np.ones_like(u0, dtype=np.float64)
    if use_scaled_res:
        try:
            res_ref = residual_only(u0, ctx)
            if res_ref.shape != u0.shape:
                raise ValueError(
                    f"residual_only shape {res_ref.shape} incompatible with unknown shape {u0.shape}"
                )

            abs_ref = np.abs(res_ref)
            finite_all = abs_ref[np.isfinite(abs_ref)]
            if finite_all.size == 0:
                logger.warning("residual_only returned no finite entries; disabling use_scaled_residual.")
                use_scaled_res = False
            else:
                global_max = float(finite_all.max())
                if global_max <= 0.0:
                    use_scaled_res = False
                else:
                    block_rel_threshold = 1.0e-3
                    scale_F = np.ones_like(abs_ref, dtype=np.float64)
                    blocks = getattr(ctx.layout, "blocks", {}) or {}
                    for _, sl in blocks.items():
                        blk = abs_ref[sl]
                        finite = blk[np.isfinite(blk)]
                        if finite.size == 0:
                            continue
                        s_blk = float(np.percentile(finite, 90.0))
                        if not np.isfinite(s_blk) or s_blk <= 0.0:
                            continue
                        if s_blk < block_rel_threshold * global_max:
                            continue
                        if s_blk < residual_scale_floor:
                            s_blk = residual_scale_floor
                        scale_F[sl] = s_blk

                    bad_mask = ~(np.isfinite(scale_F) & (scale_F > 0.0))
                    if np.any(bad_mask):
                        scale_F[bad_mask] = 1.0

                ctx.meta["residual_scale_F"] = scale_F.copy()
        except Exception as exc:
            logger.warning(
                "Failed to build residual scaling; disabling use_scaled_residual. err=%s",
                exc,
            )
            use_scaled_res = False
            scale_F = np.ones_like(u0, dtype=np.float64)

    scale_F_safe = np.where(scale_F > 0.0, scale_F, 1.0)

    # Helpers to convert between "evaluation space" and physical unknowns
    def _x_to_u_phys(x_arr: np.ndarray) -> np.ndarray:
        return x_arr * scale_safe if use_scaled_u else x_arr

    def _res_phys_to_res_eval(res_phys: np.ndarray) -> np.ndarray:
        return res_phys / scale_F_safe if use_scaled_res else res_phys

    # ------------------------------------------------------------------
    # Matrix-free Jacobian context (used for mfpc_* modes, not for SNES-level MFFD)
    # ------------------------------------------------------------------
    class _MFJacShellCtx:
        """
        Simple matrix-free J·v implementation in terms of residual_only(ctx).

        This is used for mfpc_* modes in serial, not for pure 'mf' where we
        rely on SNES's built-in matrix-free support.
        """

        def __init__(
            self,
            *,
            ctx_ref: NonlinearContext,
            scale_u: np.ndarray,
            scale_F: np.ndarray,
            use_scaled_u: bool,
            use_scaled_res: bool,
            fd_eps: float,
        ) -> None:
            self.ctx = ctx_ref
            self.scale_u = np.asarray(scale_u, dtype=np.float64)
            self.scale_F = np.asarray(scale_F, dtype=np.float64)
            self.use_scaled_u = bool(use_scaled_u)
            self.use_scaled_res = bool(use_scaled_res)
            self.fd_eps = float(fd_eps)

            self.x0: np.ndarray | None = None
            self.F0: np.ndarray | None = None
            self.n_mf_mult = 0
            self.n_mf_func_eval = 0
            self.time_mf_func = 0.0

        def _x_to_u_phys(self, x: np.ndarray) -> np.ndarray:
            return x * self.scale_u if self.use_scaled_u else x

        def _F_phys_to_eval(self, F_phys: np.ndarray) -> np.ndarray:
            return F_phys / self.scale_F if self.use_scaled_res else F_phys

        def set_base(self, x0: np.ndarray) -> None:
            """
            Cache base point x0 and F(x0) for subsequent MatMult calls.
            """
            x0 = np.asarray(x0, dtype=np.float64)
            self.x0 = x0.copy()
            u0 = self._x_to_u_phys(self.x0)
            t0 = time.perf_counter()
            F0_phys = residual_only(u0, self.ctx)
            if not np.all(np.isfinite(F0_phys)):
                F0_phys = np.where(np.isfinite(F0_phys), F0_phys, 1.0e20)
            self.n_mf_func_eval += 1
            self.time_mf_func += (time.perf_counter() - t0)
            self.F0 = self._F_phys_to_eval(F0_phys)

        def mult(self, mat, X, Y):
            """
            Y = J·X ≈ [F(x0 + h X) - F(x0)] / h, in evaluation space.
            """
            try:
                v_view = X.getArray(readonly=True)
            except TypeError:
                v_view = X.getArray()
            v = np.asarray(v_view, dtype=np.float64)

            if self.x0 is None or self.F0 is None:
                # Fallback: base at zero if not set (should not normally happen).
                self.set_base(np.zeros_like(v))

            self.n_mf_mult += 1

            nv = float(np.linalg.norm(v))
            y_view = Y.getArray()
            if nv == 0.0:
                y_view[:] = 0.0
                return

            nx = float(np.linalg.norm(self.x0))
            h = self.fd_eps * (1.0 + nx) / nv
            if h == 0.0:
                h = self.fd_eps

            x1 = self.x0 + h * v
            u1 = self._x_to_u_phys(x1)
            t0 = time.perf_counter()
            F1_phys = residual_only(u1, self.ctx)
            if not np.all(np.isfinite(F1_phys)):
                F1_phys = np.where(np.isfinite(F1_phys), F1_phys, 1.0e20)
            self.n_mf_func_eval += 1
            self.time_mf_func += (time.perf_counter() - t0)
            F1 = self._F_phys_to_eval(F1_phys)
            y_view[:] = (F1 - self.F0) / h

    # ------------------------------------------------------------------
    # MPI residual helpers (DM-based) – used only in parallel mode
    # ------------------------------------------------------------------
    ctx_local = None
    if parallel_active:
        from assembly.residual_local import ResidualLocalCtx, pack_local_to_layout, scatter_layout_to_local
        from parallel.dm_manager import global_to_local, local_state_to_global

        ctx_local = ResidualLocalCtx(layout=ctx.layout, ld=ld)

        def _layout_to_dm_vec(u_layout: np.ndarray):
            """
            Scatter global layout vector (on this rank) into DM local vectors,
            then assemble a DM global Vec.
            """
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
            """
            Gather DM global Vec back into layout-style full vector (all ranks
            allreduce to get full state).
            """
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

        def _compute_residual_into_F(X, F) -> float:
            """
            Compute residual via residual_petsc directly into DM global Vec F.
            """
            from assembly.residual_global import residual_petsc

            residual_petsc(dm_mgr, ld, ctx, X, Fg=F)
            try:
                return float(F.norm(PETSc.NormType.NORM_INFINITY))
            except Exception:
                return float(F.norm())

    # ------------------------------------------------------------------
    # SNES function F(X)
    # ------------------------------------------------------------------
    history_inf_phys: List[float] = []
    history_inf_eval: List[float] = []
    last_inf_phys = {"val": np.nan}
    last_inf_eval = {"val": np.nan}
    t_solve0 = time.perf_counter()
    ctr = {
        "n_func_eval": 0,
        "n_jac_eval": 0,
        "ksp_its_total": 0,
    }
    tim = {
        "time_func": 0.0,
        "time_jac": 0.0,
        "time_linear_total": 0.0,
    }

    def snes_func(snes, X, F):
        """
        SNES function callback: F_eval(X) in evaluation space.
        """
        t0 = time.perf_counter()
        if parallel_active:
            # Direct DM-based residual
            res_inf_eval = float(_compute_residual_into_F(X, F))
            last_inf_eval["val"] = res_inf_eval
            last_inf_phys["val"] = res_inf_eval
        else:
            # Serial: use residual_only + scaling
            try:
                x_view = X.getArray(readonly=True)
            except TypeError:
                x_view = X.getArray()
            u_phys = _x_to_u_phys(np.asarray(x_view, dtype=np.float64))

            res_phys = residual_only(u_phys, ctx)
            if not np.all(np.isfinite(res_phys)):
                res_phys = np.where(np.isfinite(res_phys), res_phys, 1.0e20)
            res_eval = _res_phys_to_res_eval(res_phys)

            f_view = F.getArray()
            f_view[:] = res_eval

            last_inf_phys["val"] = float(np.linalg.norm(res_phys, ord=np.inf))
            last_inf_eval["val"] = float(np.linalg.norm(res_eval, ord=np.inf))
        ctr["n_func_eval"] += 1
        tim["time_func"] += (time.perf_counter() - t0)

    # Monitor for SNES iterations (outer)
    def snes_monitor_fn(snes, its, fnorm):
        vp = float(last_inf_phys["val"])
        ve = float(last_inf_eval["val"])
        if np.isfinite(vp):
            history_inf_phys.append(vp)
        if np.isfinite(ve):
            history_inf_eval.append(ve)
        if snes_monitor or verbose:
            if (its + 1) % log_every == 0:
                logger.info(
                    "snes iter=%d fnorm=%.3e res_inf_eval=%.3e res_inf_phys=%.3e",
                    its + 1,
                    float(fnorm),
                    float(last_inf_eval["val"]),
                    float(last_inf_phys["val"]),
                )

    # ------------------------------------------------------------------
    # SNES / KSP / PC creation
    # ------------------------------------------------------------------
    N = int(u0.size)
    snes = PETSc.SNES().create(comm=comm)
    snes.setOptionsPrefix(prefix)

    if dm is not None:
        try:
            snes.setDM(dm)
        except Exception:
            pass

    # Create residual Vec
    if parallel_active and dm is not None:
        F = dm.createGlobalVec()
    else:
        F = PETSc.Vec().createSeq(N, comm=comm)

    # Set SNES function
    try:
        snes.setFunction(snes_func, F)
    except TypeError:
        # Some petsc4py versions use reversed argument order
        snes.setFunction(F, snes_func)

    # SNES type & tolerances
    try:
        snes.setType(snes_type)
    except Exception:
        logger.warning("Unknown snes_type='%s', falling back to newtonls", snes_type)
        snes.setType("newtonls")

    snes.setTolerances(rtol=f_rtol, atol=f_atol, max_it=max_outer_iter)

    linesearch_type_eff = _apply_linesearch_options(PETSc, prefix, linesearch_type)

    # ------------------------------------------------------------------
    # Jacobian selection
    # ------------------------------------------------------------------
    precond_diag: Dict[str, Any] = {}
    mf_ctx: _MFJacShellCtx | None = None
    J = None
    P = None
    snes_mf_enabled = (jacobian_mode_cfg == JacobianMode.MF)

    if jacobian_mode == JacobianMode.MFPC_SPARSE_FD:
        # Matrix-free operator (Python shell) + sparse FD preconditioner
        fd_eps = float(getattr(petsc_cfg, "fd_eps", 1.0e-8))
        mf_ctx = _MFJacShellCtx(
            ctx_ref=ctx,
            scale_u=scale_safe,
            scale_F=scale_F_safe,
            use_scaled_u=use_scaled_u,
            use_scaled_res=use_scaled_res,
            fd_eps=fd_eps,
        )
        try:
            J = PETSc.Mat().createPython([N, N], comm=comm)
        except Exception:
            J = PETSc.Mat().createShell([N, N], comm=comm)
        J.setPythonContext(mf_ctx)
        J.setUp()

        pattern = ctx.meta.get("jac_pattern")
        if pattern is None:
            pattern = build_jacobian_pattern(cfg, ctx.grid_ref, ctx.layout)
            ctx.meta["jac_pattern"] = pattern

        P, precond_diag = build_sparse_fd_jacobian(
            ctx,
            x0,
            eps=fd_eps,
            drop_tol=precond_drop_tol,
            pattern=pattern,
        )

        def jac_func(snes_obj, X_vec, J_mat, P_mat):
            t0 = time.perf_counter()
            ctr["n_jac_eval"] += 1
            try:
                x0_view = X_vec.getArray(readonly=True)
            except TypeError:
                x0_view = X_vec.getArray()
            x0_arr = np.asarray(x0_view, dtype=np.float64)
            if mf_ctx is not None:
                mf_ctx.set_base(x0_arr)

            _P, diagP = build_sparse_fd_jacobian(
                ctx,
                x0_arr,
                eps=fd_eps,
                drop_tol=precond_drop_tol,
                pattern=pattern,
                mat=P_mat,
            )
            precond_diag.update(diagP)
            tim["time_jac"] += (time.perf_counter() - t0)
            return True

        _snes_set_jacobian_compat(snes, J, P, jac_func)

    elif jacobian_mode == JacobianMode.MFPC_AIJA:
        # Matrix-free operator (Python shell) + Aij preconditioner
        fd_eps = float(getattr(petsc_cfg, "fd_eps", 1.0e-8))
        mf_ctx = _MFJacShellCtx(
            ctx_ref=ctx,
            scale_u=scale_safe,
            scale_F=scale_F_safe,
            use_scaled_u=use_scaled_u,
            use_scaled_res=use_scaled_res,
            fd_eps=fd_eps,
        )
        try:
            J = PETSc.Mat().createPython([N, N], comm=comm)
        except Exception:
            J = PETSc.Mat().createShell([N, N], comm=comm)
        J.setPythonContext(mf_ctx)
        J.setUp()

        # Build an AIJ preconditioner matrix once, at x0
        u0_phys = _x_to_u_phys(np.asarray(x0, dtype=np.float64))
        P, precond_diag = build_precond_mat_aij_from_A(
            ctx,
            u_phys=u0_phys,
            drop_tol=precond_drop_tol,
            comm=comm,
            max_nnz_row=precond_max_nnz_row,
        )

        # Optional left/right scaling for preconditioner (if we use scaled spaces)
        apply_scale = use_scaled_res or use_scaled_u
        L_vec = None
        R_vec = None
        if apply_scale:
            L_np = (1.0 / scale_F_safe) if use_scaled_res else np.ones_like(scale_F_safe)
            R_np = scale_safe if use_scaled_u else np.ones_like(scale_safe)
            idx = np.arange(N, dtype=PETSc.IntType)
            L_vec = PETSc.Vec().createSeq(N, comm=comm)
            R_vec = PETSc.Vec().createSeq(N, comm=comm)
            L_vec.setValues(idx, L_np.astype(PETSc.ScalarType, copy=False))
            R_vec.setValues(idx, R_np.astype(PETSc.ScalarType, copy=False))
            L_vec.assemblyBegin()
            L_vec.assemblyEnd()
            R_vec.assemblyBegin()
            R_vec.assemblyEnd()
            P.diagonalScale(L_vec, R_vec)

        def jac_func(snes_obj, X_vec, J_mat, P_mat):
            t0 = time.perf_counter()
            ctr["n_jac_eval"] += 1
            try:
                x0_view = X_vec.getArray(readonly=True)
            except TypeError:
                x0_view = X_vec.getArray()
            x0_arr = np.asarray(x0_view, dtype=np.float64)
            if mf_ctx is not None:
                mf_ctx.set_base(x0_arr)
            u_phys = _x_to_u_phys(x0_arr)

            diagP = fill_precond_mat_aij_from_A(
                P_mat,
                ctx,
                u_phys,
                drop_tol=precond_drop_tol,
            )
            precond_diag.update(diagP)
            if L_vec is not None and R_vec is not None:
                P_mat.diagonalScale(L_vec, R_vec)
            tim["time_jac"] += (time.perf_counter() - t0)
            return True

        _snes_set_jacobian_compat(snes, J, P, jac_func)

    elif jacobian_mode == JacobianMode.MF:
        # 纯 SNES 级 matrix-free Jacobian（MFFD）：
        # - J 由 SNES 内部创建 / 管理；
        # - 我们只负责构造 AIJ 预条件矩阵 P；
        # - 串行用 A(u) -> AIJ 预条件矩阵；
        # - 并行用简单位对角 mpiaij 预条件矩阵。
        try:
            snes_mf_enabled = _enable_snes_matrix_free(snes, prefix)
        except Exception as exc:
            raise RuntimeError(
                "jacobian_mode='mf' requires SNES matrix-free support, "
                "but _enable_snes_matrix_free() failed on this PETSc/petsc4py build."
            ) from exc

        # ------------------------------------------------------------------
        # 1) 构建用于 PC 的 AIJ 稀疏矩阵 P
        # ------------------------------------------------------------------
        if parallel_active:
            # 并行：分布式 mpiaij 单位矩阵（每行一个对角 1）
            P = PETSc.Mat().create(comm=comm)
            try:
                nloc = int(F.getLocalSize())
            except Exception:
                nloc = None
            if nloc is None and dm is not None:
                try:
                    nloc = int(dm.createGlobalVec().getLocalSize())
                except Exception:
                    nloc = None
            if nloc is None:
                raise RuntimeError("Unable to determine local size for mf preconditioner.")
            P.setSizes(((nloc, N), (nloc, N)))
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

            Istart, Iend = P.getOwnershipRange()
            for row in range(Istart, Iend):
                P.setValue(row, row, 1.0)

            P.assemblyBegin()
            P.assemblyEnd()

            precond_diag_local: Dict[str, Any] = {}
        else:
            # 串行：沿用 Stage 2 的 A(u) -> AIJ 预条件构造
            u0_phys = _x_to_u_phys(np.asarray(x0, dtype=np.float64))
            P, precond_diag_local = build_precond_mat_aij_from_A(
                ctx,
                u_phys=u0_phys,
                drop_tol=precond_drop_tol,
                comm=comm,
                max_nnz_row=precond_max_nnz_row,
            )

        precond_diag.update(precond_diag_local)

        # 串行下如果开启了缩放，可以对 P 做一次左 / 右对角缩放；
        # 并行模式下前面已经强制关闭 use_scaled_u / use_scaled_res，这里不会进入。
        apply_scale = (not parallel_active) and (use_scaled_res or use_scaled_u)
        if apply_scale:
            L_np = (1.0 / scale_F_safe) if use_scaled_res else np.ones_like(scale_F_safe)
            R_np = scale_safe if use_scaled_u else np.ones_like(scale_safe)
            idx = np.arange(N, dtype=PETSc.IntType)

            L_vec = PETSc.Vec().createSeq(N, comm=comm)
            R_vec = PETSc.Vec().createSeq(N, comm=comm)
            L_vec.setValues(idx, L_np.astype(PETSc.ScalarType, copy=False))
            R_vec.setValues(idx, R_np.astype(PETSc.ScalarType, copy=False))
            L_vec.assemblyBegin()
            L_vec.assemblyEnd()
            R_vec.assemblyBegin()
            R_vec.assemblyEnd()

            P.diagonalScale(L_vec, R_vec)

        # ------------------------------------------------------------------
        # 2) 把 J 交给 SNES 内部的 MFFD 管理，我们只提供 P
        # ------------------------------------------------------------------
        J = None
        try:
            # 新版 petsc4py 常见签名
            snes.setJacobian(J=J, P=P, func=None)
        except TypeError:
            # 兼容旧版位置参数 / 不同关键字形式
            try:
                snes.setJacobian(J, P, None)
            except TypeError:
                try:
                    snes.setJacobian(J, P)
                except TypeError:
                    # 最后再试只提供 P
                    snes.setJacobian(P=P)

        # 并行时不允许 global LU/ILU，保留你原有的 ASM 覆盖逻辑
        if parallel_active and pc_type in ("lu", "ilu"):
            try:
                opts = PETSc.Options(prefix)
                try:
                    opts.setValue("pc_type", "asm")
                    opts.setValue("pc_asm_overlap", "1")
                    opts.setValue("sub_pc_type", "ilu")
                    opts.setValue("sub_pc_factor_levels", "0")
                    opts.setValue("sub_ksp_type", "preonly")
                except Exception:
                    opts["pc_type"] = "asm"
                    opts["pc_asm_overlap"] = "1"
                    opts["sub_pc_type"] = "ilu"
                    opts["sub_pc_factor_levels"] = "0"
                    opts["sub_ksp_type"] = "preonly"
            except Exception:
                pass
            pc_type = "asm"
            pc_overridden = True


    else:
        # Default: dense FD Jacobian (serial-oriented).
        J = PETSc.Mat().createDense([N, N], comm=comm)
        J.setUp()
        P = J
        if pc_type in ("ilu", "jacobi", ""):
            pc_type = "lu"
            pc_overridden_dense = True

        fd_eps = float(getattr(petsc_cfg, "fd_eps", 1.0e-8))

        def jac_func(snes_obj, X_vec, J_mat, P_mat):
            t0 = time.perf_counter()
            ctr["n_jac_eval"] += 1
            try:
                x0_view = X_vec.getArray(readonly=True)
            except TypeError:
                x0_view = X_vec.getArray()
            x0_arr = np.asarray(x0_view, dtype=np.float64)
            nloc = x0_arr.size

            u0_phys = _x_to_u_phys(x0_arr)
            r0_phys = residual_only(u0_phys, ctx)
            r0_eval = _res_phys_to_res_eval(r0_phys)

            J_mat.zeroEntries()
            rows = np.arange(nloc, dtype=PETSc.IntType)
            x_work = x0_arr.copy()

            for j in range(nloc):
                dx = fd_eps * (1.0 + abs(x0_arr[j]))
                if dx == 0.0:
                    dx = fd_eps
                x_work[j] = x0_arr[j] + dx

                uj_phys = _x_to_u_phys(x_work)
                rj_phys = residual_only(uj_phys, ctx)
                rj_eval = _res_phys_to_res_eval(rj_phys)

                col = np.asarray((rj_eval - r0_eval) / dx, dtype=PETSc.ScalarType)
                cols = np.array([j], dtype=PETSc.IntType)
                J_mat.setValues(rows, cols, col.reshape(-1, 1), addv=False)

                x_work[j] = x0_arr[j]

            J_mat.assemblyBegin()
            J_mat.assemblyEnd()

            if P_mat is not J_mat:
                P_mat.assemblyBegin()
                P_mat.assemblyEnd()
            tim["time_jac"] += (time.perf_counter() - t0)
            return True

        _snes_set_jacobian_compat(snes, J, J, jac_func)

    # ------------------------------------------------------------------
    # KSP / PC setup
    # ------------------------------------------------------------------
    ksp = snes.getKSP()
    Aop = J
    Pop = P if P is not None else J
    if Aop is not None and Pop is not None:
        ksp.setOperators(Aop, Pop)

    # Honor options, but we already set some types explicitly
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
    try:
        ksp.setFromOptions()
    except Exception:
        pass

    # KSP type
    try:
        ksp.setType(ksp_type)
    except Exception:
        logger.warning("Unknown ksp_type='%s', falling back to gmres", ksp_type)
        ksp.setType("gmres")

    # PC type
    pc = ksp.getPC()
    if str(pc_type).lower() != "fieldsplit":
        try:
            pc.setType(pc_type)
        except Exception:
            logger.warning("Unknown pc_type='%s', falling back to jacobi", pc_type)
            pc.setType("jacobi")

    # KSP tolerances
    ksp.setInitialGuessNonzero(False)
    ksp.setTolerances(rtol=petsc_rtol, atol=petsc_atol, max_it=petsc_max_it)

    # Restart params for GMRES-like solvers
    try:
        ksp_type_eff = str(ksp.getType()).lower()
        if ksp_type_eff in ("gmres", "fgmres"):
            ksp.setGMRESRestart(restart)
        elif ksp_type_eff == "lgmres":
            if hasattr(ksp, "setLGMRESRestart"):
                ksp.setLGMRESRestart(restart)
    except Exception:
        logger.debug("Unable to set restart for ksp_type='%s'", ksp.getType())

    # Apply structured PC (except pure mf in parallel mode)
    diag_pc: Dict[str, Any] = {}
    if not (parallel_active and jacobian_mode_cfg == JacobianMode.MF):
        Aop_call = None
        Pop_call = None
        try:
            Aop_call, Pop_call = ksp.getOperators()
        except Exception:
            Aop_call = None
            Pop_call = None
        if Aop_call is None:
            Aop_call = J
        if Pop_call is None:
            Pop_call = Pop
        diag_pc = apply_structured_pc(
            ksp=ksp,
            cfg=cfg,
            layout=ctx.layout,
            A=Aop_call,
            P=Pop_call,
            pc_type_override=_normalize_pc_type(pc_type),
        )

    # Attach SNES monitor
    snes.setMonitor(snes_monitor_fn)
    _finalize_ksp_config(ksp, diag_pc, from_options=False)

    # KSP monitor to accumulate total linear iterations & timing
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

    # ------------------------------------------------------------------
    # Initial X vector
    # ------------------------------------------------------------------
    if parallel_active and dm is not None:
        X = dm.createGlobalVec()
        # x0 is layout-based here; we need to map to DM global Vec
        X_init = _layout_to_dm_vec(np.asarray(x0, dtype=np.float64))
        try:
            X_init.copy(X)
        except Exception:
            X.set(0.0)
            X.axpy(1.0, X_init)
    else:
        X = PETSc.Vec().createSeq(N, comm=comm)
        X_arr = X.getArray()
        X_arr[:] = np.asarray(x0, dtype=np.float64)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    if do_solve:
        snes.solve(None, X)
        if ksp_t.get("in_solve", False):
            tim["time_linear_total"] += (time.perf_counter() - ksp_t["t0"])
            ksp_t["in_solve"] = False

        if parallel_active and dm is not None:
            u_final = _dm_vec_to_layout(X)
        else:
            x_final = np.asarray(X.getArray(), dtype=np.float64).copy()
            u_final = _x_to_u_phys(x_final)

        reason = int(snes.getConvergedReason())
        ksp_reason = int(ksp.getConvergedReason())
        ksp_it = int(ksp.getIterationNumber())
        converged = reason > 0
        n_iter = int(snes.getIterationNumber())

        res_final = residual_only(u_final, ctx)
        res_norm_2 = float(np.linalg.norm(res_final))
        res_norm_inf = float(np.linalg.norm(res_final, ord=np.inf))
    else:
        # Ranks that do not participate in solve
        u_final = np.asarray(u0, dtype=np.float64).copy()
        reason = 0
        ksp_reason = 0
        ksp_it = 0
        converged = False
        n_iter = 0
        res_norm_2 = float("nan")
        res_norm_inf = float("nan")

    # ------------------------------------------------------------------
    # Diagnostics: vector/matrix types, metadata
    # ------------------------------------------------------------------
    j_type = "none"
    p_type = "none"
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

    # If we are in pure 'mf' mode and could not inspect J, report "mffd"
    # so that tests can still see a matrix-free type.
    if jacobian_mode_cfg == JacobianMode.MF:
        if not j_type or j_type == "none":
            j_type = "mffd"
        elif all(tag not in j_type for tag in ("snesmf", "mffd", "matfree", "mf")):
            j_type = "mffd"

    x_type = ""
    f_type = ""
    try:
        x_type = str(X.getType())
        f_type = str(F.getType())
    except Exception:
        pass

    extra: Dict[str, Any] = {
        "snes_reason": reason,
        "snes_reason_str": str(reason),
        "snes_type": snes.getType(),
        "linesearch_type": linesearch_type_eff or linesearch_type,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "ksp_reason": ksp_reason,
        "ksp_reason_str": str(ksp_reason),
        "ksp_it": ksp_it,
        "snes_iter": n_iter,
        "jacobian_mode": jacobian_mode_cfg,
        "res_norm_inf_eval": float(last_inf_eval["val"]),
        "petsc_snes_reason": int(reason),
        "petsc_snes_reason_str": str(reason),
        "n_func_eval": int(ctr["n_func_eval"]),
        "n_jac_eval": int(ctr["n_jac_eval"]),
        "ksp_its_total": int(ctr["ksp_its_total"]),
        "time_func": float(tim["time_func"]),
        "time_jac": float(tim["time_jac"]),
        "time_linear_total": float(tim["time_linear_total"]),
        "time_total": float(time.perf_counter() - t_solve0),
    }

    try:
        extra["comm_size"] = int(comm.getSize())
    except Exception:
        pass
    if x_type:
        extra["X_vec_type"] = x_type
    if f_type:
        extra["F_vec_type"] = f_type

    J_eff = J
    P_eff = P
    try:
        if J_eff is None or P_eff is None:
            J_now, P_now = snes.getJacobian()
            if J_eff is None and J_now is not None:
                J_eff = J_now
            if P_eff is None and P_now is not None:
                P_eff = P_now
    except Exception:
        pass

    extra["J_mat_type"] = j_type
    extra["P_mat_type"] = p_type

    try:
        extra["X_local_size"] = int(X.getLocalSize())
        extra["X_ownership_range"] = tuple(int(v) for v in X.getOwnershipRange())
    except Exception:
        pass
    try:
        if dm is not None:
            extra["dm_type"] = str(dm.getType())
    except Exception:
        pass

    if diag_pc:
        extra["pc_structured"] = dict(diag_pc)
    if jacobian_mode in (JacobianMode.MFPC_AIJA, JacobianMode.MFPC_SPARSE_FD):
        extra["precond_drop_tol"] = float(precond_drop_tol)
        if precond_max_nnz_row is not None:
            extra["precond_max_nnz_row"] = int(precond_max_nnz_row)
        if precond_diag:
            extra["precond_diag"] = dict(precond_diag)
    if jacobian_mode_cfg == JacobianMode.MF:
        extra["snes_mf_enabled"] = bool(snes_mf_enabled)
        if x_type:
            extra["X_vec_type"] = x_type
        if f_type:
            extra["F_vec_type"] = f_type
        extra["J_mat_type"] = j_type
        extra["P_mat_type"] = p_type
        extra["pc_type_overridden"] = bool(pc_overridden)
    if mf_ctx is not None:
        extra["n_mf_mult"] = int(mf_ctx.n_mf_mult)
        extra["n_mf_func_eval"] = int(mf_ctx.n_mf_func_eval)
        extra["time_mf_func"] = float(mf_ctx.time_mf_func)
    if pc_overridden:
        extra["pc_type_overridden"] = True
    if pc_overridden_dense:
        extra["pc_type_overridden_dense"] = True

    if not history_inf_phys and np.isfinite(last_inf_phys["val"]):
        history_inf_phys.append(float(last_inf_phys["val"]))
    if not history_inf_eval and np.isfinite(last_inf_eval["val"]):
        history_inf_eval.append(float(last_inf_eval["val"]))
    if history_inf_phys:
        extra["history_res_inf_phys"] = list(history_inf_phys)
    if history_inf_eval:
        extra["history_res_inf_eval"] = list(history_inf_eval)

    message = None if converged else f"SNES diverged (reason={reason})"
    if not converged and do_solve:
        logger.warning("SNES not converged: reason=%d res_inf=%.3e", reason, res_norm_inf)

    diag_payload = {
        "converged": bool(converged),
        "method": f"snes:{snes.getType()}",
        "n_iter": int(n_iter),
        "res_norm_2": float(res_norm_2),
        "res_norm_inf": float(res_norm_inf),
        "history_res_inf": list(history_inf_phys),
        "message": message,
        "extra": dict(extra),
    }

    # ------------------------------------------------------------------
    # Serial emulation broadcast (world_size > 1 but only rank 0 solved)
    # ------------------------------------------------------------------
    if serial_emulation and world_size > 1:
        try:
            from mpi4py import MPI  # noqa: F401

            mpicomm = world.tompi4py()
            u_final = mpicomm.bcast(u_final if do_solve else None, root=0)
            diag_payload = mpicomm.bcast(diag_payload if do_solve else None, root=0)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("mpi4py is required for serial MPI emulation.") from exc

    # Build diagnostics object
    diag = NonlinearDiagnostics(
        converged=diag_payload["converged"],
        method=diag_payload["method"],
        n_iter=diag_payload["n_iter"],
        res_norm_2=diag_payload["res_norm_2"],
        res_norm_inf=diag_payload["res_norm_inf"],
        history_res_inf=diag_payload["history_res_inf"],
        message=diag_payload["message"],
        extra=diag_payload["extra"],
    )

    # Optional residual diagnostics in verbose mode
    if verbose and do_solve:
        try:
            _, diag_res = build_global_residual(u_final, ctx)
            diag.extra["assembly_diag"] = diag_res
        except Exception as exc:
            diag.extra["assembly_diag_error"] = str(exc)

    # Clean up PETSc objects to prevent memory leaks
    try:
        if 'X_init' in locals() and X_init is not None:
            X_init.destroy()
    except Exception:
        pass
    try:
        if 'X' in locals() and X is not None:
            X.destroy()
    except Exception:
        pass
    try:
        if 'F' in locals() and F is not None:
            F.destroy()
    except Exception:
        pass
    try:
        if 'J' in locals() and J is not None:
            J.destroy()
    except Exception:
        pass
    try:
        if 'P' in locals() and P is not None:
            P.destroy()
    except Exception:
        pass
    try:
        if '_P' in locals() and _P is not None:
            _P.destroy()
    except Exception:
        pass
    try:
        if 'snes' in locals() and snes is not None:
            snes.destroy()
    except Exception:
        pass

    return NonlinearSolveResult(u=np.asarray(u_final, dtype=np.float64), diag=diag)
