"""
SciPy-based nonlinear solver wrapper (global Newton/Krylov).

This module only coordinates solver calls; residual assembly is handled elsewhere.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy import optimize

from solvers.nonlinear_context import NonlinearContext
from assembly.residual_global import build_global_residual, residual_only
from solvers.linesearch_ts_guard import cap_ts_in_u, estimate_ts_hard
from solvers.nonlinear_types import NonlinearDiagnostics, NonlinearSolveResult

logger = logging.getLogger(__name__)


NewtonDiagnostics = NonlinearDiagnostics
NewtonSolveResult = NonlinearSolveResult


def solve_nonlinear_scipy(
    ctx: NonlinearContext,
    u0: np.ndarray,
) -> NonlinearSolveResult:
    """
    Solve F(u)=0 using SciPy nonlinear solvers with optional scaling.
    """
    cfg = ctx.cfg
    nl = getattr(cfg, "nonlinear", None)
    if nl is None or not getattr(nl, "enabled", False):
        raise ValueError("Nonlinear solver requested but cfg.nonlinear.enabled is False or missing.")

    solver = str(getattr(nl, "solver", "newton_krylov"))
    krylov_method = str(getattr(nl, "krylov_method", "lgmres"))
    max_outer_iter = int(getattr(nl, "max_outer_iter", 20))
    inner_maxiter = int(getattr(nl, "inner_maxiter", 20))
    f_rtol = float(getattr(nl, "f_rtol", 1.0e-6))
    f_atol = float(getattr(nl, "f_atol", 1.0e-10))
    use_scaled_u = bool(getattr(nl, "use_scaled_unknowns", True))
    use_scaled_res = bool(getattr(nl, "use_scaled_residual", True))
    residual_scale_floor = float(getattr(nl, "residual_scale_floor", 1.0e-12))
    verbose = bool(getattr(nl, "verbose", False))
    log_every = int(getattr(nl, "log_every", 5))
    if log_every < 1:
        log_every = 1
    ts_cap_enabled = bool(getattr(nl, "ts_linesearch_cap", True))
    ts_cap_alpha = float(getattr(nl, "ts_linesearch_alpha", 0.8))
    ts_upper_mode = str(getattr(nl, "ts_upper_mode", "tbub_last"))

    t_old = float(getattr(ctx, "t_old", 0.0))
    t_new = t_old + float(getattr(ctx, "dt", 0.0))

    if use_scaled_u:
        u0_s = ctx.to_scaled_u(u0)
    else:
        u0_s = np.asarray(u0, dtype=np.float64)

    history: List[float] = []

    # Residual scaling based on block-wise reference residual magnitudes.
    scale_F = np.ones_like(u0_s, dtype=np.float64)
    if use_scaled_res:
        try:
            if use_scaled_u:
                u_ref = ctx.from_scaled_u(u0_s)
            else:
                u_ref = np.asarray(u0_s, dtype=np.float64)

            res_ref = residual_only(u_ref, ctx)
            if res_ref.shape != u0_s.shape:
                raise ValueError(
                    f"residual_only shape {res_ref.shape} incompatible with unknown shape {u0_s.shape}"
                )

            abs_ref = np.abs(res_ref)
            finite_all = abs_ref[np.isfinite(abs_ref)]
            if finite_all.size == 0:
                logger.warning(
                    "residual_only returned no finite entries; disabling use_scaled_residual."
                )
                use_scaled_res = False
                scale_F = np.ones_like(u0_s, dtype=np.float64)
            else:
                global_max = float(finite_all.max())
                if global_max <= 0.0:
                    use_scaled_res = False
                    scale_F = np.ones_like(u0_s, dtype=np.float64)
                else:
                    block_rel_threshold = 1.0e-3
                    scale_F = np.ones_like(abs_ref, dtype=np.float64)

                    blocks = getattr(ctx.layout, "blocks", {}) or {}
                    for blk_name, sl in blocks.items():
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
            scale_F = np.ones_like(u0_s, dtype=np.float64)

    scale_F_safe = np.where(scale_F > 0.0, scale_F, 1.0)

    def _F_scaled(u_s: np.ndarray) -> np.ndarray:
        if use_scaled_u:
            u_phys = ctx.from_scaled_u(u_s)
        else:
            u_phys = u_s

        if ts_cap_enabled and ctx.layout.has_block("Ts"):
            ts_upper = estimate_ts_hard(ctx, mode=ts_upper_mode)
            ts_idx = int(ctx.layout.idx_Ts())
            ts_before = float(u_phys[ts_idx])
            if ts_upper is not None and np.isfinite(ts_upper) and np.isfinite(ts_before):
                ts_upper = ts_before + ts_cap_alpha * (float(ts_upper) - ts_before)
            u_phys, capped, ts_after = cap_ts_in_u(u_phys, idx_Ts=ts_idx, Ts_upper=ts_upper)
            if capped:
                meta = ctx.meta
                meta["n_lambda_cap_ts"] = int(meta.get("n_lambda_cap_ts", 0)) + 1
                if ts_upper is not None and np.isfinite(ts_upper):
                    delta = float(ts_before - float(ts_upper))
                    meta["max_Ts_trial_minus_upper"] = max(
                        float(meta.get("max_Ts_trial_minus_upper", -np.inf)), delta
                    )
                meta["ts_cap_last"] = {
                    "ts_before": ts_before,
                    "ts_after": ts_after,
                    "ts_upper": float(ts_upper) if ts_upper is not None else float("nan"),
                    "alpha": ts_cap_alpha,
                }

        res_raw, diag = build_global_residual(u_phys, ctx)
        res = res_raw
        if use_scaled_res:
            res = res / scale_F_safe

        res_norm_inf = float(np.linalg.norm(res, ord=np.inf))
        history.append(res_norm_inf)
        if verbose:
            iter_id = len(history)
            if iter_id % log_every == 0:
                arg = (diag or {}).get("residual_argmax", {})
                arg_name = str(arg.get("name") or "")
                if not arg_name:
                    arg_kind = str(arg.get("kind") or "")
                    arg_idx = arg.get("idx", "")
                    arg_name = f"{arg_kind}#{arg_idx}" if arg_kind else str(arg_idx)
                arg_abs = float(arg.get("abs", np.nan))

                clamp = (diag or {}).get("clamp", {})
                clamp_flags: List[str] = []
                if clamp.get("no_condensation_applied", False):
                    clamp_flags.append("no_cond")
                if clamp.get("deltaY_min_applied", False):
                    clamp_flags.append("deltaY_min")
                gas_oob = int(clamp.get("gas_closure_oob_any", 0))
                liq_oob = int(clamp.get("liq_closure_oob_any", 0))
                if gas_oob > 0:
                    clamp_flags.append(f"gas_oob={gas_oob}")
                if liq_oob > 0:
                    clamp_flags.append(f"liq_oob={liq_oob}")
                clamp_str = ",".join(clamp_flags) if clamp_flags else "none"

                eq_source = str((diag or {}).get("eq_result", {}).get("source", ""))
                props_source = str((diag or {}).get("props", {}).get("source", ""))
                raw_inf = float((diag or {}).get("residual_norm_inf", np.nan))
                if use_scaled_res:
                    res_msg = f"{res_norm_inf:.3e} raw={raw_inf:.3e}"
                else:
                    res_msg = f"{res_norm_inf:.3e}"
                logger.info(
                    "nonlinear iter=%d t=[%.6e -> %.6e] res_inf=%s argmax=%s(%.3e) clamp=%s eq=%s props=%s",
                    iter_id,
                    t_old,
                    t_new,
                    res_msg,
                    arg_name,
                    arg_abs,
                    clamp_str,
                    eq_source,
                    props_source,
                )
        return res

    converged = False
    msg = None

    if solver == "newton_krylov":
        try:
            sol_s = optimize.newton_krylov(
                _F_scaled,
                u0_s,
                method=krylov_method,
                inner_maxiter=inner_maxiter,
                maxiter=max_outer_iter,
                f_rtol=f_rtol,
                f_tol=f_atol,
                verbose=verbose,
            )
            converged = True
        except optimize.NoConvergence as exc:
            sol_raw = getattr(exc, "x", None)
            if sol_raw is None:
                sol_raw = exc.args[0] if exc.args else u0_s
            sol_s = np.asarray(sol_raw, dtype=np.float64)
            converged = False
            msg = str(exc)
            logger.warning("newton_krylov did not converge: %s", msg)
    elif solver in ("root_hybr", "hybr"):
        sol = optimize.root(
            _F_scaled,
            u0_s,
            method="hybr",
            tol=f_rtol,
            options={"maxfev": max_outer_iter},
        )
        sol_s = np.asarray(sol.x, dtype=np.float64)
        converged = bool(sol.success)
        msg = None if converged else str(sol.message)
        if not converged:
            logger.warning("root(hybr) did not converge: %s", msg)
    else:
        raise ValueError(f"Unknown cfg.nonlinear.solver={solver!r}")

    if use_scaled_u:
        u_final = ctx.from_scaled_u(sol_s)
    else:
        u_final = np.asarray(sol_s, dtype=np.float64)

    res_final = residual_only(u_final, ctx)
    res_norm_2 = float(np.linalg.norm(res_final))
    res_norm_inf = float(np.linalg.norm(res_final, ord=np.inf))
    n_iter = len(history)

    diag = NonlinearDiagnostics(
        converged=converged,
        method=solver,
        n_iter=n_iter,
        res_norm_2=res_norm_2,
        res_norm_inf=res_norm_inf,
        history_res_inf=history,
        message=msg,
        extra={
            "n_penalty_residual": int(ctx.meta.get("penalty_count", 0)),
            "penalty_last_reason": str((ctx.meta.get("penalty_last", {}) or {}).get("penalty_reason", "")),
            "n_lambda_cap_ts": int(ctx.meta.get("n_lambda_cap_ts", 0)),
            "max_Ts_trial_minus_upper": float(ctx.meta.get("max_Ts_trial_minus_upper", np.nan)),
        },
    )
    return NonlinearSolveResult(u=u_final, diag=diag)
