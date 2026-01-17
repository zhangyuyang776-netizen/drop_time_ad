"""
Single-step timestepper (linear backend dispatch) with coupled Tg/Tl/Yg/Ts/mpp/Rd solve.

Current scope:
- Assemble once via numpy/PETSc bridge build_transport_system (already embeds Yg via global layout).
- Use layout pack/apply to map between State and global unknown vector (no hand indexing).
- Per-step property recomputation enabled (props are treated as given for the current step,
  then recomputed from the updated state for the next step).
- No grid remeshing for Rd changes.

This module:
- Delegates assembly to numpy or PETSc bridge based on cfg.solver.linear.backend.
- Delegates linear solve to solvers.solver_linear.solve_linear_system.
- Packs/apply state via core.layout helpers.
- Computes interface equilibrium per-step to supply eq_result for mpp equation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.types import CaseConfig, Grid1D, Props, State
from core.layout import UnknownLayout, pack_state, apply_u_to_state
from core.simplex import project_Y_cellwise
from assembly.residual_global import build_global_residual
from assembly.build_system_SciPy import build_transport_system as build_transport_system_numpy
from assembly.build_system_petsc import (
    build_transport_system_petsc_bridge,
    build_transport_system_petsc_native,
)
from properties.equilibrium import build_equilibrium_model, compute_interface_equilibrium
from properties.compute_props import compute_props, get_or_build_models
from physics.interface_bc import EqResultLike
from solvers.nonlinear_context import build_nonlinear_context_for_step
from solvers.solver_nonlinear import solve_nonlinear
from solvers.solver_linear import solve_linear_system
from solvers.linear_types import LinearSolveResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StepDiagnostics:
    """Diagnostics for a single timestep."""

    # Time info
    t_old: float
    t_new: float
    dt: float

    # Linear solve info
    linear_converged: bool
    linear_method: str
    linear_n_iter: int
    linear_residual_norm: float
    linear_rel_residual: float

    # Key state values (new state)
    Ts: float
    Rd: float
    mpp: float
    Tg_min: float
    Tg_max: float

    # Nonlinear solve info (global Newton)
    nonlinear_converged: bool = False
    nonlinear_method: str = ""
    nonlinear_n_iter: int = 0
    nonlinear_residual_norm: float = float("nan")
    nonlinear_residual_inf: float = float("nan")

    # Interface/radius diagnostics
    energy_balance_if: Optional[float] = None
    mass_balance_rd: Optional[float] = None

    # Extra debug data
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    """Result of a single timestep."""

    state_new: State
    props_new: Props
    diag: StepDiagnostics
    success: bool
    message: Optional[str] = None


def advance_one_step(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state: State,
    props: Props,
    t: float,
) -> StepResult:
    """Advance one step using linear backend dispatch (SciPy/PETSc)."""
    return advance_one_step_scipy(cfg=cfg, grid=grid, layout=layout, state=state, props=props, t=t)


def advance_one_step_scipy(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state: State,
    props: Props,
    t: float,
) -> StepResult:
    """
    Advance coupled unknowns (Tg, Tl, Yg, Ts, mpp, Rd) by one fixed dt.

    Uses the global nonlinear solver if cfg.nonlinear.enabled is True; otherwise uses
    the linear backend dispatch path. The name is kept for compatibility; inputs
    state/props are treated as t^n; outputs are t^{n+1}. Does not mutate inputs.
    """
    dt = float(cfg.time.dt)
    if dt <= 0.0:
        raise ValueError(f"cfg.time.dt must be positive, got {dt}")
    t_old = float(t)
    t_new = t_old + dt

    state_old = state.copy()
    props_old = props  # Step 12.2: treat props as quasi-steady (no recompute here)

    nl_cfg = getattr(cfg, "nonlinear", None)
    use_nonlinear = bool(getattr(nl_cfg, "enabled", False))
    if use_nonlinear:
        return _advance_one_step_nonlinear_scipy(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state_old,
            props_old=props_old,
            t_old=t_old,
            dt=dt,
        )

    # initial nonlinear guess (MVP: old state)
    state_guess = state_old

    try:
        A, b, diag_sys = _assemble_transport_system_step12(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state_old,
            state_guess=state_guess,
            props=props_old,
            dt=dt,
        )
    except Exception as exc:
        logger.exception("Failed to assemble transport system.")
        diag = _build_step_diagnostics_fail(t_old, t_new, dt, message=str(exc))
        _write_interface_diag_safe(cfg=cfg, diag=diag)  # P0.3: Write diag even on failure
        return StepResult(
            state_new=state_old,
            props_new=props_old,
            diag=diag,
            success=False,
            message=f"assembly failed: {exc}",
        )

    # Initial guess from current state (mainly for API symmetry; direct solver ignores it)
    u0, _, _ = pack_state(state_old, layout)

    try:
        lin_result: LinearSolveResult = solve_linear_system(A=A, b=b, cfg=cfg, x0=u0, layout=layout)
    except Exception as exc:
        logger.exception("Linear solve raised an exception.")
        diag = _build_step_diagnostics_fail(t_old, t_new, dt, message=str(exc))
        _write_interface_diag_safe(cfg=cfg, diag=diag)  # P0.3: Write diag even on failure
        return StepResult(
            state_new=state_old,
            props_new=props_old,
            diag=diag,
            success=False,
            message=f"linear solve error: {exc}",
        )

    if not lin_result.converged:
        diag = _build_step_diagnostics_fail(
            t_old,
            t_new,
            dt,
            linear=lin_result,
            diag_sys=diag_sys,
            message=lin_result.message,
        )
        _write_interface_diag_safe(cfg=cfg, diag=diag)  # P0.3: Write diag even on failure
        return StepResult(
            state_new=state_old,
            props_new=props_old,
            diag=diag,
            success=False,
            message=f"linear solve not converged: {lin_result.message}",
        )

    state_new = apply_u_to_state(
        state=state_old,
        u=lin_result.x,
        layout=layout,
        tol_closure=float(cfg.checks.sumY_tol),
        clip_negative_closure=True,  # Enable automatic correction of numerical errors
    )
    _postprocess_species_bounds(cfg, layout, state_new)

    # Apply no_condensation constraint if configured
    if cfg.physics.include_mpp and layout.has_block("mpp"):
        interface_type = getattr(cfg.physics.interface, "type", "no_condensation")
        if interface_type == "no_condensation" and state_new.mpp < 0.0:
            state_new.mpp = 0.0

    # --- Stage 2 removed: Tl is now solved in Stage 1 (coupled mode) ---
    # No separate liquid temperature solve needed
    liq_diag: Dict[str, Any] | None = None
    liq_lin: Optional[LinearSolveResult] = None

    diag = _build_step_diagnostics(
        lin_result=lin_result,
        state_new=state_new,
        diag_sys=diag_sys,
        t_old=t_old,
        dt=dt,
        layout=layout,
        cfg=cfg,
        grid=grid,
        liq_diag=liq_diag,
    )

    # Guard against mpp sign flip vs interface evaluation (diagnostic-only, no failure in nonlinear mode)
    try:
        diag_sys_evap = diag_sys.get("gas", {}).get("evaporation", {})
        mpp_eval = float(diag_sys_evap.get("mpp_eval", np.nan))
        mpp_state = float(state_new.mpp)
        if np.isfinite(mpp_eval) and np.isfinite(mpp_state):
            if np.sign(mpp_eval) != np.sign(mpp_state) and max(abs(mpp_eval), abs(mpp_state)) > 0.0:
                rel_diff = abs(mpp_state - mpp_eval) / max(abs(mpp_eval), 1e-30)
                if rel_diff > 0.2:
                    try:
                        diag.extra = dict(diag.extra)
                        diag.extra["mpp_mismatch"] = {
                            "mpp_state": mpp_state,
                            "mpp_eval": mpp_eval,
                            "rel_diff": rel_diff,
                        }
                    except Exception:
                        pass
    except Exception:
        pass

    # Recompute properties based on the updated state (for use by the caller in the next step)
    try:
        props_new, extras_new = compute_props(cfg, grid, state_new)
        try:
            diag.extra = dict(diag.extra)
            diag.extra["props_extra"] = extras_new
        except Exception:
            pass
    except Exception as exc:
        try:
            diag.extra = dict(diag.extra)
            diag.extra["props_error"] = {"type": type(exc).__name__, "message": str(exc)}
        except Exception:
            pass
        return StepResult(
            state_new=state_new,
            props_new=props_old,
            diag=diag,
            success=False,
            message=f"compute_props failed: {exc}",
        )

    sanity_msg = _sanity_check_step(cfg=cfg, grid=grid, state_new=state_new, props_new=props_new, diag=diag)
    if sanity_msg is not None:
        try:
            diag.extra = dict(diag.extra)
            diag.extra["sanity"] = {"reason": sanity_msg}
        except Exception:
            pass
        return StepResult(
            state_new=state_new,
            props_new=props_new,
            diag=diag,
            success=False,
            message=sanity_msg,
        )

    _write_step_scalars_safe(cfg=cfg, t=t_new, state=state_new, diag=diag)
    _write_interface_diag_safe(cfg=cfg, diag=diag)

    return StepResult(
        state_new=state_new,
        props_new=props_new,
        diag=diag,
        success=True,
        message=None,
    )


def _advance_one_step_nonlinear_scipy(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props_old: Props,
    t_old: float,
    dt: float,
) -> StepResult:
    """Advance one step using global nonlinear solve (backend dispatch)."""
    t_new = t_old + dt

    try:
        ctx, u0 = build_nonlinear_context_for_step(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state_old,
            props_old=props_old,
            t_old=t_old,
            dt=dt,
        )
    except Exception as exc:
        logger.exception("Failed to build nonlinear context.")
        diag = _build_step_diagnostics_fail(t_old, t_new, dt, message=str(exc))
        diag.nonlinear_converged = False
        diag.nonlinear_method = "build_context"
        return StepResult(
            state_new=state_old,
            props_new=props_old,
            diag=diag,
            success=False,
            message=f"nonlinear context build failed: {exc}",
        )

    try:
        nl_result = solve_nonlinear(ctx, u0)
    except Exception as exc:
        logger.exception("Nonlinear solve raised an exception.")
        diag = _build_step_diagnostics_fail(t_old, t_new, dt, message=str(exc))
        diag.nonlinear_converged = False
        diag.nonlinear_method = getattr(getattr(cfg, "nonlinear", None), "solver", "unknown")
        return StepResult(
            state_new=state_old,
            props_new=props_old,
            diag=diag,
            success=False,
            message=f"nonlinear solve error: {exc}",
        )

    if not nl_result.diag.converged:
        diag = _build_step_diagnostics_fail(
            t_old,
            t_new,
            dt,
            message=nl_result.diag.message or "Nonlinear solver did not converge",
        )
        _apply_nonlinear_diag(diag, nl_result.diag)
        try:
            diag.extra = dict(diag.extra)
            diag.extra["nonlinear"] = {
                "history_res_inf": list(nl_result.diag.history_res_inf),
                "message": nl_result.diag.message,
                "extra": dict(getattr(nl_result.diag, "extra", {}) or {}),
            }
        except Exception:
            pass
        _write_interface_diag_safe(cfg=cfg, diag=diag)  # P0.3: Write diag for nonlinear failure
        return StepResult(
            state_new=state_old,
            props_new=props_old,
            diag=diag,
            success=False,
            message=diag.extra.get("message", "Nonlinear solver did not converge"),
        )

    u_final = nl_result.u
    if not np.all(np.isfinite(u_final)):
        diag = _build_step_diagnostics_fail(
            t_old,
            t_new,
            dt,
            message="Nonlinear solver returned non-finite u",
        )
        _apply_nonlinear_diag(diag, nl_result.diag)
        try:
            diag.extra = dict(diag.extra)
            diag.extra["nonlinear"] = {
                "history_res_inf": list(nl_result.diag.history_res_inf),
                "message": nl_result.diag.message,
                "extra": dict(getattr(nl_result.diag, "extra", {}) or {}),
            }
        except Exception:
            pass
        return StepResult(
            state_new=state_old,
            props_new=props_old,
            diag=diag,
            success=False,
            message="Nonlinear returned non-finite u",
        )
    state_new = ctx.make_state_from_u(u_final, clip_negative_closure=True)
    _postprocess_species_bounds(cfg, layout, state_new)

    # Apply no_condensation constraint if configured
    if cfg.physics.include_mpp and layout.has_block("mpp"):
        interface_type = getattr(cfg.physics.interface, "type", "no_condensation")
        if interface_type == "no_condensation" and state_new.mpp < 0.0:
            state_new.mpp = 0.0

    diag_sys: Dict[str, Any] = {}
    try:
        _, diag_res = build_global_residual(u_final, ctx)
        if isinstance(diag_res, dict):
            diag_sys = dict(diag_res.get("assembly", {}) or {})
    except Exception as exc:
        logger.warning("Failed to assemble diagnostics for nonlinear step: %s", exc)

    lin_stub = LinearSolveResult(
        x=np.asarray(u_final, dtype=np.float64),
        converged=True,
        n_iter=0,
        residual_norm=0.0,
        rel_residual=0.0,
        method="nonlinear-global",
        message=None,
    )

    diag = _build_step_diagnostics(
        lin_result=lin_stub,
        state_new=state_new,
        diag_sys=diag_sys,
        t_old=t_old,
        dt=dt,
        layout=layout,
        cfg=cfg,
        grid=grid,
        liq_diag=None,
    )
    _apply_nonlinear_diag(diag, nl_result.diag)
    try:
        diag.extra = dict(diag.extra)
        diag.extra["nonlinear"] = {
            "history_res_inf": list(nl_result.diag.history_res_inf),
            "message": nl_result.diag.message,
            "extra": dict(getattr(nl_result.diag, "extra", {}) or {}),
        }
    except Exception:
        pass

    # Guard against mpp sign flip vs interface evaluation (diagnostic-only, not altering state)
    try:
        diag_sys_evap = diag_sys.get("gas", {}).get("evaporation", {})
        mpp_eval = float(diag_sys_evap.get("mpp_eval", np.nan))
        mpp_state = float(state_new.mpp)
        if np.isfinite(mpp_eval) and np.isfinite(mpp_state):
            if np.sign(mpp_eval) != np.sign(mpp_state) and max(abs(mpp_eval), abs(mpp_state)) > 0.0:
                rel_diff = abs(mpp_state - mpp_eval) / max(abs(mpp_eval), 1e-30)
                if rel_diff > 0.2:
                    try:
                        diag.extra = dict(diag.extra)
                        diag.extra["mpp_mismatch"] = {
                            "mpp_state": mpp_state,
                            "mpp_eval": mpp_eval,
                            "rel_diff": rel_diff,
                        }
                    except Exception:
                        pass
                    return StepResult(
                        state_new=state_new,
                        props_new=props_old,
                        diag=diag,
                        success=False,
                        message=f"mpp sign mismatch: state={mpp_state:.3e}, eval={mpp_eval:.3e}, rel_diff={rel_diff:.3e}",
                    )
    except Exception:
        pass

    # Recompute properties based on the updated state (for use by the caller in the next step)
    try:
        props_new, extras_new = compute_props(cfg, grid, state_new)
        try:
            diag.extra = dict(diag.extra)
            diag.extra["props_extra"] = extras_new
        except Exception:
            pass
    except Exception as exc:
        try:
            diag.extra = dict(diag.extra)
            diag.extra["props_error"] = {"type": type(exc).__name__, "message": str(exc)}
        except Exception:
            pass
        return StepResult(
            state_new=state_new,
            props_new=props_old,
            diag=diag,
            success=False,
            message=f"compute_props failed: {exc}",
        )

    sanity_msg = _sanity_check_step(cfg=cfg, grid=grid, state_new=state_new, props_new=props_new, diag=diag)
    if sanity_msg is not None:
        try:
            diag.extra = dict(diag.extra)
            diag.extra["sanity"] = {"reason": sanity_msg}
        except Exception:
            pass
        return StepResult(
            state_new=state_new,
            props_new=props_new,
            diag=diag,
            success=False,
            message=sanity_msg,
        )

    _write_step_scalars_safe(cfg=cfg, t=t_new, state=state_new, diag=diag)
    _write_interface_diag_safe(cfg=cfg, diag=diag)

    return StepResult(
        state_new=state_new,
        props_new=props_new,
        diag=diag,
        success=True,
        message=None,
    )


def _assemble_transport_system_step12(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    state_guess: State,
    props: Props,
    dt: float,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Wrapper for build_transport_system with basic shape checks."""
    eq_result = None
    if cfg.physics.include_mpp and layout.has_block("mpp"):
        eq_result = _build_eq_result_for_step(cfg, grid, state_guess, props)
        eq_result = _complete_Yg_eq_with_closure(cfg, layout, eq_result)
    N = layout.n_dof()
    solver_cfg = getattr(cfg, "solver", None)
    linear_cfg = getattr(solver_cfg, "linear", None)
    backend = getattr(linear_cfg, "backend", None)
    if backend is None:
        backend = getattr(getattr(cfg, "nonlinear", None), "backend", "scipy")
    backend = str(backend).lower()
    assembly_mode = str(getattr(linear_cfg, "assembly_mode", "bridge_dense")).lower()
    if backend == "petsc":
        if assembly_mode == "bridge_dense":
            result = build_transport_system_petsc_bridge(
                cfg=cfg,
                grid=grid,
                layout=layout,
                state_old=state_old,
                state_guess=state_guess,
                props=props,
                dt=dt,
                eq_result=eq_result,
                return_diag=True,
            )
        elif assembly_mode == "native_aij":
            result = build_transport_system_petsc_native(
                cfg=cfg,
                grid=grid,
                layout=layout,
                state_old=state_old,
                state_guess=state_guess,
                props=props,
                dt=dt,
                eq_result=eq_result,
                return_diag=True,
            )
        else:
            raise ValueError(
                f"Unknown PETSc linear assembly_mode '{assembly_mode}' "
                " (expected 'bridge_dense' or 'native_aij')."
            )
        if len(result) == 3:
            A, b, diag_sys = result
        else:
            A, b = result  # type: ignore[misc]
            diag_sys = {}
        m, n = A.getSize()
        if (m, n) != (N, N):
            raise ValueError(f"Assembly produced A size {(m, n)}, expected {(N, N)}")
        if b.getSize() != N:
            raise ValueError(f"Assembly produced b size {b.getSize()}, expected {N}")
    else:
        result = build_transport_system_numpy(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state_old,
            state_guess=state_guess,
            props=props,
            dt=dt,
            eq_result=eq_result,
            return_diag=True,
        )
        # build_transport_system_numpy may return a tuple of length 2 or 3 depending on return_diag
        if len(result) == 3:
            A, b, diag_sys = result
        else:
            A, b = result  # type: ignore[misc]
            diag_sys = {}
        if A.shape != (N, N):
            raise ValueError(f"Assembly produced A shape {A.shape}, expected {(N, N)}")
        if b.shape != (N,):
            raise ValueError(f"Assembly produced b shape {b.shape}, expected {(N,)}")
    if diag_sys is None:
        diag_sys = {}
    return A, b, diag_sys


def _postprocess_species_bounds(cfg: CaseConfig, layout: UnknownLayout, state: State) -> None:
    """Clamp/check gas species after apply_u_to_state; recompute closure from reduced species."""
    if not layout.has_block("Yg") or layout.Ns_g_eff == 0:
        return
    clamp = bool(getattr(cfg.checks, "clamp_negative_Y", False))
    min_Y = float(getattr(cfg.checks, "min_Y", 0.0))
    tol = float(getattr(cfg.checks, "sumY_tol", 1e-10))

    # Clamp solved (reduced) species if requested
    if clamp:
        for k_red in range(layout.Ns_g_eff):
            k_full = layout.gas_reduced_to_full_idx[k_red]
            state.Yg[k_full, :] = np.maximum(state.Yg[k_full, :], min_Y)
    else:
        solved_full = [layout.gas_reduced_to_full_idx[k] for k in range(layout.Ns_g_eff)]
        ymin = float(np.min(state.Yg[solved_full, :]))
        ymax = float(np.max(state.Yg[solved_full, :]))
        if ymin < -tol:
            raise ValueError(f"Gas species below tolerance after solve (min={ymin:.3e}, tol={tol:.3e})")
        if ymax > 1.0 + tol:
            raise ValueError(f"Gas species exceed 1 after solve (max={ymax:.3e}, tol={tol:.3e})")

    # Recompute closure from all non-closure species (align with layout/_reconstruct_closure)
    closure_idx = getattr(layout, "gas_closure_index", None)
    if closure_idx is not None:
        sum_other = np.sum(state.Yg, axis=0) - state.Yg[closure_idx, :]
        closure = 1.0 - sum_other

        if not clamp:
            if np.any(closure < -tol):
                raise ValueError(f"Gas closure species negative beyond tol={tol}: min={float(np.min(closure)):.3e}")
            if np.any(closure > 1.0 + tol):
                raise ValueError(f"Gas closure species exceeds 1 beyond tol={tol}: max={float(np.max(closure)):.3e}")
        else:
            # Soft clamp small violations to stay consistent with layout reconstruction
            closure = np.where((closure < 0.0) & (closure >= -tol), 0.0, closure)
            closure = np.where((closure > 1.0) & (closure <= 1.0 + tol), 1.0, closure)
            closure = np.clip(closure, 0.0, 1.0)

        state.Yg[closure_idx, :] = closure

    # FIXED: Use simplex projection instead of clip to enforce sum(Y) = 1.0
    # Previous np.clip() would change sum(Y), violating mass conservation.
    # Simplex projection maintains sum(Y) = 1.0 while ensuring Y >= min_Y.
    if clamp:
        state.Yg = project_Y_cellwise(state.Yg, min_Y=min_Y, axis=0)
    else:
        # Even without clamp, apply projection to fix any floating-point errors
        state.Yg = project_Y_cellwise(state.Yg, min_Y=1e-14, axis=0)

def _build_step_diagnostics(
    lin_result: LinearSolveResult,
    state_new: State,
    diag_sys: Dict[str, Any],
    t_old: float,
    dt: float,
    layout: UnknownLayout,
    cfg: CaseConfig,
    grid: Grid1D,
    liq_diag: Optional[Dict[str, Any]] = None,
) -> StepDiagnostics:
    """Assemble StepDiagnostics from solve result and system diag."""
    t_new = t_old + dt
    Tg_min = float(np.min(state_new.Tg)) if state_new.Tg.size else np.nan
    Tg_max = float(np.max(state_new.Tg)) if state_new.Tg.size else np.nan

    energy_balance_if = None
    if "Ts_energy" in diag_sys:
        energy_balance_if = float(diag_sys["Ts_energy"].get("balance", np.nan))

    mass_balance_rd = None
    if "radius_eq" in diag_sys:
        mass_balance_rd = float(diag_sys["radius_eq"].get("mass_balance", np.nan))

    extra: Dict[str, Any] = {"diag_sys": diag_sys}
    if liq_diag is not None:
        extra["liq_T"] = liq_diag
    if layout.has_block("Yg") and state_new.Yg.size:
        Yg_min = float(np.min(state_new.Yg))
        Yg_max = float(np.max(state_new.Yg))
        closure_idx = getattr(layout, "gas_closure_index", None)
        closure_min = float(np.min(state_new.Yg[closure_idx, :])) if closure_idx is not None else None
        cond_name = None
        cond_val = None
        liq_bal = getattr(cfg.species, "liq_balance_species", None)
        if liq_bal and liq_bal in getattr(cfg.species, "liq2gas_map", {}):
            cond_name = cfg.species.liq2gas_map[liq_bal]
            k_red = layout.gas_full_to_reduced.get(cond_name)
            if k_red is not None and grid.Ng > 0:
                k_full = layout.gas_reduced_to_full_idx[k_red]
                cond_val = float(state_new.Yg[k_full, 0])
        extra["Yg_stats"] = {
            "min": Yg_min,
            "max": Yg_max,
            "closure_min": closure_min,
            "condensable_name": cond_name,
            "condensable_if_cell": cond_val,
        }

    # P0.2: Construct interface_diag dict for interface_diag.csv output
    interface_diag = {
        "t": t_new,
        "dt": dt,
        "Ts": float(state_new.Ts),
        "Rd": float(state_new.Rd),
        "mpp": float(state_new.mpp),
        "m_dot": float("nan"),  # Will be computed below
        "psat": float("nan"),
        "sum_y_cond": float("nan"),
        "eq_source": "none",
        "eq_exc_type": "",
        "eq_exc_msg": "",
        # P0.1: New diagnostic fields for source tracking
        "psat_source": "",
        "hvap_source": "",
        "fallback_reason": "",
        "finite_ok": "",
    }

    # Compute m_dot = 4*pi*Rd^2*mpp
    try:
        Rd_val = float(state_new.Rd)
        mpp_val = float(state_new.mpp)
        if np.isfinite(Rd_val) and np.isfinite(mpp_val):
            interface_diag["m_dot"] = 4.0 * np.pi * (Rd_val ** 2) * mpp_val
    except Exception:
        pass

    # Try to build equilibrium result and extract psat/sum_y_cond + meta
    if cfg.physics.include_mpp and layout.has_block("mpp"):
        try:
            eq = _build_eq_result_for_step(cfg, grid, state_new, props=None)
            psat_arr = np.asarray(eq.get("psat", []))
            y_cond_arr = np.asarray(eq.get("y_cond", []))
            if psat_arr.size > 0:
                interface_diag["psat"] = float(psat_arr.reshape(-1)[0])
            if y_cond_arr.size > 0:
                interface_diag["sum_y_cond"] = float(np.sum(y_cond_arr))
            interface_diag["eq_source"] = "computed"
            # P0.1: Extract meta tracking info
            meta = eq.get("meta", {})
            interface_diag["psat_source"] = meta.get("psat_source", "")
            interface_diag["hvap_source"] = meta.get("hvap_source", "")
            interface_diag["fallback_reason"] = meta.get("fallback_reason", "")
            interface_diag["finite_ok"] = str(meta.get("finite_ok", ""))
        except Exception as exc:
            interface_diag["eq_source"] = "failed"
            interface_diag["eq_exc_type"] = type(exc).__name__
            interface_diag["eq_exc_msg"] = str(exc)

    extra["interface_diag"] = interface_diag

    return StepDiagnostics(
        t_old=t_old,
        t_new=t_new,
        dt=dt,
        linear_converged=lin_result.converged,
        linear_method=lin_result.method,
        linear_n_iter=lin_result.n_iter,
        linear_residual_norm=lin_result.residual_norm,
        linear_rel_residual=lin_result.rel_residual,
        Ts=float(state_new.Ts),
        Rd=float(state_new.Rd),
        mpp=float(state_new.mpp),
        Tg_min=Tg_min,
        Tg_max=Tg_max,
        energy_balance_if=energy_balance_if,
        mass_balance_rd=mass_balance_rd,
        extra=extra,
    )


def _apply_nonlinear_diag(diag: StepDiagnostics, nl_diag: Any) -> None:
    """Copy nonlinear diagnostics into StepDiagnostics."""
    try:
        diag.nonlinear_converged = bool(getattr(nl_diag, "converged", False))
        diag.nonlinear_method = str(getattr(nl_diag, "method", ""))
        diag.nonlinear_n_iter = int(getattr(nl_diag, "n_iter", 0))
        diag.nonlinear_residual_norm = float(getattr(nl_diag, "res_norm_2", np.nan))
        diag.nonlinear_residual_inf = float(getattr(nl_diag, "res_norm_inf", np.nan))
    except Exception:
        pass


def _sanity_check_step(cfg: CaseConfig, grid: Grid1D, state_new: State, props_new: Props, diag: StepDiagnostics) -> Optional[str]:
    """
    Step-level sanity checks after solve and before writing outputs.

    Returns
    -------
    None if OK, or a string describing the first detected issue.
    """
    # --- numeric health ---
    if not np.isfinite(state_new.Rd) or state_new.Rd <= 0.0:
        return "Rd non-positive or non-finite after step"
    if not np.isfinite(state_new.Ts):
        return "Ts non-finite after step"
    if not np.isfinite(state_new.mpp):
        return "mpp non-finite after step"
    if np.any(~np.isfinite(state_new.Tg)):
        return "Tg contains non-finite entries after step"
    if cfg.physics.enable_liquid and np.any(~np.isfinite(state_new.Tl)):
        return "Tl contains non-finite entries after step"

    # --- temperature bounds ---
    if getattr(cfg.checks, "enforce_T_bounds", False):
        T_min = float(cfg.checks.T_min)
        T_max = float(cfg.checks.T_max)

        def _check_T(arr: np.ndarray, name: str) -> Optional[str]:
            if arr.size == 0:
                return None
            amin = float(np.min(arr))
            amax = float(np.max(arr))
            if amin < T_min or amax > T_max:
                return f"{name} out of bounds [{T_min}, {T_max}] (min={amin:.3e}, max={amax:.3e})"
            return None

        msg = _check_T(state_new.Tg, "Tg")
        if msg:
            return msg
        if cfg.physics.enable_liquid:
            msg = _check_T(state_new.Tl, "Tl")
            if msg:
                return msg
        msg = _check_T(np.array([state_new.Ts], dtype=np.float64), "Ts")
        if msg:
            return msg

    # --- mass fractions sum and negativity ---
    if getattr(cfg.checks, "enforce_sumY", False):
        tol = float(getattr(cfg.checks, "sumY_tol", 1e-10))
        if cfg.physics.solve_Yg and state_new.Yg.size:
            s = np.sum(state_new.Yg, axis=0)
            max_err = float(np.max(np.abs(s - 1.0)))
            if max_err > tol:
                return f"Gas Y sum deviates by {max_err:.3e} (tol={tol:.3e})"
        if cfg.physics.solve_Yl and state_new.Yl.size:
            s = np.sum(state_new.Yl, axis=0)
            max_err = float(np.max(np.abs(s - 1.0)))
            if max_err > tol:
                return f"Liquid Y sum deviates by {max_err:.3e} (tol={tol:.3e})"

    if not getattr(cfg.checks, "clamp_negative_Y", False):
        min_Y_tol = float(getattr(cfg.checks, "min_Y", 0.0))
        if cfg.physics.solve_Yg and state_new.Yg.size:
            ymin = float(np.min(state_new.Yg))
            if ymin < -min_Y_tol:
                return f"Gas Y has values below -min_Y (min={ymin:.3e}, min_Y={min_Y_tol:.3e})"
        if cfg.physics.solve_Yl and state_new.Yl.size:
            ymin = float(np.min(state_new.Yl))
            if ymin < -min_Y_tol:
                return f"Liquid Y has values below -min_Y (min={ymin:.3e}, min_Y={min_Y_tol:.3e})"

    # --- linear solve health ---
    if not diag.linear_converged:
        return "Linear solver reported non-convergence"
    rel_res = float(diag.linear_rel_residual)
    if not np.isfinite(rel_res):
        return "Linear relative residual is non-finite"
    rtol = float(getattr(cfg.petsc, "rtol", 1.0e-8))
    thresh = max(rtol, 1.0e-8)
    if rel_res > max(thresh, rtol * 10.0):
        return f"Linear relative residual too large ({rel_res:.3e} > {thresh:.3e})"

    # --- time consistency ---
    if diag.dt <= 0.0:
        return f"Non-positive dt in diagnostics: dt={diag.dt}"
    if diag.t_new <= diag.t_old:
        return f"Non-increasing time: t_new={diag.t_new}, t_old={diag.t_old}"
    expected = diag.t_old + diag.dt
    tol = 1e-12 * max(1.0, abs(diag.t_old), abs(diag.t_new), abs(diag.dt))
    if abs(expected - diag.t_new) > tol:
        return f"t_new mismatch: expected {expected:.12e} vs diag {diag.t_new:.12e}"

    return None


def _build_step_diagnostics_fail(
    t_old: float,
    t_new: float,
    dt: float,
    message: Optional[str] = None,
    linear: Optional[LinearSolveResult] = None,
    diag_sys: Optional[Dict[str, Any]] = None,
) -> StepDiagnostics:
    """Diagnostics when assembly or solve fails."""
    diag_sys = diag_sys or {}
    method = linear.method if linear is not None else "unknown"
    n_iter = linear.n_iter if linear is not None else 0
    res_norm = linear.residual_norm if linear is not None else np.nan
    rel_res = linear.rel_residual if linear is not None else np.nan

    # P0.3: Add interface_diag even for failed steps
    interface_diag = {
        "t": t_new,
        "dt": dt,
        "Ts": float("nan"),
        "Rd": float("nan"),
        "mpp": float("nan"),
        "m_dot": float("nan"),
        "psat": float("nan"),
        "sum_y_cond": float("nan"),
        "eq_source": "step_failed",
        "eq_exc_type": "",
        "eq_exc_msg": message or "",
        "psat_source": "",
        "hvap_source": "",
        "fallback_reason": "",
        "finite_ok": "False",
    }

    return StepDiagnostics(
        t_old=t_old,
        t_new=t_new,
        dt=dt,
        linear_converged=False,
        linear_method=method,
        linear_n_iter=n_iter,
        linear_residual_norm=res_norm,
        linear_rel_residual=rel_res,
        Ts=np.nan,
        Rd=np.nan,
        mpp=np.nan,
        Tg_min=np.nan,
        Tg_max=np.nan,
        energy_balance_if=None,
        mass_balance_rd=None,
        extra={"diag_sys": diag_sys, "message": message, "interface_diag": interface_diag},
    )


_WRITERS_MODULE_NAME = "io_writers_cached"


def _write_step_scalars_safe(cfg: CaseConfig, t: float, state: State, diag: StepDiagnostics) -> None:
    """
    Safely load and call write_step_scalars without importing stdlib io module.
    Uses a cached dynamic import to avoid conflicts with builtin io.
    """
    try:
        module = sys.modules.get(_WRITERS_MODULE_NAME)
        if module is None:
            path = Path(__file__).resolve().parent.parent / "io" / "writers.py"
            spec = importlib.util.spec_from_file_location(_WRITERS_MODULE_NAME, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load writers module from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[_WRITERS_MODULE_NAME] = module
            spec.loader.exec_module(module)  # type: ignore[arg-type]
        write_fn = getattr(module, "write_step_scalars", None)
        if write_fn is None:
            raise AttributeError("write_step_scalars not found in writers module")
        write_fn(cfg=cfg, t=t, state=state, diag=diag)
    except Exception as exc:
        logger.warning("write_step_scalars failed: %s", exc)


def _write_interface_diag_safe(cfg: CaseConfig, diag: StepDiagnostics) -> None:
    """
    Safely load and call write_interface_diag without importing stdlib io module.
    Uses the same cached dynamic import as _write_step_scalars_safe.

    P0.2: This writes interface_diag.csv with one row per timestep containing
    interface equilibrium diagnostics and state variables.
    """
    try:
        module = sys.modules.get(_WRITERS_MODULE_NAME)
        if module is None:
            path = Path(__file__).resolve().parent.parent / "io" / "writers.py"
            spec = importlib.util.spec_from_file_location(_WRITERS_MODULE_NAME, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load writers module from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[_WRITERS_MODULE_NAME] = module
            spec.loader.exec_module(module)  # type: ignore[arg-type]
        write_fn = getattr(module, "write_interface_diag", None)
        if write_fn is None:
            raise AttributeError("write_interface_diag not found in writers module")
        write_fn(cfg=cfg, diag=diag)
    except Exception as exc:
        logger.warning("write_interface_diag failed: %s", exc)


def _build_eq_result_for_step(
    cfg: CaseConfig,
    grid: Grid1D,
    state: State,
    props: Props,
) -> EqResultLike:
    """
    Compute interface equilibrium result for the current step.

    P0.1: Now uses compute_interface_equilibrium_full to get meta tracking.
    """
    from properties.equilibrium import compute_interface_equilibrium_full

    il_if = grid.Nl - 1
    ig_if = 0

    Ts = float(state.Ts)
    Yl_face = np.asarray(state.Yl[:, il_if], dtype=np.float64)
    Yg_face = np.asarray(state.Yg[:, ig_if], dtype=np.float64)
    Ns_g = Yg_face.shape[0]
    Ns_l = Yl_face.shape[0]

    M_g, M_l = _get_molar_masses_from_cfg(cfg, Ns_g=Ns_g, Ns_l=Ns_l)
    eq_model = build_equilibrium_model(cfg, Ns_g=Ns_g, Ns_l=Ns_l, M_g=M_g, M_l=M_l)

    Pg = float(getattr(cfg.initial, "P_inf", 101325.0))
    # P0.1: Use full version to get meta
    result = compute_interface_equilibrium_full(
        eq_model,
        Ts=Ts,
        Pg=Pg,
        Yl_face=Yl_face,
        Yg_face=Yg_face,
    )
    return {
        "Yg_eq": result.Yg_eq,
        "y_cond": result.y_cond,
        "psat": result.psat,
        "meta": result.meta,  # P0.1: Include meta tracking
    }


def _get_molar_masses_from_cfg(cfg: CaseConfig, Ns_g: int, Ns_l: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build molar mass arrays (kg/mol) aligned with species order in cfg."""
    mw_map = dict(getattr(cfg.species, "mw_kg_per_mol", {}))

    gas_names = list(getattr(cfg.species, "gas_species_full", []) or [])
    liq_names = list(getattr(cfg.species, "liq_species", []) or [])
    if len(gas_names) != Ns_g:
        raise ValueError(f"gas_species_full length {len(gas_names)} != Ns_g {Ns_g}")
    if len(liq_names) != Ns_l:
        raise ValueError(f"liq_species length {len(liq_names)} != Ns_l {Ns_l}")
    missing_l = [name for name in liq_names if name not in mw_map]
    if missing_l:
        raise ValueError(f"mw_kg_per_mol missing liquid entries: {missing_l}")

    gas_model, _ = get_or_build_models(cfg)
    mech_names = list(getattr(gas_model.gas, "species_names", []))
    if mech_names and mech_names != gas_names:
        raise ValueError("gas_species_full does not match Cantera mechanism species order.")
    M_g = np.asarray(gas_model.gas.molecular_weights, dtype=np.float64) / 1000.0
    if M_g.size != Ns_g:
        raise ValueError(f"Cantera molecular_weights size {M_g.size} != Ns_g {Ns_g}")

    def _build(names: list[str], Ns: int) -> np.ndarray:
        arr = np.ones(Ns, dtype=np.float64)
        for i, name in enumerate(names):
            if i >= Ns:
                break
            arr[i] = float(mw_map[name])
        return arr

    M_l = _build(liq_names, Ns_l)
    return M_g, M_l


def _complete_Yg_eq_with_closure(cfg: CaseConfig, layout: UnknownLayout, eq_result: EqResultLike) -> EqResultLike:
    """
    Ensure eq_result['Yg_eq'] is a valid full-length mass fraction vector (sum=1, closure filled).

    This guards against tests/monkeypatches that provide a partial Yg_eq (e.g., only condensables).
    """
    if eq_result is None or "Yg_eq" not in eq_result:
        return eq_result

    Y = np.asarray(eq_result["Yg_eq"], dtype=np.float64)
    Ns = len(getattr(cfg.species, "gas_species", Y))
    if Y.ndim != 1 or Y.size != Ns:
        raise ValueError(f"eq_result['Yg_eq'] must be 1D length {Ns}, got shape {Y.shape}")

    k_cl = getattr(layout, "gas_closure_index", None)
    tol = 1e-14
    Y = np.clip(Y, 0.0, 1.0)

    if k_cl is not None and 0 <= k_cl < Ns:
        sum_other = float(np.sum(Y) - Y[k_cl])
        Y[k_cl] = max(0.0, 1.0 - sum_other)
    s = float(np.sum(Y))
    if s > tol and not np.isclose(s, 1.0, rtol=1e-12, atol=1e-12):
        Y = Y / s
    eq_result = dict(eq_result)
    eq_result["Yg_eq"] = Y
    return eq_result
