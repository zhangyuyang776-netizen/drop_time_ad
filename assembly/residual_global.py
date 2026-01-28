"""
Global nonlinear residual F(u) for Tg/Tl/Yg/Yl/Ts/mpp/Rd (SciPy backend, Step 19.4.4).

Current status:
- Uses build_transport_system with state_guess = ctx.make_state(u).
- Material properties are recomputed from state_guess each residual evaluation.
- Interface equilibrium (Yg_eq) is recomputed from state_guess when include_mpp is enabled.
- On compute_props failure, optionally sanitize Yg and retry; if still failing, return a penalty residual.
- Defines F(u) = A(u) @ u - b(u).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from assembly.build_system_SciPy import build_transport_system
from core.logging_utils import is_root_rank
from properties.compute_props import compute_props, get_or_build_models
from properties.equilibrium import (
    InterfaceEquilibriumError,
    build_equilibrium_model,
    interface_equilibrium,
    decide_regime,
)
from solvers.nonlinear_context import NonlinearContext

logger = logging.getLogger(__name__)

_SANITIZE_Y = bool(int(os.environ.get("DROPLET_SANITIZE_Y", "1")))
_SANITIZE_STRICT = bool(int(os.environ.get("DROPLET_SANITIZE_STRICT", "1")))


def _ctx_tag(ctx: NonlinearContext) -> str:
    step = getattr(ctx, "step", "NA")
    t_old = getattr(ctx, "t_old", "NA")
    dt = getattr(ctx, "dt", "NA")
    return f"step={step} t={t_old} dt={dt}"


def _get_penalty_cfg(cfg) -> tuple[str, float, str]:
    nl = getattr(cfg, "nonlinear", None)
    mode = str(getattr(nl, "sanitize_mode", "penalty")).strip().lower()
    if mode not in ("penalty", "abort"):
        mode = "penalty"
    penalty_value = float(getattr(nl, "penalty_value", 1.0e20))
    penalty_scope = str(getattr(nl, "penalty_scope", "interface_only")).strip().lower()
    if penalty_scope not in ("interface_only", "all"):
        penalty_scope = "interface_only"
    return mode, penalty_value, penalty_scope


def _record_penalty(ctx: NonlinearContext, info: Dict[str, Any]) -> None:
    meta = getattr(ctx, "meta", None)
    if meta is None or not isinstance(meta, dict):
        return
    meta["penalty_count"] = int(meta.get("penalty_count", 0)) + 1
    meta["penalty_last"] = dict(info)


def _require_finite_array(name: str, arr: np.ndarray, *, ctx: NonlinearContext, stage: str) -> None:
    if arr.size == 0:
        return
    if not np.all(np.isfinite(arr)):
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        raise ValueError(f"[{stage}] Non-finite {name} ({_ctx_tag(ctx)}): min={mn:.6g} max={mx:.6g}")


def _require_finite_scalar(name: str, val: float, *, ctx: NonlinearContext, stage: str) -> None:
    if not np.isfinite(val):
        raise ValueError(f"[{stage}] Non-finite {name} ({_ctx_tag(ctx)}): {val!r}")


def _validate_mass_fractions(
    name: str,
    Y: np.ndarray,
    *,
    ctx: NonlinearContext,
    stage: str,
    enforce_sum: bool,
    sum_tol: float,
    min_Y: float,
) -> None:
    if Y.size == 0:
        return
    _require_finite_array(name, Y, ctx=ctx, stage=stage)
    ymin = float(np.min(Y))
    if ymin < min_Y - 1.0e-12:
        raise ValueError(
            f"[{stage}] {name} has values below min_Y={min_Y:.3e} ({_ctx_tag(ctx)}): min={ymin:.6g}"
        )
    if enforce_sum:
        s = np.sum(Y, axis=0)
        if not np.all(np.isfinite(s)):
            raise ValueError(f"[{stage}] Non-finite {name} sum ({_ctx_tag(ctx)})")
        max_err = float(np.max(np.abs(s - 1.0)))
        if max_err > sum_tol:
            raise ValueError(
                f"[{stage}] {name} sum off by {max_err:.3e} (tol={sum_tol:.3e}) ({_ctx_tag(ctx)})"
            )


def _validate_state_before_residual(state: State, cfg, *, ctx: NonlinearContext, stage: str) -> None:
    _require_finite_array("Tg", np.asarray(state.Tg, dtype=np.float64), ctx=ctx, stage=stage)
    _require_finite_array("Tl", np.asarray(state.Tl, dtype=np.float64), ctx=ctx, stage=stage)
    _require_finite_scalar("Ts", float(state.Ts), ctx=ctx, stage=stage)
    _require_finite_scalar("mpp", float(state.mpp), ctx=ctx, stage=stage)
    _require_finite_scalar("Rd", float(state.Rd), ctx=ctx, stage=stage)

    checks = getattr(cfg, "checks", None)
    enforce_sum = bool(getattr(checks, "enforce_sumY", True))
    sum_tol = float(getattr(checks, "sumY_tol", 1.0e-10))
    min_Y = float(getattr(checks, "min_Y", 0.0))
    _validate_mass_fractions(
        "Yg", np.asarray(state.Yg, dtype=np.float64), ctx=ctx, stage=stage, enforce_sum=enforce_sum, sum_tol=sum_tol, min_Y=min_Y
    )
    _validate_mass_fractions(
        "Yl", np.asarray(state.Yl, dtype=np.float64), ctx=ctx, stage=stage, enforce_sum=enforce_sum, sum_tol=sum_tol, min_Y=min_Y
    )


def _validate_props_before_residual(props: Props, *, ctx: NonlinearContext, stage: str) -> None:
    for name, arr in (
        ("rho_g", props.rho_g),
        ("cp_g", props.cp_g),
        ("k_g", props.k_g),
        ("rho_l", props.rho_l),
        ("cp_l", props.cp_l),
        ("k_l", props.k_l),
    ):
        _require_finite_array(name, np.asarray(arr, dtype=np.float64), ctx=ctx, stage=stage)
        if arr.size and np.any(arr <= 0.0):
            mn = float(np.min(arr))
            raise ValueError(f"[{stage}] {name} has non-positive entries ({_ctx_tag(ctx)}): min={mn:.6g}")

    if props.D_g is not None:
        _require_finite_array("D_g", np.asarray(props.D_g, dtype=np.float64), ctx=ctx, stage=stage)
    if props.D_l is not None:
        _require_finite_array("D_l", np.asarray(props.D_l, dtype=np.float64), ctx=ctx, stage=stage)
    if props.h_g is not None:
        _require_finite_array("h_g", np.asarray(props.h_g, dtype=np.float64), ctx=ctx, stage=stage)
    if props.h_gk is not None:
        _require_finite_array("h_gk", np.asarray(props.h_gk, dtype=np.float64), ctx=ctx, stage=stage)
    if props.h_l is not None:
        _require_finite_array("h_l", np.asarray(props.h_l, dtype=np.float64), ctx=ctx, stage=stage)
    if props.psat_l is not None:
        _require_finite_array("psat_l", np.asarray(props.psat_l, dtype=np.float64), ctx=ctx, stage=stage)
    if props.hvap_l is not None:
        _require_finite_array("hvap_l", np.asarray(props.hvap_l, dtype=np.float64), ctx=ctx, stage=stage)
    if props.h_vap_if is not None:
        _require_finite_scalar("h_vap_if", float(props.h_vap_if), ctx=ctx, stage=stage)


def _sanitize_mass_fractions(Y: np.ndarray, eps: float = 1.0e-30) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)
    Y = np.maximum(Y, 0.0)
    s = float(np.sum(Y))
    if s <= eps or not np.isfinite(s):
        j = int(np.argmax(Y)) if np.any(Y > 0.0) else (Y.size - 1)
        Y[:] = 0.0
        Y[j] = 1.0
        return Y
    Y /= s
    return Y


def _sanitize_state_Yg(state: State) -> tuple[State, int]:
    state_s = state.copy()
    Yg = np.asarray(state_s.Yg, dtype=np.float64)
    n_fix = 0
    for ig in range(Yg.shape[1]):
        y = Yg[:, ig]
        s = float(np.sum(y))
        miny = float(np.min(y)) if y.size else 0.0
        if (not np.isfinite(s)) or s <= 0.0 or abs(s - 1.0) > 1.0e-8 or miny < -1.0e-12:
            Yg[:, ig] = _sanitize_mass_fractions(y)
            n_fix += 1
    state_s.Yg = Yg
    return state_s, n_fix


def _penalty_residual(
    u: np.ndarray,
    layout,
    reason: Exception,
    *,
    scale: float = 1.0e20,
    scope: str = "interface_only",
    ctx: NonlinearContext | None = None,
    stage: str = "",
    extra: Dict[str, Any] | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    u = np.asarray(u, dtype=np.float64)
    res = np.zeros_like(u)
    scope_norm = str(scope or "interface_only").strip().lower()

    if scope_norm == "interface_only":
        blocks = [b for b in ("Ts", "mpp", "Rd") if layout.has_block(b)]
        if not blocks:
            res[:] = float(scale)
        else:
            for b in blocks:
                sl = layout.block_slice(b)
                res[sl] = float(scale)
    else:
        res[:] = float(scale)

    res_norm_2 = float(np.linalg.norm(res))
    res_norm_inf = float(np.linalg.norm(res, ord=np.inf))
    penalty_diag = {
        "penalty_used": True,
        "penalty_scope": scope_norm,
        "penalty_value": float(scale),
        "penalty_stage": str(stage),
        "penalty_reason": str(reason),
    }
    if isinstance(extra, dict):
        penalty_diag.update(extra)
    if ctx is not None:
        _record_penalty(ctx, penalty_diag)

    diag: Dict[str, Any] = {
        "assembly": {"penalty": True, "error": str(reason)},
        "penalty": dict(penalty_diag),
        "residual_norm_2": res_norm_2,
        "residual_norm_inf": res_norm_inf,
        "u_min": float(np.min(u)) if u.size else np.nan,
        "u_max": float(np.max(u)) if u.size else np.nan,
        "props": {"source": "penalty", "error": str(reason)},
    }
    try:
        if layout.has_block("Ts"):
            diag["Ts_guess"] = float(u[layout.idx_Ts()])
        if layout.has_block("Rd"):
            diag["Rd_guess"] = float(u[layout.idx_Rd()])
    except Exception:
        pass
    return res, diag


def _build_molar_mass_from_cfg(names: list[str], Ns: int, mw_map: dict[str, float]) -> np.ndarray:
    """Build molar mass array (kg/mol) aligned with provided names."""
    M = np.ones(Ns, dtype=np.float64)
    for i, name in enumerate(names):
        if i >= Ns:
            break
        if name in mw_map:
            try:
                M[i] = float(mw_map[name])
            except Exception:
                M[i] = 1.0
    return M


def _get_or_build_eq_model(
    ctx: NonlinearContext,
    state_guess,
) -> Any:
    """Return cached equilibrium model or build a new one when species sizes change."""
    meta = ctx.meta
    Ns_g = int(state_guess.Yg.shape[0])
    Ns_l = int(state_guess.Yl.shape[0])
    key = (Ns_g, Ns_l)

    if meta.get("eq_model_key") == key and meta.get("eq_model") is not None:
        return meta["eq_model"]

    cfg = ctx.cfg
    mw_map = dict(getattr(cfg.species, "mw_kg_per_mol", {}) or {})
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

    M_l = _build_molar_mass_from_cfg(liq_names, Ns_l, mw_map)

    eq_model = build_equilibrium_model(cfg, Ns_g=Ns_g, Ns_l=Ns_l, M_g=M_g, M_l=M_l)
    meta["eq_model"] = eq_model
    meta["eq_model_key"] = key
    try:
        specs = [eq_model.p2db.get_params(name) for name in eq_model.liq_names]
        tc_min = float(min(float(spec["Tc"]) for spec in specs))
        if np.isfinite(tc_min):
            meta.setdefault("tc_min", tc_min)
    except Exception:
        pass
    try:
        meta.setdefault("Ts_guard_dT", float(getattr(eq_model, "Ts_guard_dT", 3.0)))
    except Exception:
        pass
    return eq_model


def _update_iface_regime(ctx: NonlinearContext, eq_result: dict, cfg) -> None:
    """
    Apply hysteresis-based regime switching and persist in ctx.meta.

    Regimes: "evap" (default) or "sat".
    """
    if not isinstance(eq_result, dict):
        return
    meta = eq_result.get("meta")
    if meta is None or not isinstance(meta, dict):
        return

    s_raw = meta.get("sum_x_psat_over_P")
    if s_raw is None:
        s_raw = meta.get("sum_y_cond_raw", meta.get("psat_over_P", float("nan")))

    try:
        s_val = float(s_raw)
    except Exception:
        s_val = float("nan")
    if not np.isfinite(s_val):
        return

    eq_cfg = getattr(getattr(getattr(cfg, "physics", None), "interface", None), "equilibrium", None)
    tol_enter = float(getattr(eq_cfg, "sat_tol_enter", 1.0e-3)) if eq_cfg is not None else 1.0e-3
    tol_exit = float(getattr(eq_cfg, "sat_tol_exit", 5.0e-3)) if eq_cfg is not None else 5.0e-3
    max_switch = int(getattr(eq_cfg, "regime_lock_max", 1)) if eq_cfg is not None else 1

    prev = str(ctx.meta.get("iface_regime", "evap")).strip().lower()
    switch_count = int(ctx.meta.get("iface_regime_switch_count", 0))
    new, would_switch = decide_regime(s_val, prev, tol_enter=tol_enter, tol_exit=tol_exit)

    switched = False
    locked = False
    if new != prev:
        if switch_count < max_switch:
            ctx.meta["iface_regime"] = new
            ctx.meta["iface_regime_switch_count"] = switch_count + 1
            switched = True
        else:
            locked = True
            ctx.meta["iface_regime_locked"] = True
            new = prev
    else:
        ctx.meta.setdefault("iface_regime", prev)

    meta["sum_x_psat_over_P"] = float(s_val)
    meta["regime"] = new
    meta["regime_switched"] = bool(switched)
    meta["regime_locked"] = bool(locked)
    meta["regime_switch_count"] = int(ctx.meta.get("iface_regime_switch_count", 0))


def _closure_clamp_diag(
    Y_full: np.ndarray,
    closure_idx: int,
    *,
    tol: float,
    phase: str,
) -> Dict[str, Any]:
    """Compute closure diagnostics to detect out-of-bounds closure before clamping."""
    sum_other = np.sum(Y_full, axis=0) - Y_full[closure_idx, :]
    closure_raw = 1.0 - sum_other
    soft = ((closure_raw < 0.0) & (closure_raw >= -tol)) | ((closure_raw > 1.0) & (closure_raw <= 1.0 + tol))
    hard = (closure_raw < -tol) | (closure_raw > 1.0 + tol)
    any_out = (closure_raw < 0.0) | (closure_raw > 1.0)
    return {
        f"{phase}_closure_raw_min": float(np.min(closure_raw)) if closure_raw.size else np.nan,
        f"{phase}_closure_raw_max": float(np.max(closure_raw)) if closure_raw.size else np.nan,
        f"{phase}_closure_oob_any": int(np.count_nonzero(any_out)),
        f"{phase}_closure_oob_soft": int(np.count_nonzero(soft)),
        f"{phase}_closure_oob_hard": int(np.count_nonzero(hard)),
    }


def build_transport_system_from_ctx(
    ctx: NonlinearContext,
    u: np.ndarray,
    *,
    return_diag: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Assemble dense A(u), b(u) using the same logic as build_global_residual.

    Parameters
    ----------
    ctx : NonlinearContext
        Timestep context containing cfg/grid/layout/state_old/props_old/dt.
    u : ndarray
        Global unknown vector aligned with ctx.layout.
    return_diag : bool
        If True, also return assembly diagnostics from build_transport_system.
    """
    u = np.asarray(u, dtype=np.float64)
    N = ctx.layout.n_dof()
    if u.shape != (N,):
        raise ValueError(f"Global unknown vector shape {u.shape} != ({N},)")

    cfg = ctx.cfg
    grid = ctx.grid_ref
    layout = ctx.layout
    state_old = ctx.state_old
    props_old = ctx.props_old
    dt = float(ctx.dt)

    # Ensure ctx.meta exists and is a dict for debug bookkeeping
    meta = getattr(ctx, "meta", None)
    if meta is None or not isinstance(meta, dict):
        meta = {}
        try:
            ctx.meta = meta
        except Exception:
            pass

    try:
        state_guess = ctx.make_state(u)
    except Exception as exc:
        logger.warning("make_state(u) failed in residual_global: %s; fallback to state_old.", exc)
        state_guess = state_old

    t_min_cfg = float(getattr(getattr(cfg, "checks", None), "T_min", 1.0))
    if not np.isfinite(t_min_cfg) or t_min_cfg <= 0.0:
        t_min_cfg = 1.0

    state_props = state_guess.copy()
    if state_props.Tg.size:
        state_props.Tg = np.maximum(state_props.Tg, t_min_cfg)
    if state_props.Tl.size:
        state_props.Tl = np.maximum(state_props.Tl, t_min_cfg)
    state_props.Ts = max(float(state_props.Ts), t_min_cfg)

    try:
        props, _props_extras = compute_props(cfg, grid, state_props)
    except Exception:
        if os.environ.get("DROPLET_STRICT_PROPS", "0") == "1":
            raise
        logger.exception("compute_props failed in residual_global; falling back to props_old.")
        props = props_old

    eq_result = None
    eq_model = None
    phys = cfg.physics
    needs_eq = bool(getattr(phys, "include_mpp", False) and layout.has_block("mpp"))
    if needs_eq:
        cache = ctx.meta.get("eq_result_cache")
        eq_model = _get_or_build_eq_model(ctx, state_guess)
        # Initialize variables for diagnostic logging (avoid UnboundLocalError in exception handler)
        Ts_if = float("nan")
        Pg_if = float(getattr(cfg.initial, "P_inf", 101325.0))
        y_cond_sum = float("nan")
        psat_val = float("nan")
        Yl_shape = None
        Yg_shape = None
        try:
            il_if = grid.Nl - 1
            ig_if = 0
            Ts_if = float(state_props.Ts)
            Pg_if = float(getattr(cfg.initial, "P_inf", 101325.0))
            Yl_face = np.asarray(state_props.Yl[:, il_if], dtype=np.float64)
            Yg_face = np.asarray(state_props.Yg[:, ig_if], dtype=np.float64)
            Yl_shape = tuple(Yl_face.shape)
            Yg_shape = tuple(Yg_face.shape)
            # P4: Use interface_equilibrium to get full diagnostics including meta
            eq = interface_equilibrium(
                eq_model,
                Ts=Ts_if,
                Pg=Pg_if,
                Yl_face=Yl_face,
                Yg_face=Yg_face,
            )
            eq_result = {
                "Yg_eq": np.asarray(eq.Yg_eq),
                "y_cond": np.asarray(eq.y_cond),
                "psat": np.asarray(eq.psat),
                "meta": dict(eq.meta),  # P4: Preserve meta for source tracking
            }
            Yg_eq = eq_result["Yg_eq"]
            y_cond = eq_result["y_cond"]
            psat = eq_result["psat"]

            # P0.2: NaN/Inf sentinel - catch non-finite values immediately
            psat_arr = np.asarray(psat)
            y_cond_arr = np.asarray(y_cond)
            Yg_eq_arr = np.asarray(Yg_eq)
            if not np.isfinite(Ts_if):
                if is_root_rank():
                    logger.error(
                        "NONFINITE Ts_if=%.6g step=%s t=%s Pg_if=%.6g",
                        Ts_if, getattr(ctx, "step", "NA"), getattr(ctx, "t_old", "NA"), Pg_if
                    )
                raise ValueError(f"Non-finite Ts_if={Ts_if}")
            if not np.all(np.isfinite(psat_arr)):
                psat_bad = psat_arr[~np.isfinite(psat_arr)]
                if is_root_rank():
                    logger.error(
                        "NONFINITE psat step=%s t=%s Ts_if=%.6g bad_values=%s",
                        getattr(ctx, "step", "NA"), getattr(ctx, "t_old", "NA"), Ts_if, psat_bad
                    )
                raise ValueError(f"Non-finite psat at Ts_if={Ts_if}")
            if not np.all(np.isfinite(y_cond_arr)):
                y_cond_bad = y_cond_arr[~np.isfinite(y_cond_arr)]
                if is_root_rank():
                    logger.error(
                        "NONFINITE y_cond step=%s t=%s Ts_if=%.6g psat=%.6g bad_values=%s",
                        getattr(ctx, "step", "NA"), getattr(ctx, "t_old", "NA"), Ts_if,
                        psat_arr[0] if psat_arr.size > 0 else float("nan"), y_cond_bad
                    )
                raise ValueError(f"Non-finite y_cond at Ts_if={Ts_if}")
            if not np.all(np.isfinite(Yg_eq_arr)):
                Yg_eq_min = float(np.nanmin(Yg_eq_arr)) if Yg_eq_arr.size > 0 else float("nan")
                Yg_eq_max = float(np.nanmax(Yg_eq_arr)) if Yg_eq_arr.size > 0 else float("nan")
                if is_root_rank():
                    logger.error(
                        "NONFINITE Yg_eq step=%s t=%s Ts_if=%.6g psat=%.6g sum_y_cond=%.6g Yg_eq_min=%.6g Yg_eq_max=%.6g",
                        getattr(ctx, "step", "NA"), getattr(ctx, "t_old", "NA"), Ts_if,
                        psat_arr[0] if psat_arr.size > 0 else float("nan"),
                        float(np.sum(y_cond_arr)),
                        Yg_eq_min, Yg_eq_max
                    )
                raise ValueError(f"Non-finite Yg_eq at Ts_if={Ts_if}")

            # Extract diagnostics from successful computation
            y_cond_sum = float(np.sum(y_cond))
            psat_val = float(np.asarray(psat).reshape(-1)[0]) if np.size(psat) > 0 else float("nan")
            ctx.meta["eq_result_cache"] = dict(eq_result)
            try:
                _update_iface_regime(ctx, eq_result, cfg)
            except Exception:
                pass
        except Exception as exc:
            exc_type = type(exc).__name__
            exc_msg = str(exc)
            # P1: Record failure reason to ctx.meta
            ctx.meta["eq_last_error"] = f"{exc_type}: {exc_msg}"
            # P1: Fail-fast - no cache fallback
            raise InterfaceEquilibriumError(
                f"compute_interface_equilibrium failed at Ts={Ts_if:.6g}, Pg={Pg_if:.6g}"
            ) from exc

    result = build_transport_system(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        state_guess=state_guess,
        eq_model=eq_model,
        eq_result=eq_result,
        return_diag=bool(return_diag),
    )

    if return_diag:
        if isinstance(result, tuple) and len(result) == 3:
            A, b, diag_sys = result
        else:
            A, b = result  # type: ignore[misc]
            diag_sys = {}
        if A.shape != (N, N):
            raise ValueError(f"Assembly produced A shape {A.shape}, expected {(N, N)}")
        if b.shape != (N,):
            raise ValueError(f"Assembly produced b shape {b.shape}, expected {(N,)}")
        return A, b, diag_sys

    A, b = result  # type: ignore[misc]
    if A.shape != (N, N):
        raise ValueError(f"Assembly produced A shape {A.shape}, expected {(N, N)}")
    if b.shape != (N,):
        raise ValueError(f"Assembly produced b shape {b.shape}, expected {(N,)}")
    return A, b


def build_global_residual(
    u: np.ndarray,
    ctx: NonlinearContext,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build the global residual F(u) for one timestep.

    Parameters
    ----------
    u : ndarray
        Global unknown vector aligned with ctx.layout.
    ctx : NonlinearContext
        Timestep context containing cfg/grid/layout/state_old/props_old/dt.
    Notes
    -----
    This always uses state_guess = ctx.make_state(u) and recomputes props from state_guess.
    """
    u = np.asarray(u, dtype=np.float64)
    N = ctx.layout.n_dof()
    if u.shape != (N,):
        raise ValueError(f"Global unknown vector shape {u.shape} != ({N},)")

    cfg = ctx.cfg
    grid = ctx.grid_ref
    layout = ctx.layout
    state_old = ctx.state_old
    props_old = ctx.props_old
    dt = float(ctx.dt)
    sanitize_mode, penalty_value, penalty_scope = _get_penalty_cfg(cfg)

    try:
        state_guess = ctx.make_state(u)
    except Exception as exc:
        logger.warning("make_state(u) failed in residual_global: %s; fallback to state_old.", exc)
        state_guess = state_old

    t_min_cfg = float(getattr(getattr(cfg, "checks", None), "T_min", 1.0))
    if not np.isfinite(t_min_cfg) or t_min_cfg <= 0.0:
        t_min_cfg = 1.0

    state_props = state_guess.copy()
    Tg_raw_min = float(np.min(state_props.Tg)) if state_props.Tg.size else np.nan
    Tl_raw_min = float(np.min(state_props.Tl)) if state_props.Tl.size else np.nan
    Ts_raw = float(state_props.Ts)
    Tg_clamped = int(np.count_nonzero(state_props.Tg < t_min_cfg)) if state_props.Tg.size else 0
    Tl_clamped = int(np.count_nonzero(state_props.Tl < t_min_cfg)) if state_props.Tl.size else 0
    Ts_clamped = int(Ts_raw < t_min_cfg)
    if Tg_clamped:
        state_props.Tg = np.maximum(state_props.Tg, t_min_cfg)
    if Tl_clamped:
        state_props.Tl = np.maximum(state_props.Tl, t_min_cfg)
    if Ts_clamped:
        state_props.Ts = t_min_cfg

    if _SANITIZE_STRICT:
        try:
            _validate_state_before_residual(state_props, cfg, ctx=ctx, stage="pre_props")
        except Exception as exc:
            if sanitize_mode == "penalty":
                extra = {
                    "Ts_trial": float(state_props.Ts),
                    "stage": "pre_props",
                }
                return _penalty_residual(
                    u,
                    layout,
                    exc,
                    scale=penalty_value,
                    scope=penalty_scope,
                    ctx=ctx,
                    stage="pre_props",
                    extra=extra,
                )
            raise

    debug = os.environ.get("DROPLET_PETSC_DEBUG", "0") == "1"
    debug_once = os.environ.get("DROPLET_PETSC_DEBUG_ONCE", "1") == "1"
    strict_props = os.environ.get("DROPLET_STRICT_PROPS", "0") == "1"
    debug_rank = -1
    if debug:
        try:
            from mpi4py import MPI  # type: ignore

            debug_rank = int(MPI.COMM_WORLD.Get_rank())
        except Exception:
            debug_rank = -1

    def _dbg_log_Y(tag: str, state_dbg: State) -> None:
        if not debug:
            return
        key = f"_dbg_y_{tag}"
        if debug_once and ctx.meta.get(key):
            return
        ctx.meta[key] = True
        try:
            Yg = np.asarray(state_dbg.Yg, dtype=np.float64)
            if Yg.ndim != 2:
                logger.warning("[DBG][rank=%s] %s Yg shape=%r", debug_rank, tag, getattr(Yg, "shape", None))
                return
            cell_idx = 7
            if Yg.shape[1] <= cell_idx:
                logger.warning(
                    "[DBG][rank=%s] %s Yg cell=%d out of range (ncell=%d)",
                    debug_rank,
                    tag,
                    cell_idx,
                    int(Yg.shape[1]),
                )
                return
            Y = Yg[:, cell_idx]
            s = float(np.sum(Y)) if Y.size else 0.0
            mn = float(np.min(Y)) if Y.size else 0.0
            mx = float(np.max(Y)) if Y.size else 0.0
            idx = np.argsort(-Y)[:3] if Y.size else np.array([], dtype=np.int64)
            top = [(int(i), float(Y[i])) for i in idx]
            logger.warning(
                "[DBG][rank=%s] %s Y@cell%d sum=%.6g min=%.3e max=%.3e top=%s",
                debug_rank,
                tag,
                cell_idx,
                s,
                mn,
                mx,
                top,
            )
        except Exception as exc:
            logger.warning("[DBG][rank=%s] %s Y debug failed: %r", debug_rank, tag, exc)

    props_source = "props_old"
    props_extras = None
    try:
        _dbg_log_Y("AFTER_VEC2STATE_BEFORE_PROPS", state_props)
        props, props_extras = compute_props(cfg, grid, state_props)
        props_source = "state_guess_clamped" if (Tg_clamped or Tl_clamped or Ts_clamped) else "state_guess"
        _dbg_log_Y("AFTER_PROPS_AFTER_RESIDUAL", state_props)
        if _SANITIZE_STRICT:
            try:
                _validate_props_before_residual(props, ctx=ctx, stage="post_props")
            except Exception as exc:
                if sanitize_mode == "penalty":
                    extra = {"stage": "post_props"}
                    return _penalty_residual(
                        u,
                        layout,
                        exc,
                        scale=penalty_value,
                        scope=penalty_scope,
                        ctx=ctx,
                        stage="post_props",
                        extra=extra,
                    )
                raise
    except Exception as exc:
        if _SANITIZE_STRICT and isinstance(exc, ValueError):
            msg = str(exc)
            if msg.startswith("[pre_props]") or msg.startswith("[post_props]"):
                if sanitize_mode == "penalty":
                    return _penalty_residual(
                        u,
                        layout,
                        exc,
                        scale=penalty_value,
                        scope=penalty_scope,
                        ctx=ctx,
                        stage="sanitize",
                        extra={"stage": "sanitize"},
                    )
                raise
        if strict_props:
            raise
        logger.exception("compute_props failed in residual_global; attempting sanitize+retry.")
        if _SANITIZE_Y:
            try:
                state_sanitized, n_fix = _sanitize_state_Yg(state_props)
                if n_fix:
                    logger.warning("Sanitized Yg in %d gas cells for retry.", n_fix)
                props, props_extras = compute_props(cfg, grid, state_sanitized)
                props_source = "state_guess_sanitized"
                _dbg_log_Y("AFTER_PROPS_SANITIZED", state_sanitized)
            except Exception as exc2:
                logger.exception("compute_props failed after sanitize retry.")
                return _penalty_residual(
                    u,
                    layout,
                    exc2,
                    scale=penalty_value,
                    scope=penalty_scope,
                    ctx=ctx,
                    stage="props_retry",
                    extra={"stage": "props_retry"},
                )
        else:
            return _penalty_residual(
                u,
                layout,
                exc,
                scale=penalty_value,
                scope=penalty_scope,
                ctx=ctx,
                stage="props",
                extra={"stage": "props"},
            )

    eq_result = None
    eq_model = None
    eq_source = "none"
    phys = cfg.physics
    needs_eq = bool(getattr(phys, "include_mpp", False) and layout.has_block("mpp"))
    if needs_eq:
        cache = ctx.meta.get("eq_result_cache")
        eq_model = _get_or_build_eq_model(ctx, state_guess)
        # Initialize variables for diagnostic logging (avoid UnboundLocalError in exception handler)
        Ts_if = float("nan")
        Pg_if = float(getattr(cfg.initial, "P_inf", 101325.0))
        y_cond_sum = float("nan")
        psat_val = float("nan")
        Yl_shape = None
        Yg_shape = None
        try:
            il_if = grid.Nl - 1
            ig_if = 0
            Ts_if = float(state_props.Ts)
            Pg_if = float(getattr(cfg.initial, "P_inf", 101325.0))
            Yl_face = np.asarray(state_props.Yl[:, il_if], dtype=np.float64)
            Yg_face = np.asarray(state_props.Yg[:, ig_if], dtype=np.float64)
            Yl_shape = tuple(Yl_face.shape)
            Yg_shape = tuple(Yg_face.shape)
            # P4: Use interface_equilibrium to get full diagnostics including meta
            eq = interface_equilibrium(
                eq_model,
                Ts=Ts_if,
                Pg=Pg_if,
                Yl_face=Yl_face,
                Yg_face=Yg_face,
            )
            eq_result = {
                "Yg_eq": np.asarray(eq.Yg_eq),
                "y_cond": np.asarray(eq.y_cond),
                "psat": np.asarray(eq.psat),
                "meta": dict(eq.meta),  # P4: Preserve meta for source tracking
                "x_cond": np.asarray(eq.x_cond) if hasattr(eq, "x_cond") else np.asarray([]),
            }
            Yg_eq = eq_result["Yg_eq"]
            y_cond = eq_result["y_cond"]
            psat = eq_result["psat"]

            # P0.2: NaN/Inf sentinel - catch non-finite values immediately
            psat_arr = np.asarray(psat)
            y_cond_arr = np.asarray(y_cond)
            Yg_eq_arr = np.asarray(Yg_eq)
            if not np.isfinite(Ts_if):
                if is_root_rank():
                    logger.error(
                        "NONFINITE Ts_if=%.6g step=%s t=%s Pg_if=%.6g",
                        Ts_if, getattr(ctx, "step", "NA"), getattr(ctx, "t_old", "NA"), Pg_if
                    )
                raise ValueError(f"Non-finite Ts_if={Ts_if}")
            if not np.all(np.isfinite(psat_arr)):
                psat_bad = psat_arr[~np.isfinite(psat_arr)]
                if is_root_rank():
                    logger.error(
                        "NONFINITE psat step=%s t=%s Ts_if=%.6g bad_values=%s",
                        getattr(ctx, "step", "NA"), getattr(ctx, "t_old", "NA"), Ts_if, psat_bad
                    )
                raise ValueError(f"Non-finite psat at Ts_if={Ts_if}")
            if not np.all(np.isfinite(y_cond_arr)):
                y_cond_bad = y_cond_arr[~np.isfinite(y_cond_arr)]
                if is_root_rank():
                    logger.error(
                        "NONFINITE y_cond step=%s t=%s Ts_if=%.6g psat=%.6g bad_values=%s",
                        getattr(ctx, "step", "NA"), getattr(ctx, "t_old", "NA"), Ts_if,
                        psat_arr[0] if psat_arr.size > 0 else float("nan"), y_cond_bad
                    )
                raise ValueError(f"Non-finite y_cond at Ts_if={Ts_if}")
            if not np.all(np.isfinite(Yg_eq_arr)):
                Yg_eq_min = float(np.nanmin(Yg_eq_arr)) if Yg_eq_arr.size > 0 else float("nan")
                Yg_eq_max = float(np.nanmax(Yg_eq_arr)) if Yg_eq_arr.size > 0 else float("nan")
                if is_root_rank():
                    logger.error(
                        "NONFINITE Yg_eq step=%s t=%s Ts_if=%.6g psat=%.6g sum_y_cond=%.6g Yg_eq_min=%.6g Yg_eq_max=%.6g",
                        getattr(ctx, "step", "NA"), getattr(ctx, "t_old", "NA"), Ts_if,
                        psat_arr[0] if psat_arr.size > 0 else float("nan"),
                        float(np.sum(y_cond_arr)),
                        Yg_eq_min, Yg_eq_max
                    )
                raise ValueError(f"Non-finite Yg_eq at Ts_if={Ts_if}")

            # Extract diagnostics from successful computation
            y_cond_sum = float(np.sum(y_cond))
            psat_val = float(np.asarray(psat).reshape(-1)[0]) if np.size(psat) > 0 else float("nan")
            eq_source = "computed"
            ctx.meta["eq_result_cache"] = dict(eq_result)
            try:
                if isinstance(eq_result, dict) and eq_model is not None:
                    idxL = getattr(eq_model, "idx_cond_l", np.array([], dtype=int))
                    x_cond = np.asarray(eq_result.get("x_cond", []), dtype=np.float64)
                    psat_arr = np.asarray(eq_result.get("psat", []), dtype=np.float64)
                    if idxL.size and x_cond.size and psat_arr.size:
                        s_raw = float(np.sum(x_cond * psat_arr[idxL]) / max(Pg_if, 1.0e-30))
                    else:
                        s_raw = float("nan")
                    eq_result["meta"]["sum_x_cond_raw"] = s_raw
                    eq_result["meta"]["psat_over_P"] = s_raw
            except Exception:
                pass
            try:
                tbub_val = float(eq_result.get("meta", {}).get("Tbub", float("nan")))
                if np.isfinite(tbub_val):
                    ctx.meta["tbub_last"] = tbub_val
                    ctx.meta["Ts_guard_dT"] = float(eq_result.get("meta", {}).get("Ts_guard_dT", ctx.meta.get("Ts_guard_dT", 3.0)))
            except Exception:
                pass
            try:
                _update_iface_regime(ctx, eq_result, cfg)
            except Exception:
                pass
        except Exception as exc:
            exc_type = type(exc).__name__
            exc_msg = str(exc)
            # P1: Record failure reason to ctx.meta
            ctx.meta["eq_last_error"] = f"{exc_type}: {exc_msg}"
            if sanitize_mode == "penalty":
                extra = {
                    "stage": "eq",
                    "Ts_trial": Ts_if,
                    "Pg_trial": Pg_if,
                    "eq_exc_type": exc_type,
                    "eq_exc_msg": exc_msg,
                }
                return _penalty_residual(
                    u,
                    layout,
                    exc,
                    scale=penalty_value,
                    scope=penalty_scope,
                    ctx=ctx,
                    stage="eq",
                    extra=extra,
                )
            # P1: Fail-fast - no cache fallback
            raise InterfaceEquilibriumError(
                f"compute_interface_equilibrium failed at Ts={Ts_if:.6g}, Pg={Pg_if:.6g}"
            ) from exc
    else:
        eq_source = "disabled"

    result = build_transport_system(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        state_guess=state_guess,
        eq_model=eq_model,
        eq_result=eq_result,
        return_diag=True,
    )

    if isinstance(result, tuple) and len(result) == 3:
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

    res = A @ u - b

    res_norm_2 = float(np.linalg.norm(res))
    res_norm_inf = float(np.linalg.norm(res, ord=np.inf))

    diag: Dict[str, Any] = {
        "assembly": diag_sys,
        "residual_norm_2": res_norm_2,
        "residual_norm_inf": res_norm_inf,
        "u_min": float(np.min(u)) if u.size else np.nan,
        "u_max": float(np.max(u)) if u.size else np.nan,
    }

    diag["props"] = {"source": props_source}
    diag["eq_result"] = {"source": eq_source, "enabled": needs_eq}
    try:
        diag["props"]["rho_g_min"] = float(np.min(props.rho_g)) if props.rho_g.size else np.nan
        diag["props"]["rho_g_max"] = float(np.max(props.rho_g)) if props.rho_g.size else np.nan
        diag["props"]["state_Tg_min"] = float(np.min(state_guess.Tg)) if state_guess.Tg.size else np.nan
        diag["props"]["state_Tg_max"] = float(np.max(state_guess.Tg)) if state_guess.Tg.size else np.nan
        diag["props"]["T_min_used"] = float(t_min_cfg)
        diag["props"]["Tg_min_raw"] = Tg_raw_min
        diag["props"]["Tl_min_raw"] = Tl_raw_min
        diag["props"]["Ts_raw"] = Ts_raw
        diag["props"]["Tg_clamped_count"] = Tg_clamped
        diag["props"]["Tl_clamped_count"] = Tl_clamped
        diag["props"]["Ts_clamped"] = Ts_clamped
    except Exception:
        pass
    if isinstance(props_extras, dict):
        try:
            diag["props"]["extras_keys"] = list(props_extras.keys())
        except Exception:
            pass

    clamp_diag: Dict[str, Any] = {}
    tol_sumY = float(getattr(cfg.checks, "sumY_tol", 1e-12))
    if layout.has_block("Yg") and layout.gas_closure_index is not None:
        try:
            clamp_diag.update(
                _closure_clamp_diag(
                    state_guess.Yg,
                    layout.gas_closure_index,
                    tol=tol_sumY,
                    phase="gas",
                )
            )
        except Exception:
            logger.debug("Failed to compute gas closure clamp diagnostics.", exc_info=True)
    if layout.has_block("Yl") and layout.liq_closure_index is not None:
        try:
            clamp_diag.update(
                _closure_clamp_diag(
                    state_guess.Yl,
                    layout.liq_closure_index,
                    tol=tol_sumY,
                    phase="liq",
                )
            )
        except Exception:
            logger.debug("Failed to compute liquid closure clamp diagnostics.", exc_info=True)

    evap_diag = diag_sys.get("evaporation", {}) if isinstance(diag_sys, dict) else {}
    try:
        no_cond = bool(evap_diag.get("no_condensation_applied", False))
        deltaY_raw = float(evap_diag.get("DeltaY_raw", np.nan))
        deltaY_eff = float(evap_diag.get("DeltaY_eff", np.nan))
        clamp_diag["no_condensation_applied"] = no_cond
        clamp_diag["DeltaY_raw"] = deltaY_raw
        clamp_diag["DeltaY_eff"] = deltaY_eff
        if np.isfinite(deltaY_raw) and np.isfinite(deltaY_eff):
            clamp_diag["deltaY_min_applied"] = abs(deltaY_eff) > abs(deltaY_raw)
    except Exception:
        pass

    if clamp_diag:
        diag["clamp"] = clamp_diag

    if res.size:
        idx_max = int(np.argmax(np.abs(res)))
        entry = layout.entries[idx_max] if idx_max < len(layout.entries) else None
        diag["residual_argmax"] = {
            "idx": idx_max,
            "abs": float(abs(res[idx_max])),
            "value": float(res[idx_max]),
            "kind": getattr(entry, "kind", ""),
            "name": getattr(entry, "name", ""),
            "phase": getattr(entry, "phase", ""),
            "cell": getattr(entry, "cell", None),
            "spec": getattr(entry, "spec", None),
        }

    try:
        if layout.has_block("Ts"):
            diag["Ts_guess"] = float(u[layout.idx_Ts()])
        if layout.has_block("Rd"):
            diag["Rd_guess"] = float(u[layout.idx_Rd()])
    except Exception:
        logger.debug("Failed to extract Ts/Rd from u for diagnostics.", exc_info=True)

    if debug:
        try:
            key = "_dbg_residual_global_summary"
            if (not debug_once) or (not meta.get(key)):
                meta[key] = True
                argmax = diag.get("residual_argmax", {}) or {}
                clamp_info = diag.get("clamp", {}) or {}
                logger.warning(
                    "[DBG][rank=%s] residual_global: ||F||_inf=%.3e ||F||_2=%.3e "
                    "u[min,max]=[%.3e, %.3e] props_src=%s eq_src=%s "
                    "clamp=%s argmax={idx=%s, abs=%.3e, kind=%s, phase=%s, cell=%s, spec=%s}",
                    debug_rank,
                    res_norm_inf,
                    res_norm_2,
                    diag.get("u_min", np.nan),
                    diag.get("u_max", np.nan),
                    props_source,
                    eq_source,
                    {k: clamp_info.get(k) for k in sorted(clamp_info.keys())},
                    argmax.get("idx"),
                    argmax.get("abs", 0.0) or 0.0,
                    argmax.get("kind", ""),
                    argmax.get("phase", ""),
                    argmax.get("cell", None),
                    argmax.get("spec", None),
                )
        except Exception:
            logger.debug("Failed to emit residual_global debug summary.", exc_info=True)

    return res, diag


def residual_only(u: np.ndarray, ctx: NonlinearContext) -> np.ndarray:
    """
    Wrapper for solvers that only accept residuals.
    """
    res, _ = build_global_residual(u, ctx)
    return res


def residual_only_owned_rows(
    u: np.ndarray,
    ctx: NonlinearContext,
    ownership_range: tuple[int, int],
) -> np.ndarray:
    """
    Compute residual F(u) and return the slice corresponding to owned rows
    in [rstart, rend).

    Note: this is a simple baseline implementation:
    - uses global residual_only(u, ctx)
    - then slices by global row indices.
    Later可以换成真正的 residual_local kernel，而接口保持不变。
    """
    rstart, rend = ownership_range
    rstart = int(rstart)
    rend = int(rend)
    if rstart < 0 or rend < 0 or rstart > rend:
        raise ValueError(f"Invalid ownership_range={ownership_range}.")

    res_full = residual_only(u, ctx)
    if res_full.ndim != 1:
        res_full = res_full.ravel()

    N = res_full.size
    if rend > N:
        raise ValueError(
            f"ownership_range end {rend} exceeds residual size {N}."
        )

    return res_full[rstart:rend].copy()


def residual_petsc(
    mgr,
    ld,
    ctx: NonlinearContext,
    Xg,
    *,
    Fg=None,
):
    """
    Assemble residual into a PETSc Vec (DMComposite ordering) using local ghosted arrays.
    """
    from assembly.residual_local import ResidualLocalCtx, pack_local_to_layout, scatter_layout_to_local
    from parallel.dm_manager import global_to_local, local_to_global_add

    comm = mgr.comm
    rank = int(comm.getRank())

    Xl_liq, Xl_gas, Xl_if = global_to_local(mgr, Xg)
    aXl_liq = mgr.dm_liq.getVecArray(Xl_liq)
    aXl_gas = mgr.dm_gas.getVecArray(Xl_gas)
    aXl_if = Xl_if.getArray()

    ctx_local = ResidualLocalCtx(layout=ctx.layout, ld=ld)
    u_local = pack_local_to_layout(ctx_local, aXl_liq, aXl_gas, aXl_if, rank=rank)

    if comm.getSize() == 1:
        u_global = u_local
    else:
        try:
            from mpi4py import MPI

            mpicomm = comm.tompi4py()
            u_global = mpicomm.allreduce(u_local, op=MPI.SUM)
        except Exception as exc:
            raise RuntimeError("mpi4py is required for residual_petsc in MPI mode.") from exc

    res_global, _diag = build_global_residual(u_global, ctx)

    Fl_liq = mgr.dm_liq.createLocalVec()
    Fl_gas = mgr.dm_gas.createLocalVec()
    Fl_if = mgr.dm_if.createLocalVec()
    Fl_liq.set(0.0)
    Fl_gas.set(0.0)
    Fl_if.set(0.0)

    aFl_liq = mgr.dm_liq.getVecArray(Fl_liq)
    aFl_gas = mgr.dm_gas.getVecArray(Fl_gas)
    aFl_if = Fl_if.getArray()

    scatter_layout_to_local(
        ctx_local,
        res_global,
        aFl_liq,
        aFl_gas,
        aFl_if,
        rank=rank,
        owned_only=True,
    )

    Fg_out = local_to_global_add(mgr, Fl_liq, Fl_gas, Fl_if)
    if Fg is not None:
        try:
            Fg_out.copy(Fg)
        except Exception:
            Fg.set(0.0)
            Fg.axpy(1.0, Fg_out)
        return Fg
    return Fg_out


# ============================================================================
# TEST-ONLY: Smoke test helper for P2 boiling guard verification
# ============================================================================


def _smoke_call_interface_equilibrium(cfg, layout, grid, state_props, eq_model=None):
    """
    Test-only helper to call interface equilibrium from residual path.

    P2 Smoke Test: Verifies boiling guard is applied in actual residual assembly path.

    Args:
        cfg: CaseConfig with physics.include_mpp=True
        layout: UnknownLayout with mpp block
        grid: Grid1D with Nl, Ng
        state_props: State with Yl, Yg, Ts
        eq_model: Optional EquilibriumModel (will build if None)

    Returns:
        EquilibriumResult with meta containing ys_cap_hit, y_bg_total, etc.

    Raises:
        InterfaceEquilibriumError: If equilibrium computation fails
    """
    # Check needs_eq condition (same as residual_global logic)
    phys = cfg.physics
    needs_eq = bool(getattr(phys, "include_mpp", False) and layout.has_block("mpp"))

    if not needs_eq:
        raise ValueError(
            "Smoke test requires include_mpp=True and layout with 'mpp' block"
        )

    # Build equilibrium model if not provided
    if eq_model is None:
        Ns_g = len(cfg.species.gas_species_full)
        Ns_l = len(cfg.species.liq_species)
        M_g = _build_molar_mass_from_cfg(
            list(cfg.species.gas_species_full), Ns_g, cfg.species.molar_mass
        )
        M_l = _build_molar_mass_from_cfg(
            list(cfg.species.liq_species), Ns_l, cfg.species.molar_mass
        )
        eq_model = build_equilibrium_model(cfg, Ns_g, Ns_l, M_g, M_l)

    # Extract interface conditions (same as residual_global)
    il_if = grid.Nl - 1
    ig_if = 0
    Ts_if = float(state_props.Ts)
    Pg_if = float(getattr(cfg.initial, "P_inf", 101325.0))
    Yl_face = np.asarray(state_props.Yl[:, il_if], dtype=np.float64)
    Yg_face = np.asarray(state_props.Yg[:, ig_if], dtype=np.float64)

    # Call interface_equilibrium (not legacy wrapper)
    # This returns EquilibriumResult with meta containing ys_cap_hit, y_bg_total
    eq_result = interface_equilibrium(
        eq_model,
        Ts=Ts_if,
        Pg=Pg_if,
        Yl_face=Yl_face,
        Yg_face=Yg_face,
    )

    return eq_result
