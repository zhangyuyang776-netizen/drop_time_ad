from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from properties.p2_mix_rules import mass_to_mole_fractions, sanitize_mass_fractions
from properties.p2_pure_models import hvap as hvap_model
from properties.p2_pure_models import psat as psat_model
from properties.p2_safeguards import clip_temperature, smooth_cap


EPS = 1.0e-30


def mass_to_mole(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    denom = float(np.sum(Y / np.maximum(M, EPS)))
    if denom <= EPS or not np.isfinite(denom):
        out = np.zeros_like(Y)
        if out.size:
            out[0] = 1.0
        return out
    return (Y / np.maximum(M, EPS)) / denom


def mole_to_mass(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    numer = X * np.maximum(M, EPS)
    denom = float(np.sum(numer))
    if denom <= EPS or not np.isfinite(denom):
        out = np.zeros_like(X)
        if out.size:
            out[0] = 1.0
        return out
    return numer / denom


def _prop_bounds(spec: Mapping[str, Any], prop: str) -> Tuple[float, float]:
    entry = spec[prop]
    if "Tmin" in entry and "Tmax" in entry:
        return float(entry["Tmin"]), float(entry["Tmax"])
    t_valid = spec.get("T_valid")
    if not t_valid or len(t_valid) != 2:
        raise ValueError(f"Missing T_valid for species '{spec.get('name', '?')}'.")
    return float(t_valid[0]), float(t_valid[1])


def bubble_point_T(
    P: float,
    x_l: np.ndarray,
    psat_funcs: Sequence[Callable[[float], float]],
    T_lo: float,
    T_hi: float,
    *,
    tol: float = 1.0e-6,
    max_iter: int = 80,
) -> Tuple[float, Dict[str, Any]]:
    if not np.isfinite(P) or P <= 0.0:
        raise ValueError(f"Invalid pressure P={P}.")
    if len(psat_funcs) != len(x_l):
        raise ValueError("psat_funcs length must match x_l length.")

    def f(T: float) -> float:
        return float(np.sum([x_l[i] * psat_funcs[i](T) for i in range(len(x_l))]) - P)

    f_lo = f(T_lo)
    f_hi = f(T_hi)
    meta: Dict[str, Any] = {"f_lo": f_lo, "f_hi": f_hi, "bracketed": False}
    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        raise ValueError("Non-finite bubble point bracket values.")
    if f_lo > 0.0 or f_hi < 0.0:
        meta["bracketed"] = False
        return float("nan"), meta

    meta["bracketed"] = True
    a = float(T_lo)
    b = float(T_hi)
    fa = f_lo
    fb = f_hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if not np.isfinite(fm):
            raise ValueError("Non-finite bubble point function value.")
        if abs(fm) <= tol or abs(b - a) <= tol:
            return mid, meta
        if fm > 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b), meta


def _build_psat_funcs(
    specs: Sequence[Mapping[str, Any]],
) -> List[Callable[[float], float]]:
    funcs: List[Callable[[float], float]] = []
    for spec in specs:
        Tc = float(spec["Tc"])
        Pc = spec.get("Pc")

        def _fn(T: float, spec=spec, Tc=Tc, Pc=Pc) -> float:
            return float(psat_model(T, spec["psat"], Tc=Tc, Pc=Pc))

        funcs.append(_fn)
    return funcs


def _psat_vec(T: float, specs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    vals = []
    for spec in specs:
        Tc = float(spec["Tc"])
        Pc = spec.get("Pc")
        val = float(psat_model(T, spec["psat"], Tc=Tc, Pc=Pc))
        if not np.isfinite(val) or val < 0.0:
            raise ValueError("Invalid psat value.")
        vals.append(val)
    return np.asarray(vals, dtype=np.float64)


def _hvap_vec(T: float, specs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    vals = []
    for spec in specs:
        Tc = float(spec["Tc"])
        val = float(hvap_model(T, spec["hvap"], Tc=Tc))
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError("Invalid hvap value.")
        vals.append(val)
    return np.asarray(vals, dtype=np.float64)


def _project_condensables(
    y_cond_raw: np.ndarray,
    *,
    eps_bg: float,
) -> Tuple[np.ndarray, bool, float]:
    y_raw = np.asarray(y_cond_raw, dtype=np.float64)
    if not np.all(np.isfinite(y_raw)):
        raise ValueError("Non-finite y_cond values.")
    if np.any(y_raw < 0.0):
        raise ValueError("Negative y_cond values.")

    boundary = 1.0 - float(eps_bg)
    if y_raw.size == 1:
        raw = float(y_raw[0])
        y_val = min(max(raw, float(eps_bg)), boundary)
        scaled = not np.isclose(raw, y_val)
        scale = y_val / raw if raw > 0.0 else 0.0
        return np.asarray([y_val], dtype=np.float64), scaled, scale

    y_sum = float(np.sum(y_raw))
    if y_sum > boundary:
        scale = boundary / max(y_sum, 1.0e-300)
        return y_raw * scale, True, scale
    return y_raw.copy(), False, 1.0


def _project_background(
    y_bg_total: float,
    X_bg_norm: np.ndarray,
    mask_bg: np.ndarray,
    *,
    eps_bg: float,
) -> Tuple[np.ndarray, bool, bool]:
    y_bg = np.zeros_like(X_bg_norm, dtype=np.float64)
    if not np.any(mask_bg) or y_bg_total <= 0.0:
        return y_bg, False, False

    bg_vals = y_bg_total * X_bg_norm[mask_bg]
    n_bg = int(bg_vals.size)
    floor_applied = bool(np.any(bg_vals < float(eps_bg)))
    floor_relaxed = False
    if y_bg_total <= n_bg * float(eps_bg):
        bg_vals = np.full(n_bg, y_bg_total / n_bg, dtype=np.float64)
        floor_relaxed = True
        floor_applied = True
    else:
        bg_vals = np.maximum(bg_vals, float(eps_bg))
        bg_vals *= y_bg_total / float(np.sum(bg_vals))
    y_bg[mask_bg] = bg_vals
    return y_bg, floor_applied, floor_relaxed


def interface_equilibrium(
    P: float,
    Ts: float,
    Yl_face: Sequence[float],
    Yg_far: Sequence[float],
    *,
    liq_names: Sequence[str],
    gas_names: Sequence[str],
    idx_cond_l: Sequence[int],
    idx_cond_g: Sequence[int],
    M_l: Sequence[float],
    M_g: Sequence[float],
    db: Any,
    Ts_guard_dT: float = 3.0,
    Ts_guard_width_K: float = 0.5,
    Ts_sat_eps_K: float = 0.01,
    eps_bg: float = 1.0e-12,
    tbub_override: float | None = None,
    tbub_bracketed: bool | None = None,
) -> Dict[str, Any]:
    if not np.isfinite(P) or P <= 0.0:
        raise ValueError(f"Invalid pressure P={P}.")
    if not np.isfinite(Ts) or Ts <= 0.0:
        raise ValueError(f"Invalid surface temperature Ts={Ts}.")

    Yl, yl_meta = sanitize_mass_fractions(Yl_face)
    if len(liq_names) != Yl.shape[0]:
        raise ValueError("liq_names length does not match Yl_face length.")

    specs = [db.get_params(name) for name in liq_names]
    M_l_arr = np.asarray(M_l, dtype=np.float64)
    if M_l_arr.shape[0] != len(specs):
        raise ValueError("M_l length does not match liq_names length.")

    x_l, x_meta = mass_to_mole_fractions(Yl, M_l_arr)

    tmins = []
    tmaxs = []
    for spec in specs:
        Tmin, Tmax = _prop_bounds(spec, "psat")
        tmins.append(Tmin)
        tmaxs.append(Tmax)
    T_lo = float(max(tmins))
    T_hi = float(min(tmaxs))
    if T_lo >= T_hi:
        raise ValueError("Invalid psat temperature bounds for bubble point.")

    if tbub_override is None:
        psat_funcs = _build_psat_funcs(specs)
        Tbub, bub_meta = bubble_point_T(P, x_l, psat_funcs, T_lo, T_hi)
        bracketed = bool(bub_meta.get("bracketed", False))
    else:
        Tbub = float(tbub_override)
        bracketed = bool(tbub_bracketed) if tbub_bracketed is not None else np.isfinite(Tbub)
        bub_meta = {"bracketed": bracketed}

    fallback_reason = ""
    Ts_raw = float(Ts)
    Ts_eff = float(Ts_raw)
    Ts_guard = float("nan")
    Ts_hard = float("nan")
    guard_active = False
    if bracketed and np.isfinite(Tbub):
        Ts_guard = float(Tbub) - float(Ts_guard_dT)
        Ts_hard = float(Tbub) - float(Ts_sat_eps_K)
        Ts_eff = smooth_cap(Ts_eff, Ts_hard, width=float(Ts_guard_width_K))
        guard_active = bool(np.isfinite(Ts_raw) and np.isfinite(Ts_guard) and Ts_raw > Ts_guard - 1.0e-12)
    else:
        fallback_reason = "bubble_bracket_failed"

    Ts_eval, clamp_low, clamp_high = clip_temperature(Ts_eff, T_lo, T_hi)
    clamp_active = clamp_low or clamp_high
    if clamp_active:
        fallback_reason = (fallback_reason + ";Ts_clamped").strip(";")

    # --- Saturation check uses Ts_raw clipped just below Tbub (not Ts_guard) ---
    Ts_sat_target = float(Ts_raw)
    Ts_sat_eps = max(0.0, float(Ts_sat_eps_K))
    if bracketed and np.isfinite(Tbub):
        if np.isfinite(Ts_hard):
            Ts_sat_target = min(Ts_sat_target, float(Ts_hard))
        else:
            Ts_sat_target = min(Ts_sat_target, float(Tbub) - Ts_sat_eps)
    Ts_sat_eval, _sat_lo, _sat_hi = clip_temperature(Ts_sat_target, T_lo, T_hi)

    psat = _psat_vec(Ts_eval, specs)
    psat_sat = _psat_vec(Ts_sat_eval, specs)
    hvap = _hvap_vec(Ts_eval, specs)

    idxL = np.asarray(idx_cond_l, dtype=int)
    idxG = np.asarray(idx_cond_g, dtype=int)
    x_cond = x_l[idxL] if idxL.size else np.zeros(0, dtype=np.float64)
    p_partial = x_cond * psat[idxL] if idxL.size else np.zeros(0, dtype=np.float64)
    y_cond_raw = p_partial / float(P)
    y_cond_raw_sum = float(np.sum(y_cond_raw)) if y_cond_raw.size else 0.0
    if idxL.size:
        s_raw = float(np.sum(x_cond * psat_sat[idxL]) / float(P))
    else:
        s_raw = 0.0
    y_cond, clamp_hit, cond_scale_factor = _project_condensables(y_cond_raw, eps_bg=eps_bg)
    y_cond_sum = float(np.sum(y_cond))
    y_bg_total = 1.0 - y_cond_sum

    Yg_far_arr = np.asarray(Yg_far, dtype=np.float64)
    M_g_arr = np.asarray(M_g, dtype=np.float64)
    if Yg_far_arr.shape[0] != len(gas_names) or M_g_arr.shape[0] != len(gas_names):
        raise ValueError("Gas arrays must match gas_names length.")

    Xg_far = mass_to_mole(Yg_far_arr, M_g_arr)
    mask_bg = np.ones(len(gas_names), dtype=bool)
    mask_bg[idxG] = False
    X_bg = np.where(mask_bg, Xg_far, 0.0)
    s_bg = float(np.sum(X_bg))
    if s_bg <= EPS:
        raise ValueError("Farfield background sum is zero.")
    X_bg_norm = X_bg / s_bg
    y_bg, bg_floor_applied, bg_floor_relaxed = _project_background(
        y_bg_total, X_bg_norm, mask_bg, eps_bg=eps_bg
    )

    y_all = np.zeros(len(gas_names), dtype=np.float64)
    if idxG.size:
        y_all[idxG] = y_cond
    y_all[mask_bg] = y_bg[mask_bg]

    Yg_eq = mole_to_mass(y_all, M_g_arr)
    s_Y = float(np.sum(Yg_eq))
    if not np.isfinite(s_Y) or abs(s_Y - 1.0) > 1.0e-10:
        if s_Y > EPS:
            Yg_eq /= s_Y
        else:
            raise ValueError("Invalid Yg_eq normalization.")

    meta: Dict[str, Any] = {
        "Ts_eff": float(Ts_eval),
        "Tbub": float(Tbub) if np.isfinite(Tbub) else float("nan"),
        "Ts_upper": float(Ts_guard) if np.isfinite(Ts_guard) else float("nan"),
        "Ts_guard": float(Ts_guard) if np.isfinite(Ts_guard) else float("nan"),
        "Ts_hard": float(Ts_hard) if np.isfinite(Ts_hard) else float("nan"),
        "guard_active": bool(guard_active),
        "clamp_active": bool(clamp_hit),
        "Ts_guard_width_K": float(Ts_guard_width_K),
        "Ts_sat_eps_K": float(Ts_sat_eps_K),
        "Ts_raw": float(Ts_raw),
        "Ts_sat_clip": float(Ts_sat_eval),
        "psat_source": "p2db",
        "hvap_source": "p2db",
        "fallback_reason": fallback_reason,
        "y_cond_sum": y_cond_sum,
        "sum_y_cond_raw": y_cond_raw_sum,
        "sum_x_psat_over_P": float(s_raw),
        "psat_over_P": float(s_raw),
        "y_bg_total": y_bg_total,
        "cond_scaled": bool(clamp_hit),
        "cond_scale_factor": float(cond_scale_factor),
        "bg_floor_applied": bool(bg_floor_applied),
        "bg_floor_relaxed": bool(bg_floor_relaxed),
        "yl_meta": yl_meta,
        "x_meta": x_meta,
    }

    return {
        "Ts_eff": float(Ts_eval),
        "Tbub": float(Tbub) if np.isfinite(Tbub) else float("nan"),
        "y_cond": y_cond,
        "Yg_eq": Yg_eq,
        "psat": psat,
        "hvap": hvap,
        "x_l": x_l,
        "x_cond": x_cond,
        "meta": meta,
    }
