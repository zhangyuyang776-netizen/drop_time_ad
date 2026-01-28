#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _get_eq_guard_dt(cfg, default: float = 3.0) -> float:
    try:
        eq_cfg = getattr(getattr(getattr(cfg, "physics", None), "interface", None), "equilibrium", None)
        if eq_cfg is not None:
            val = float(getattr(eq_cfg, "Ts_guard_dT", default))
            if np.isfinite(val) and val > 0.0:
                return val
    except Exception:
        pass
    return float(default)


def _get_eq_sat_eps(cfg, default: float = 0.01) -> float:
    try:
        eq_cfg = getattr(getattr(getattr(cfg, "physics", None), "interface", None), "equilibrium", None)
        if eq_cfg is not None:
            val = float(getattr(eq_cfg, "Ts_sat_eps_K", default))
            if np.isfinite(val) and val >= 0.0:
                return val
    except Exception:
        pass
    return float(default)


def estimate_ts_upper(
    ctx,
    *,
    mode: str = "tbub_last",
    guard_dT: float | None = None,
) -> float | None:
    """
    Estimate a conservative upper bound for Ts (physical units).

    mode:
      - "tbub_last": use last Tbub from ctx.meta
      - "tc_min":    use cached Tc_min from ctx.meta
    """
    guard = float(guard_dT) if guard_dT is not None else _get_eq_guard_dt(ctx.cfg)
    meta = getattr(ctx, "meta", {}) or {}

    mode = str(mode or "tbub_last").strip().lower()
    if mode in ("tbub_last", "tbub"):
        tbub = float(meta.get("tbub_last", float("nan")))
        if np.isfinite(tbub):
            return float(tbub) - guard

    if mode in ("tc_min", "tcmin"):
        tc_min = float(meta.get("tc_min", float("nan")))
        if np.isfinite(tc_min):
            return float(tc_min) - guard

    return None


def estimate_ts_hard(
    ctx,
    *,
    mode: str = "tbub_last",
    sat_eps: float | None = None,
) -> float | None:
    """
    Estimate a hard upper bound for Ts (physical units): Tbub - Ts_sat_eps_K.
    """
    eps = float(sat_eps) if sat_eps is not None else _get_eq_sat_eps(ctx.cfg)
    meta = getattr(ctx, "meta", {}) or {}

    mode = str(mode or "tbub_last").strip().lower()
    if mode in ("tbub_last", "tbub"):
        tbub = float(meta.get("tbub_last", float("nan")))
        if np.isfinite(tbub):
            return float(tbub) - eps

    if mode in ("tc_min", "tcmin"):
        tc_min = float(meta.get("tc_min", float("nan")))
        if np.isfinite(tc_min):
            return float(tc_min) - eps

    return None

def cap_lambda_by_ts(
    Ts: float,
    dTs: float,
    lambda_in: float,
    Ts_upper: float | None,
    *,
    alpha: float = 0.8,
) -> Tuple[float, Dict[str, float | bool]]:
    """
    Cap linesearch lambda so Ts_trial <= Ts_upper.
    Returns (lambda_out, info_dict).
    """
    lam_in = float(lambda_in)
    info: Dict[str, float | bool] = {
        "lambda_before": lam_in,
        "lambda_after": lam_in,
        "ts": float(Ts),
        "dts": float(dTs),
        "ts_upper": float(Ts_upper) if Ts_upper is not None else float("nan"),
        "capped": False,
    }

    if Ts_upper is None or not np.isfinite(Ts_upper):
        return lam_in, info
    if not np.isfinite(Ts) or not np.isfinite(dTs):
        return lam_in, info
    if dTs <= 0.0:
        return lam_in, info

    lam_max = float(alpha) * (float(Ts_upper) - float(Ts)) / float(dTs)
    if not np.isfinite(lam_max):
        return lam_in, info

    lam_max = max(0.0, min(1.0, lam_max))
    lam_out = min(lam_in, lam_max)
    info["lambda_after"] = lam_out
    info["capped"] = lam_out < lam_in - 1.0e-14
    return lam_out, info


def cap_ts_in_u(
    u_phys: np.ndarray,
    *,
    idx_Ts: int,
    Ts_upper: float | None,
) -> Tuple[np.ndarray, bool, float]:
    """
    Clamp Ts in the physical-space vector u to Ts_upper.
    Returns (u_new, capped, ts_new).
    """
    u_out = np.asarray(u_phys, dtype=np.float64).copy()
    if Ts_upper is None or not np.isfinite(Ts_upper):
        return u_out, False, float(u_out[idx_Ts])

    Ts_val = float(u_out[idx_Ts])
    if not np.isfinite(Ts_val):
        return u_out, False, Ts_val
    if Ts_val <= Ts_upper:
        return u_out, False, Ts_val

    u_out[idx_Ts] = float(Ts_upper)
    return u_out, True, float(u_out[idx_Ts])
