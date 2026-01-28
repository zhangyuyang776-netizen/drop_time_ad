from __future__ import annotations

from typing import Tuple

import numpy as np


def clip_temperature(
    T: float,
    Tmin: float,
    Tmax: float,
    *,
    delta: float = 1.0e-6,
) -> Tuple[float, bool, bool]:
    t_low = float(Tmin) + delta
    t_high = float(Tmax) - delta
    if t_low >= t_high:
        t_low = float(Tmin)
        t_high = float(Tmax)
    clamped_low = T < t_low
    clamped_high = T > t_high
    return min(max(T, t_low), t_high), clamped_low, clamped_high


def apply_condensable_clamp(
    y_cond: np.ndarray,
    *,
    eps_bg: float,
) -> Tuple[np.ndarray, bool]:
    y = np.asarray(y_cond, dtype=np.float64).copy()
    y_sum = float(np.sum(y))
    boundary = 1.0 - float(eps_bg)
    if y_sum > boundary:
        scale = boundary / max(y_sum, 1.0e-300)
        y *= scale
        return y, True
    return y, False


def smooth_cap(
    T: float,
    T_upper: float,
    *,
    width: float = 0.5,
) -> float:
    """
    Smooth upper cap for temperature to avoid hard min() kinks.

    This implementation is a C1 (first-derivative continuous) *exact* cap:

    - For T <= T_upper - width: returns T (identity)
    - For T >= T_upper: returns T_upper (hard cap)
    - In between: uses a cubic Hermite blend so that:
        * value matches identity at T_upper - width
        * value matches cap at T_upper
        * slope transitions smoothly from 1 -> 0

    Compared to softplus-style caps, this avoids the downward bias at T == T_upper
    (softplus caps yield T_upper - width*log(2) at the join).
    """
    T = float(T)
    T_upper = float(T_upper)
    w = float(width)
    if not np.isfinite(T) or not np.isfinite(T_upper):
        return T
    if w <= 0.0:
        return min(T, T_upper)

    # Fast paths
    if T <= T_upper - w:
        return T
    if T >= T_upper:
        return T_upper

    # Transition region: x in (0, 1)
    x = (T - (T_upper - w)) / w
    # Cubic Hermite with f(0)=0, f'(0)=1, f(1)=1, f'(1)=0
    f = (-x * x * x) + (x * x) + x
    return (T_upper - w) + w * float(f)
