from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from properties.p2_pure_models import cp_l, h_l, k_l, mu_l, rho_l, sigma_l


def _clip_temperature(T: float, Tmin: float, Tmax: float, delta: float = 1.0e-6) -> Tuple[float, bool, bool]:
    t_low = float(Tmin) + delta
    t_high = float(Tmax) - delta
    if t_low >= t_high:
        t_low = float(Tmin)
        t_high = float(Tmax)
    clamped_low = T < t_low
    clamped_high = T > t_high
    return min(max(T, t_low), t_high), clamped_low, clamped_high


def sanitize_mass_fractions(Yl: Sequence[float]) -> Tuple[np.ndarray, Dict[str, Any]]:
    Y = np.asarray(Yl, dtype=np.float64).copy()
    meta: Dict[str, Any] = {"renormalized": False, "clamped": False, "zeroed_nonfinite": False}

    if not np.all(np.isfinite(Y)):
        Y[~np.isfinite(Y)] = 0.0
        meta["zeroed_nonfinite"] = True

    if np.any((Y < 0.0) | (Y > 1.0)):
        Y = np.clip(Y, 0.0, 1.0)
        meta["clamped"] = True

    s = float(np.sum(Y))
    if not np.isfinite(s) or s <= 0.0:
        Y[:] = 0.0
        if Y.size:
            Y[0] = 1.0
        meta["renormalized"] = True
        meta["fallback_reason"] = "yl_zero_or_invalid"
        return Y, meta

    if abs(s - 1.0) > 1.0e-12:
        Y /= s
        meta["renormalized"] = True

    return Y, meta


def mass_to_mole_fractions(Y: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if Y.shape != M.shape:
        raise ValueError("Y and M must have the same shape for mass->mole conversion.")
    denom = np.sum(Y / M)
    if not np.isfinite(denom) or denom <= 0.0:
        x = np.zeros_like(Y)
        if x.size:
            x[0] = 1.0
        meta["fallback_reason"] = "x_from_invalid_mass"
        return x, meta
    x = (Y / M) / denom
    return x, meta


def _eval_pure_props(
    T: float,
    spec: Mapping[str, Any],
    *,
    T_ref: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    Tmin, Tmax = spec["T_valid"]
    T_eval, clamp_low, clamp_high = _clip_temperature(T, Tmin, Tmax)
    meta: Dict[str, Any] = {}
    if clamp_low or clamp_high:
        meta["clamped_T"] = True
        meta["T_eval"] = T_eval

    Tc = float(spec["Tc"])

    rho = float(rho_l(T_eval, spec["rho"], Tc=Tc))
    cp = float(cp_l(T_eval, spec["cp"]))
    k = float(k_l(T_eval, spec["k"]))
    mu = float(mu_l(T_eval, spec["mu"]))
    sigma = float(sigma_l(T_eval, spec["sigma"], Tc=Tc))
    h = float(h_l(T_eval, spec["cp"], T_ref=T_ref, h_ref=0.0))

    props = {"rho": rho, "cp": cp, "k": k, "mu": mu, "sigma": sigma, "h": h}

    for key, val in props.items():
        if not np.isfinite(val):
            raise ValueError(f"Non-finite pure property '{key}' at T={T_eval}.")

    return props, meta


def eval_pure_props(
    T: float,
    spec: Mapping[str, Any],
    *,
    T_ref: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Public wrapper for pure-component properties."""
    return _eval_pure_props(T, spec, T_ref=T_ref)


def mix_props(
    T: float,
    P: float,
    Yl: Sequence[float],
    species_list: Sequence[str],
    db: Any,
    *,
    T_ref: float | None = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    del P  # P2: bulk liquid properties are T-only

    Y, y_meta = sanitize_mass_fractions(Yl)
    if len(species_list) != Y.shape[0]:
        raise ValueError("species_list length does not match Yl length.")

    t_ref = float(T_ref) if T_ref is not None else float(getattr(db, "meta", {}).get("T_ref", 298.15))

    M = np.zeros(len(species_list), dtype=np.float64)
    rho_pure = np.zeros_like(M)
    cp_pure = np.zeros_like(M)
    k_pure = np.zeros_like(M)
    mu_pure = np.zeros_like(M)
    sigma_pure = np.zeros_like(M)
    h_pure = np.zeros_like(M)

    fallback_reasons: List[str] = []
    for i, name in enumerate(species_list):
        spec = db.get_params(name)
        M[i] = float(spec["M"])
        props, meta = _eval_pure_props(T, spec, T_ref=t_ref)
        rho_pure[i] = props["rho"]
        cp_pure[i] = props["cp"]
        k_pure[i] = props["k"]
        mu_pure[i] = props["mu"]
        sigma_pure[i] = props["sigma"]
        h_pure[i] = props["h"]
        if meta.get("clamped_T"):
            fallback_reasons.append(f"{name}:T_clamped")

    x, x_meta = mass_to_mole_fractions(Y, M)
    if "fallback_reason" in x_meta:
        fallback_reasons.append(x_meta["fallback_reason"])

    inv_rho = float(np.sum(np.where(rho_pure > 0.0, Y / rho_pure, 0.0)))
    rho_mix = 1.0 / max(inv_rho, 1.0e-30)
    cp_mix = float(np.sum(Y * cp_pure))
    k_mix = float(np.sum(Y * k_pure))
    mu_safe = np.clip(mu_pure, 1.0e-30, None)
    mu_mix = float(np.exp(np.sum(x * np.log(mu_safe))))
    sigma_mix = float(np.sum(x * sigma_pure))
    h_mix = float(np.sum(Y * h_pure))

    for key, val in {
        "rho_l": rho_mix,
        "cp_l": cp_mix,
        "k_l": k_mix,
        "mu_l": mu_mix,
        "h_l": h_mix,
        "sigma_l": sigma_mix,
    }.items():
        if not np.isfinite(val):
            raise ValueError(f"Non-finite mixture property '{key}'.")

    meta: Dict[str, Any] = {
        "Yl": Y,
        "x_l": x,
        "Y_meta": y_meta,
    }
    if fallback_reasons:
        meta["fallback_reason"] = ";".join(fallback_reasons)

    props = {
        "rho_l": rho_mix,
        "cp_l": cp_mix,
        "k_l": k_mix,
        "mu_l": mu_mix,
        "h_l": h_mix,
        "sigma_l": sigma_mix,
    }
    return props, meta
