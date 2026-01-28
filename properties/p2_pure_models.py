from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np


def _as_array(T: Any) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(T, dtype=np.float64)
    return arr, arr.ndim == 0


def _return(arr: np.ndarray, scalar: bool) -> Any:
    if scalar:
        return float(arr)
    return arr


def _require_coeffs(entry: Mapping[str, Any], names: Sequence[str], *, prop: str, model: str) -> Tuple[float, ...]:
    coeffs = entry.get("coeffs")
    if isinstance(coeffs, Mapping):
        values = []
        for name in names:
            if name not in coeffs:
                raise ValueError(f"Missing coeff '{name}' for {prop} model '{model}'.")
            val = coeffs[name]
            if val is None:
                raise ValueError(f"Null coeff '{name}' for {prop} model '{model}'.")
            values.append(float(val))
        return tuple(values)
    if not isinstance(coeffs, (list, tuple)):
        raise ValueError(f"Invalid coeffs for {prop} model '{model}': expected list/dict.")
    if len(coeffs) < len(names):
        raise ValueError(f"Not enough coeffs for {prop} model '{model}': need {len(names)}.")
    values = []
    for idx, name in enumerate(names):
        val = coeffs[idx]
        if val is None:
            raise ValueError(f"Null coeff '{name}' for {prop} model '{model}'.")
        values.append(float(val))
    return tuple(values)


def _ensure_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid {name}: {value!r}")


def psat(T: Any, entry: Mapping[str, Any], *, Tc: float, Pc: float | None) -> Any:
    model = str(entry.get("model", "")).strip().lower()
    if model in ("wagner25", "wagner"):
        if Pc is None:
            raise ValueError("Pc is required for wagner25 psat model.")
        return _psat_wagner25(T, entry, Tc=Tc, Pc=Pc)
    if model in ("yaws_ln", "yaws_lnp", "yaws"):
        return _psat_yaws_ln(T, entry)
    if model in ("antoine_log10_pa", "antoine_log10", "antoine"):
        return _psat_antoine_log10_pa(T, entry)
    raise ValueError(f"Unsupported psat model '{model}'.")


def hvap(T: Any, entry: Mapping[str, Any], *, Tc: float) -> Any:
    model = str(entry.get("model", "")).strip().lower()
    if model == "watson":
        return _hvap_watson(T, entry, Tc=Tc)
    raise ValueError(f"Unsupported hvap model '{model}'.")


def rho_l(T: Any, entry: Mapping[str, Any], *, Tc: float) -> Any:
    model = str(entry.get("model", "")).strip().lower()
    if model == "rackett":
        return _rho_rackett(T, entry, Tc=Tc)
    if model == "poly":
        return _poly_eval(T, entry, order=3)
    raise ValueError(f"Unsupported rho model '{model}'.")


def cp_l(T: Any, entry: Mapping[str, Any]) -> Any:
    model = str(entry.get("model", "")).strip().lower()
    if model == "poly4":
        return _poly_eval(T, entry, order=4)
    if model == "poly":
        return _poly_eval(T, entry, order=3)
    raise ValueError(f"Unsupported cp model '{model}'.")


def k_l(T: Any, entry: Mapping[str, Any]) -> Any:
    model = str(entry.get("model", "")).strip().lower()
    if model == "poly2":
        return _poly_eval(T, entry, order=2)
    if model == "poly":
        return _poly_eval(T, entry, order=3)
    raise ValueError(f"Unsupported k model '{model}'.")


def mu_l(T: Any, entry: Mapping[str, Any]) -> Any:
    model = str(entry.get("model", "")).strip().lower()
    if model in ("yaws_ln", "yaws"):
        return _mu_yaws_ln(T, entry)
    if model == "andrade":
        return _mu_andrade(T, entry)
    raise ValueError(f"Unsupported mu model '{model}'.")


def sigma_l(T: Any, entry: Mapping[str, Any], *, Tc: float) -> Any:
    model = str(entry.get("model", "")).strip().lower()
    if model in ("crit_power", "critical_power"):
        return _sigma_crit_power(T, entry, Tc=Tc)
    raise ValueError(f"Unsupported sigma model '{model}'.")


def h_l(T: Any, cp_entry: Mapping[str, Any], *, T_ref: float = 298.15, h_ref: float = 0.0) -> Any:
    model = str(cp_entry.get("model", "")).strip().lower()
    if model == "poly4":
        coeffs = _require_coeffs(cp_entry, ("a0", "a1", "a2", "a3", "a4"), prop="cp", model=model)
        return _poly_integral(T, coeffs, T_ref=T_ref, h_ref=h_ref)
    if model == "poly":
        coeffs = _require_coeffs(cp_entry, ("a0", "a1", "a2", "a3"), prop="cp", model=model)
        return _poly_integral(T, coeffs, T_ref=T_ref, h_ref=h_ref)
    raise ValueError(f"Unsupported cp model '{model}' for h_l.")


def _psat_wagner25(T: Any, entry: Mapping[str, Any], *, Tc: float, Pc: float) -> Any:
    _ensure_positive(Tc, "Tc")
    _ensure_positive(Pc, "Pc")
    a, b, c, d = _require_coeffs(entry, ("a", "b", "c", "d"), prop="psat", model="wagner25")
    Tarr, scalar = _as_array(T)
    if np.any(Tarr <= 0.0):
        raise ValueError("Temperature must be positive for psat.")
    Tr = Tarr / float(Tc)
    tau = 1.0 - Tr
    term = a * tau + b * np.power(tau, 1.5) + c * np.power(tau, 3.0) + d * np.power(tau, 6.0)
    ln_pr = term / Tr
    psat_val = float(Pc) * np.exp(ln_pr)
    return _return(psat_val, scalar)


def _psat_yaws_ln(T: Any, entry: Mapping[str, Any]) -> Any:
    A, B, C, D, E = _require_coeffs(entry, ("A", "B", "C", "D", "E"), prop="psat", model="yaws_ln")
    Tarr, scalar = _as_array(T)
    if np.any(Tarr <= 0.0):
        raise ValueError("Temperature must be positive for psat.")
    ln_p = A + B / Tarr + C * np.log(Tarr) + D * np.power(Tarr, E)
    psat_val = np.exp(ln_p)
    return _return(psat_val, scalar)


def _psat_antoine_log10_pa(T: Any, entry: Mapping[str, Any]) -> Any:
    A, B, C = _require_coeffs(entry, ("A", "B", "C"), prop="psat", model="antoine_log10_Pa")
    Tarr, scalar = _as_array(T)
    if np.any(Tarr + C == 0.0):
        raise ValueError("Invalid Antoine coefficients: T + C equals zero.")
    log10_p = A - B / (Tarr + C)
    psat_val = np.power(10.0, log10_p)
    return _return(psat_val, scalar)


def _hvap_watson(T: Any, entry: Mapping[str, Any], *, Tc: float) -> Any:
    _ensure_positive(Tc, "Tc")
    Tref = entry.get("Tref", entry.get("T_ref"))
    Hvap_ref = entry.get("Hvap_Tref", entry.get("hvap_ref"))
    exponent = entry.get("n", entry.get("exponent"))
    if Tref is None or Hvap_ref is None or exponent is None:
        raise ValueError("Watson hvap requires Tref, Hvap_Tref, and n.")
    Tref = float(Tref)
    Hvap_ref = float(Hvap_ref)
    exponent = float(exponent)
    _ensure_positive(Tref, "Tref")
    _ensure_positive(Hvap_ref, "Hvap_Tref")

    Tarr, scalar = _as_array(T)
    term_num = 1.0 - Tarr / float(Tc)
    term_den = 1.0 - Tref / float(Tc)
    if np.any(term_den <= 0.0):
        raise ValueError("Invalid Watson reference: Tref >= Tc.")
    hvap_val = Hvap_ref * np.power(term_num / term_den, exponent)
    return _return(hvap_val, scalar)


def _rho_rackett(T: Any, entry: Mapping[str, Any], *, Tc: float) -> Any:
    _ensure_positive(Tc, "Tc")
    rho_c, z_ra = _require_coeffs(entry, ("rho_c", "Z_RA"), prop="rho", model="rackett")
    _ensure_positive(rho_c, "rho_c")
    _ensure_positive(z_ra, "Z_RA")
    Tarr, scalar = _as_array(T)
    Tr = Tarr / float(Tc)
    exponent = -np.power(1.0 - Tr, 2.0 / 7.0)
    rho_val = rho_c * np.power(z_ra, exponent)
    return _return(rho_val, scalar)


def _poly_eval(T: Any, entry: Mapping[str, Any], *, order: int) -> Any:
    if order < 0:
        raise ValueError("Order must be non-negative.")
    names = tuple(f"a{i}" for i in range(order + 1))
    coeffs = _require_coeffs(entry, names, prop="poly", model=f"poly{order}")
    Tarr, scalar = _as_array(T)
    val = np.zeros_like(Tarr, dtype=np.float64)
    for power, coef in enumerate(coeffs):
        val = val + coef * np.power(Tarr, power)
    return _return(val, scalar)


def _poly_integral(T: Any, coeffs: Sequence[float], *, T_ref: float, h_ref: float) -> Any:
    Tarr, scalar = _as_array(T)
    if not np.isfinite(T_ref):
        raise ValueError("Invalid T_ref for h_l.")
    val = np.zeros_like(Tarr, dtype=np.float64) + float(h_ref)
    for power, coef in enumerate(coeffs):
        expn = power + 1
        val = val + coef / expn * (np.power(Tarr, expn) - float(T_ref) ** expn)
    return _return(val, scalar)


def _mu_yaws_ln(T: Any, entry: Mapping[str, Any]) -> Any:
    A, B, C, D, E = _require_coeffs(entry, ("A", "B", "C", "D", "E"), prop="mu", model="yaws_ln")
    Tarr, scalar = _as_array(T)
    if np.any(Tarr <= 0.0):
        raise ValueError("Temperature must be positive for mu.")
    ln_mu = A + B / Tarr + C * np.log(Tarr) + D * np.power(Tarr, E)
    mu_val = np.exp(ln_mu)
    return _return(mu_val, scalar)


def _mu_andrade(T: Any, entry: Mapping[str, Any]) -> Any:
    A, B, C = _require_coeffs(entry, ("A", "B", "C"), prop="mu", model="andrade")
    Tarr, scalar = _as_array(T)
    if np.any(Tarr <= 0.0):
        raise ValueError("Temperature must be positive for mu.")
    ln_mu = A + B / Tarr + C * np.log(Tarr)
    mu_val = np.exp(ln_mu)
    return _return(mu_val, scalar)


def _sigma_crit_power(T: Any, entry: Mapping[str, Any], *, Tc: float) -> Any:
    _ensure_positive(Tc, "Tc")
    A, n = _require_coeffs(entry, ("A", "n"), prop="sigma", model="crit_power")
    _ensure_positive(A, "A")
    _ensure_positive(n, "n")
    Tarr, scalar = _as_array(T)
    term = 1.0 - Tarr / float(Tc)
    sigma_val = A * np.power(term, n)
    return _return(sigma_val, scalar)
