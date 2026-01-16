"""
Energy flux helpers (Step 14.2 MVP).

Conventions (must match CaseConventions):
- radial_normal = "+er" (outward)
- flux_sign     = "outward_positive"
- conductive heat flux: q_cond = -k * dT/dr
- diffusive enthalpy flux: q_diff = sum_k h_k * J_k (single species: h * J)

This module only provides diagnostic/helper computations:
- No residual assembly
- No state mutation
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from core.types import CaseConfig

FloatArray = np.ndarray


def _check_conventions(cfg: CaseConfig) -> None:
    conv = cfg.conventions
    if conv.radial_normal != "+er":
        raise ValueError("energy_flux assumes radial_normal='+er'")
    if conv.flux_sign != "outward_positive":
        raise ValueError("energy_flux assumes flux_sign='outward_positive'")
    if conv.heat_flux_def != "q=-k*dTdr":
        raise ValueError("energy_flux assumes heat_flux_def='q=-k*dTdr'")


def split_energy_flux_cond_diff_single(
    cfg: CaseConfig,
    k_face: float,
    dTdr_face: float,
    h_face: float,
    J_face: float,
) -> Tuple[float, float, float]:
    """
    Split energy flux (single-species / scalar J) on a face.

    Returns
    -------
    q_total, q_cond, q_diff  (all outward-positive, W/m^2)

    Definitions
    -----------
    q_cond = -k_face * dTdr_face
    q_diff = h_face * J_face
    q_total = q_cond + q_diff
    """
    _check_conventions(cfg)

    kf = float(k_face)
    dT = float(dTdr_face)
    hf = float(h_face)
    Jf = float(J_face)

    if not np.isfinite([kf, dT, hf, Jf]).all():
        raise ValueError("Non-finite input to split_energy_flux_cond_diff_single")

    q_cond = -kf * dT
    q_diff = hf * Jf
    q_total = q_cond + q_diff

    if not np.isfinite([q_cond, q_diff, q_total]).all():
        raise ValueError("Non-finite output from split_energy_flux_cond_diff_single")

    return q_total, q_cond, q_diff


def split_energy_flux_cond_diff_multispecies(
    cfg: CaseConfig,
    k_face: float,
    dTdr_face: float,
    h_k_face: FloatArray,
    J_k_face: FloatArray,
) -> Tuple[float, float, float]:
    """
    Split energy flux (multi-species) on a face.

    q_diff = sum_k h_k * J_k  (dot product)

    h_k_face : (Ns,)
    J_k_face : (Ns,)
    """
    _check_conventions(cfg)

    kf = float(k_face)
    dT = float(dTdr_face)

    hk = np.asarray(h_k_face, dtype=np.float64).reshape(-1)
    Jk = np.asarray(J_k_face, dtype=np.float64).reshape(-1)
    if hk.shape != Jk.shape:
        raise ValueError(f"h_k_face shape {hk.shape} != J_k_face shape {Jk.shape}")
    if hk.size == 0:
        raise ValueError("Empty h_k_face/J_k_face")

    if not np.isfinite(kf) or not np.isfinite(dT):
        raise ValueError("Non-finite k_face/dTdr_face in split_energy_flux_cond_diff_multispecies")
    if np.any(~np.isfinite(hk)) or np.any(~np.isfinite(Jk)):
        raise ValueError("Non-finite hk/Jk in split_energy_flux_cond_diff_multispecies")

    q_cond = -kf * dT
    q_diff = float(np.dot(hk, Jk))
    q_total = q_cond + q_diff

    if not np.isfinite([q_cond, q_diff, q_total]).all():
        raise ValueError("Non-finite output from split_energy_flux_cond_diff_multispecies")

    return q_total, q_cond, q_diff
