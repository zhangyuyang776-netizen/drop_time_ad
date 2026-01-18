"""
Interface phase-equilibrium utilities (Raoult + psat) with interface-noncondensables background fill.

Responsibilities:
- Build an EquilibriumModel from CaseConfig and species data (indices, molar masses, farfield Y).
- Compute interface-equilibrium gas composition Yg_eq given Ts, Pg, Yl_face, Yg_face.
- Provide psat helpers (CoolProp when available, Clausius fallback).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import CoolProp.CoolProp as CP
except Exception:  # pragma: no cover - environment dependent
    CP = None

from core.types import CaseConfig

FloatArray = np.ndarray
EPS = 1e-30
EPS_BG = 1e-5  # P2: Minimum background gas mole fraction (boiling guard)


def _apply_boiling_guard(y_cond: np.ndarray, eps_bg: float = EPS_BG) -> Tuple[np.ndarray, bool, float]:
    """
    Apply boiling guard to condensable mole fractions (pure function for testing).

    P2: Ensures background species have minimum mole fraction by scaling condensables.

    Args:
        y_cond: Condensable mole fractions (will be copied, not modified in-place)
        eps_bg: Minimum background mole fraction (default: EPS_BG)

    Returns:
        (y_cond_new, ys_cap_hit, y_bg_total):
            - y_cond_new: Scaled condensable mole fractions
            - ys_cap_hit: True if guard was triggered
            - y_bg_total: Resulting background total mole fraction
    """
    y_cond = np.asarray(y_cond, dtype=np.float64).copy()
    y_cond_sum = float(np.sum(y_cond))
    ys_cap_hit = False

    if y_cond_sum > 1.0 - eps_bg:
        # Scale condensables to leave minimum background fraction
        scale = (1.0 - eps_bg) / max(y_cond_sum, 1e-300)
        y_cond *= scale
        y_cond_sum = float(np.sum(y_cond))
        ys_cap_hit = True

    y_bg_total = max(1.0 - y_cond_sum, 0.0)
    return y_cond, ys_cap_hit, y_bg_total


# P3: Unified constraint flow functions


def _apply_condensable_guard(
    y_cond: np.ndarray, boundary: float = 1.0 - EPS_BG, smooth: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    P3: Apply condensable guard with unified constraint.

    Args:
        y_cond: Condensable mole fractions
        boundary: Maximum allowed sum of condensables (default: 1-EPS_BG)
        smooth: If True, use smooth guard; if False, use hard guard (default)

    Returns:
        (y_cond_guarded, meta):
            - y_cond_guarded: Scaled condensables if needed
            - meta: {"ys_cap_hit": bool, "y_cond_sum": float, "boundary": float, "guard_type": str}
    """
    y_cond = np.asarray(y_cond, dtype=np.float64).copy()
    y_cond_sum = float(np.sum(y_cond))
    ys_cap_hit = False
    guard_type = "smooth" if smooth else "hard"

    if smooth:
        # Smooth guard: gradually scale as approaching boundary
        # Start smooth scaling at 90% of boundary, fully scaled at boundary
        smooth_start = 0.9 * boundary
        if y_cond_sum > smooth_start:
            if y_cond_sum > boundary:
                # Beyond boundary: apply hard limit
                scale = boundary / max(y_cond_sum, 1e-300)
                ys_cap_hit = True
            else:
                # In transition region [smooth_start, boundary]: smooth scaling
                # Use smooth transition: scale = 1 - k*(x - smooth_start)/(boundary - smooth_start)
                # where k controls transition steepness (k=0.5 for moderate smoothness)
                transition = (y_cond_sum - smooth_start) / max(boundary - smooth_start, 1e-300)
                k = 0.5
                scale_reduction = k * transition
                target_sum = y_cond_sum * (1.0 - scale_reduction)
                scale = target_sum / max(y_cond_sum, 1e-300)
                ys_cap_hit = True if y_cond_sum > 0.95 * boundary else False
            y_cond *= scale
            y_cond_sum = float(np.sum(y_cond))
    else:
        # Hard guard: sharp cutoff at boundary (default, P3 behavior)
        if y_cond_sum > boundary:
            scale = boundary / max(y_cond_sum, 1e-300)
            y_cond *= scale
            y_cond_sum = float(np.sum(y_cond))
            ys_cap_hit = True

    meta = {
        "ys_cap_hit": ys_cap_hit,
        "y_cond_sum": y_cond_sum,
        "boundary": boundary,
        "guard_type": guard_type,
    }
    return y_cond, meta


def _fill_background(
    y_bg_total: float,
    X_farfield_background: np.ndarray,
    idx_bg: np.ndarray,
    Ns_g: int,
) -> np.ndarray:
    """
    P3: Fill background species using farfield composition (must include closure species).

    Args:
        y_bg_total: Total mole fraction to allocate to background
        X_farfield_background: Farfield mole fractions (full gas vector)
        idx_bg: Boolean mask for background species
        Ns_g: Total number of gas species

    Returns:
        y_bg: (Ns_g,) background mole fractions

    Raises:
        InterfaceEquilibriumError: If farfield background sum is zero
    """
    X_bg = np.where(idx_bg, X_farfield_background, 0.0)
    s_bg = float(np.sum(X_bg))

    if s_bg <= EPS:
        raise InterfaceEquilibriumError(
            "Farfield background sum is zero - cannot fill background species. "
            "Check initial.Yg configuration."
        )

    X_bg_norm = X_bg / s_bg
    y_bg = y_bg_total * X_bg_norm
    return y_bg


def _finalize_simplex(
    y_all: np.ndarray, tol_neg: float = 1e-14, tol_sum: float = 1e-10
) -> np.ndarray:
    """
    P3: Finalize mole fraction simplex with fail-fast (no hard clip).

    Light numerical correction only:
    - Tiny negative values (<tol_neg) set to 0, then renormalize
    - Larger errors raise InterfaceEquilibriumError

    Args:
        y_all: Mole fractions to finalize
        tol_neg: Tolerance for negative values
        tol_sum: Tolerance for sum deviation from 1.0

    Returns:
        y_all_final: Corrected mole fractions

    Raises:
        InterfaceEquilibriumError: If negative values too large or sum far from 1.0
    """
    y_all = np.asarray(y_all, dtype=np.float64).copy()
    y_min = float(np.min(y_all))
    y_sum = float(np.sum(y_all))

    # Check for large negative values (fail-fast)
    if y_min < -tol_neg:
        raise InterfaceEquilibriumError(
            f"Large negative mole fraction detected: min={y_min:.6e} (tol={tol_neg:.6e}). "
            "Refusing to clip - indicates numerical issue."
        )

    # Light correction: set tiny negatives to 0
    if y_min < 0:
        y_all = np.maximum(y_all, 0.0)
        y_sum = float(np.sum(y_all))

    # Check sum deviation (fail-fast if too large)
    if not np.isfinite(y_sum) or abs(y_sum - 1.0) > tol_sum:
        raise InterfaceEquilibriumError(
            f"Mole fraction sum far from 1.0: sum={y_sum:.15f} (tol={tol_sum:.6e}). "
            "Refusing to renormalize - indicates constraint violation."
        )

    # Renormalize if needed (only tiny deviation)
    if abs(y_sum - 1.0) > 1e-15 and y_sum > EPS:
        y_all /= y_sum

    return y_all


class InterfaceEquilibriumError(RuntimeError):
    """
    Exception raised when interface equilibrium computation fails.

    P1: This exception is raised to implement fail-fast behavior,
    preventing NaN propagation when equilibrium calculation fails.
    """
    pass


def mass_to_mole(Y: FloatArray, M: FloatArray) -> FloatArray:
    """Convert mass fractions to mole fractions."""
    Y = np.asarray(Y, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    denom = np.sum(Y / np.maximum(M, EPS))
    if denom <= 0.0:
        return np.zeros_like(Y)
    X = (Y / np.maximum(M, EPS)) / denom
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def mole_to_mass(X: FloatArray, M: FloatArray) -> FloatArray:
    """Convert mole fractions to mass fractions."""
    X = np.asarray(X, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    numer = X * np.maximum(M, EPS)
    denom = np.sum(numer)
    if denom <= 0.0:
        return np.zeros_like(X)
    Y = numer / denom
    return np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(slots=True)
class EquilibriumModel:
    method: str  # "raoult_psat"
    psat_model: str  # "auto" | "coolprop" | "clausius"

    gas_names: List[str]
    liq_names: List[str]
    idx_cond_l: np.ndarray
    idx_cond_g: np.ndarray

    M_g: np.ndarray
    M_l: np.ndarray

    Yg_farfield: np.ndarray
    Xg_farfield: np.ndarray

    cp_backend: str
    cp_fluids: List[str]

    T_ref: float = 298.15
    psat_ref: Dict[str, float] = field(default_factory=dict)

    # P4: Custom saturation support
    sat_source: str = "coolprop"  # "coolprop" | "custom"
    sat_model: Optional[Any] = None  # SaturationModelMM instance (avoid circular import)
    sat_db: Optional[Any] = None  # LiquidSatDB instance


@dataclass(slots=True)
class EquilibriumResult:
    """Full interface-equilibrium outputs for debugging/diagnostics."""

    Yg_eq: np.ndarray          # (Ns_g,) equilibrium gas mass fractions
    y_all: np.ndarray          # (Ns_g,) assembled gas mole fractions
    y_cond: np.ndarray         # (Nv,) condensable mole fractions (gas-side)
    psat: np.ndarray           # (Ns_l,) saturation pressures for all liquid species
    X_liq: np.ndarray          # (Ns_l,) liquid mole fractions
    x_cond: np.ndarray         # (Nv,) liquid mole fractions for condensables
    p_partial: np.ndarray      # (Nv,) partial pressures for condensables
    sum_partials: float        # total partial pressure of condensables
    y_bg_total: float          # total mole fraction allocated to background gases
    X_bg_norm: np.ndarray      # (Ns_g,) normalized background mole fractions (source)
    mask_bg: np.ndarray        # (Ns_g,) True for background (non-condensable) species
    meta: Dict[str, Any] = field(default_factory=dict)  # P0.1: source tracking metadata


def _psat_coolprop_single(fluid_label: str, T: float) -> Optional[float]:
    """Return psat [Pa] using CoolProp if available."""
    if CP is None:
        return None
    try:
        val = float(CP.PropsSI("P", "T", T, "Q", 0, fluid_label))
        if not np.isfinite(val) or val < 0.0:
            return None
        return val
    except Exception:
        return None


def _psat_clausius_single(fluid_label: str, T: float, T_ref: float, p_ref: Optional[float]) -> float:
    """
    Simple Clausius-Clapeyron fallback: p = p_ref * exp(-B(1/T - 1/T_ref)).

    P3: Fail-fast on invalid psat_ref (no silent return 0.0).
    """
    # P3: Fail-fast on invalid psat_ref
    if p_ref is None or not np.isfinite(p_ref) or p_ref <= 0.0:
        raise InterfaceEquilibriumError(
            f"Invalid psat_ref for {fluid_label}: {p_ref} (must be finite and positive)"
        )

    # Placeholder constant slope; can be replaced with real parameters if available
    B = 2000.0
    val = p_ref * np.exp(-B * (1.0 / max(T, EPS) - 1.0 / max(T_ref, EPS)))

    # P3: Fail-fast on non-finite result (no nan_to_num)
    if not np.isfinite(val):
        raise InterfaceEquilibriumError(
            f"Non-finite psat from Clausius for {fluid_label}: psat={val}, T={T:.2f}K, p_ref={p_ref}"
        )

    return float(val)


def _compute_psat_single(
    fluid_label: str, T: float, psat_model: str, T_ref: float, p_ref: Optional[float]
) -> Tuple[float, str, str]:
    """
    Compute psat with strict single-path routing (P2: no fallback).

    Returns:
        (value, source, reason)
        - value: psat in Pa
        - source: "coolprop" | "clausius"
        - reason: always "" (no fallback allowed)

    Raises:
        InterfaceEquilibriumError: If CoolProp fails when psat_model="coolprop" or "auto"
    """
    if psat_model in ("coolprop", "auto"):
        val = _psat_coolprop_single(fluid_label, T)
        if val is not None:
            return (val, "coolprop", "")
        # P2: CoolProp failed - raise instead of fallback
        raise InterfaceEquilibriumError(
            f"CoolProp psat failed for {fluid_label} at T={T:.2f}K (psat_model={psat_model})"
        )
    # Direct Clausius (psat_model == "clausius")
    val = _psat_clausius_single(fluid_label, T, T_ref, p_ref)
    return (val, "clausius", "")


def _psat_vec_all_with_meta(
    model: EquilibriumModel, T: float
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute psat vector for all liquid species with source tracking.

    P2: Strict single-path - no fallback allowed.
    P4: Routes to custom saturation model if sat_source="custom".

    Returns:
        (psat, sources, reasons)
        - psat: (Ns_l,) saturation pressures [Pa]
        - sources: list of "coolprop" or "clausius" or "custom" for each species
        - reasons: always "" (no fallback; failures raise InterfaceEquilibriumError)
    """
    Ns_l = len(model.liq_names)
    psat = np.zeros(Ns_l, dtype=np.float64)
    sources: List[str] = []
    reasons: List[str] = []

    # P4 DEBUG: Log sat_source to verify configuration
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        f"[P4 DEBUG] _psat_vec_all_with_meta: model.sat_source='{model.sat_source}', "
        f"sat_model={'present' if model.sat_model else 'None'}, "
        f"sat_db={'present' if model.sat_db else 'None'}"
    )

    # P4: Route to custom saturation model if configured
    if model.sat_source == "custom":
        if model.sat_model is None or model.sat_db is None:
            raise InterfaceEquilibriumError(
                "sat_source='custom' but sat_model or sat_db is None. "
                "Check build_equilibrium_model configuration."
            )

        for i, name in enumerate(model.liq_names):
            # Get species parameters from database
            sp_params = model.sat_db.get_params(name)

            # Compute psat using custom model
            val = model.sat_model.psat(sp_params, T)

            # P3: Fail-fast on invalid psat
            if not np.isfinite(val) or val < 0.0:
                raise InterfaceEquilibriumError(
                    f"Invalid psat for {name} at T={T:.2f}K: psat={val} (source=custom)"
                )

            psat[i] = val
            sources.append("custom")
            reasons.append("")

        return psat, sources, reasons

    # Original CoolProp/Clausius path (sat_source="coolprop")
    for i, name in enumerate(model.liq_names):
        p_ref = model.psat_ref.get(name)
        base_label = model.cp_fluids[i] if i < len(model.cp_fluids) else name
        # Respect backend if provided and not already namespaced
        if "::" in base_label:
            fluid_label = base_label
        else:
            fluid_label = f"{model.cp_backend}::{base_label}"
        val, src, reason = _compute_psat_single(fluid_label, T, model.psat_model, model.T_ref, p_ref)

        # P3: Fail-fast on invalid psat (no silent nan_to_num)
        if not np.isfinite(val) or val < 0.0:
            raise InterfaceEquilibriumError(
                f"Invalid psat for {name} at T={T:.2f}K: psat={val} (source={src})"
            )

        psat[i] = val
        sources.append(src)
        reasons.append(reason)

    return psat, sources, reasons


def _psat_vec_all(model: EquilibriumModel, T: float) -> np.ndarray:
    """Compute psat vector for all liquid species using model settings (legacy wrapper)."""
    psat, _, _ = _psat_vec_all_with_meta(model, T)
    return psat


def build_equilibrium_model(
    cfg: CaseConfig,
    Ns_g: int,
    Ns_l: int,
    M_g: np.ndarray,
    M_l: np.ndarray,
) -> EquilibriumModel:
    """Construct EquilibriumModel from config and provided molar masses."""
    eq_cfg = cfg.physics.interface.equilibrium
    sp_cfg = cfg.species
    liq2gas_map = dict(sp_cfg.liq2gas_map)

    gas_names = list(cfg.species.gas_species_full)
    liq_names = list(sp_cfg.liq_species)

    if len(gas_names) != Ns_g:
        raise ValueError(f"gas species length {len(gas_names)} != Ns_g {Ns_g}")
    if len(liq_names) != Ns_l:
        raise ValueError(f"liq species length {len(liq_names)} != Ns_l {Ns_l}")

    # Condensables list: allow empty -> infer from liq2gas_map values (order preserved)
    cond_gas = list(eq_cfg.condensables_gas)
    if len(cond_gas) == 0:
        cond_gas = list(liq2gas_map.values())

    # map condensables: provided as gas-phase names; map to liquid via liq2gas_map values
    idx_cond_l: List[int] = []
    idx_cond_g: List[int] = []
    gas_index = {name: i for i, name in enumerate(gas_names)}
    liq_index = {name: i for i, name in enumerate(liq_names)}
    gas2liq_map = {v: k for k, v in liq2gas_map.items()}
    for g_name in cond_gas:
        l_name = gas2liq_map.get(g_name)
        if l_name is None:
            raise ValueError(f"Condensable gas {g_name} not mapped in liq2gas_map.")
        if l_name not in liq_index or g_name not in gas_index:
            raise ValueError(f"Condensable mapping {l_name}->{g_name} not found in species lists.")
        idx_cond_l.append(liq_index[l_name])
        idx_cond_g.append(gas_index[g_name])

    idx_cond_l_arr = np.array(idx_cond_l, dtype=int)
    idx_cond_g_arr = np.array(idx_cond_g, dtype=int)

    # CoolProp fluids alignment checks
    cp_fluids = list(eq_cfg.coolprop.fluids)
    if len(cp_fluids) == 0:
        cp_fluids = list(liq_names)
    elif len(cp_fluids) == 1 and Ns_l == 1:
        cp_fluids = list(cp_fluids)
    elif len(cp_fluids) != Ns_l:
        raise ValueError(
            f"CoolProp fluids length {len(cp_fluids)} invalid for Ns_l={Ns_l}; "
            "must be 0 (auto), Ns_l, or 1 when Ns_l==1."
        )

    # farfield gas from initial.Yg ordered by gas species
    Yg_far = np.zeros(Ns_g, dtype=np.float64)
    init_Yg = cfg.initial.Yg
    for name, frac in init_Yg.items():
        if name in gas_index:
            Yg_far[gas_index[name]] = float(frac)
    Yg_far = np.nan_to_num(Yg_far, nan=0.0, posinf=0.0, neginf=0.0)
    # renormalize if sum !=1
    s_far = np.sum(Yg_far)
    if s_far > EPS:
        Yg_far /= s_far
    Xg_far = mass_to_mole(Yg_far, M_g)

    # P4: Load custom saturation model if configured
    sat_source = getattr(eq_cfg, "sat_source", "coolprop")
    sat_model = None
    sat_db = None

    # P4 DEBUG: Log configuration loading
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        f"[P4 DEBUG] build_equilibrium_model: sat_source from cfg = '{sat_source}'"
    )

    if sat_source == "custom":
        # Import here to avoid circular dependency
        from properties.liquid_sat_db import load_liquid_sat_db
        from properties.saturation_models import create_saturation_model

        # Get custom_sat config
        custom_sat_cfg = getattr(eq_cfg, "custom_sat", None)
        if custom_sat_cfg is None:
            raise ValueError(
                "sat_source='custom' requires 'custom_sat' configuration "
                "(model and params_file)"
            )

        # Extract model name and params file
        model_name = getattr(custom_sat_cfg, "model", "mm_integral_watson")
        params_file = getattr(custom_sat_cfg, "params_file", None)
        if params_file is None:
            raise ValueError(
                "sat_source='custom' requires 'custom_sat.params_file' "
                "(path to liquid_sat_params.yaml)"
            )

        # Load database and create model
        sat_db = load_liquid_sat_db(params_file)
        sat_model = create_saturation_model(model_name)

        # Validate that all liquid species have parameters in DB
        for liq_name in liq_names:
            if not sat_db.has_species(liq_name):
                raise InterfaceEquilibriumError(
                    f"Liquid species '{liq_name}' not found in custom saturation database "
                    f"(loaded from {params_file}). Available: {sat_db.list_species()}"
                )

    return EquilibriumModel(
        method=eq_cfg.method,
        psat_model=eq_cfg.psat_model,
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=idx_cond_l_arr,
        idx_cond_g=idx_cond_g_arr,
        M_g=np.asarray(M_g, dtype=np.float64),
        M_l=np.asarray(M_l, dtype=np.float64),
        Yg_farfield=Yg_far,
        Xg_farfield=Xg_far,
        cp_backend=eq_cfg.coolprop.backend,
        cp_fluids=cp_fluids,
        T_ref=298.15,
        psat_ref={},
        sat_source=sat_source,
        sat_model=sat_model,
        sat_db=sat_db,
    )


def compute_interface_equilibrium_full(
    model: EquilibriumModel,
    Ts: float,
    Pg: float,
    Yl_face: np.ndarray,
    Yg_face: np.ndarray,
) -> EquilibriumResult:
    """
    Compute interface equilibrium (Raoult, ideal solution) with full diagnostics.

    P3: Unified constraint flow (no hidden clip/max/cap).

    Steps:
    1) Reshape/clean inputs; renormalize Yl_face defensively.
    2) Convert liquid mass -> mole fractions.
    3) Build psat vector (P3: fail-fast on invalid psat, no clip).
    4) Raoult partial pressures for condensables (P3: fail-fast on NaN).
    5) Condensable gas mole fractions.
    6) P3: Apply condensable guard (unified constraint).
    7) P3: Fill background using farfield composition (fail-fast if impossible).
    8) Assemble full gas mole fractions.
    9) P3: Finalize simplex (light correction only, fail-fast on large errors).
    10) Convert mole -> mass fractions for gas.
    """
    # P3: Input validation (fail-fast on non-finite or invalid inputs)
    if not np.isfinite(Ts) or Ts <= 0.0:
        raise InterfaceEquilibriumError(
            f"Invalid surface temperature Ts={Ts} (must be finite and positive)"
        )
    if not np.isfinite(Pg) or Pg <= 0.0:
        raise InterfaceEquilibriumError(
            f"Invalid gas pressure Pg={Pg} (must be finite and positive)"
        )

    Ns_g = len(model.M_g)
    Ns_l = len(model.M_l)

    Yl_face = np.asarray(Yl_face, dtype=np.float64).reshape(Ns_l)
    Yg_face = np.asarray(Yg_face, dtype=np.float64).reshape(Ns_g)

    # P3: Check array inputs for finite values
    if not np.all(np.isfinite(Yl_face)):
        raise InterfaceEquilibriumError(
            f"Non-finite values in Yl_face: {Yl_face}"
        )
    if not np.all(np.isfinite(Yg_face)):
        raise InterfaceEquilibriumError(
            f"Non-finite values in Yg_face: {Yg_face}"
        )

    # Clean liquid face (defensive, allow zero)
    yl_sum = float(np.sum(Yl_face))
    if yl_sum <= 0.0:
        Yl_face = np.zeros_like(Yl_face)
    else:
        Yl_face = Yl_face / yl_sum

    X_liq = mass_to_mole(Yl_face, model.M_l)

    # P3: Compute psat (fail-fast in _psat_vec_all_with_meta, no clip)
    psat, psat_sources, psat_reasons = _psat_vec_all_with_meta(model, Ts)

    idxL = model.idx_cond_l
    idxG = model.idx_cond_g
    x_cond = X_liq[idxL] if idxL.size else np.zeros(0, dtype=np.float64)
    p_partial = x_cond * psat[idxL] if idxL.size else np.zeros(0, dtype=np.float64)

    # P3: Fail-fast on NaN partial pressures (no nan_to_num)
    if not np.all(np.isfinite(p_partial)):
        raise InterfaceEquilibriumError(
            f"Non-finite partial pressure at Ts={Ts:.6g}, Pg={Pg:.6g}"
        )

    Pg_safe = max(float(Pg), 1.0)
    y_cond = p_partial / Pg_safe if idxG.size else np.zeros(0, dtype=np.float64)

    # P4: Calculate psat_over_P for early guard triggering near boiling point
    psat_over_P = 0.0
    if idxL.size > 0:
        psat_over_P = float(np.max(psat[idxL]) / max(Pg, 1.0))

    # P4: Early guard trigger if psat is close to Pg (near boiling point)
    # This prevents numerical issues when approaching phase change
    GUARD_PSAT_RATIO = 0.8  # Trigger guard when psat/Pg >= 0.8
    early_trigger = psat_over_P >= GUARD_PSAT_RATIO

    # P3: Apply condensable guard (unified constraint, sole authority over condensables)
    # Check environment variable for smooth guard (default: hard guard)
    use_smooth_guard = os.environ.get("DROPLET_SMOOTH_GUARD", "0") in ("1", "true", "True", "TRUE")

    if early_trigger and not use_smooth_guard:
        # P4: Force guard trigger near boiling point
        boundary = 1.0 - EPS_BG
        y_cond_sum_pre = float(np.sum(y_cond))
        if y_cond_sum_pre > boundary:
            # Already exceeds boundary, normal guard applies
            y_cond, guard_meta = _apply_condensable_guard(y_cond, boundary=boundary, smooth=False)
        else:
            # Force scaling to boundary to ensure background has space
            scale = boundary / max(y_cond_sum_pre, 1e-300)
            y_cond = y_cond * scale
            y_cond_sum = float(np.sum(y_cond))
            guard_meta = {
                "ys_cap_hit": True,  # Forced by psat_over_P criterion
                "y_cond_sum": y_cond_sum,
                "boundary": boundary,
                "guard_type": "hard_early",
            }
    else:
        # Normal guard (regular or smooth)
        y_cond, guard_meta = _apply_condensable_guard(y_cond, boundary=1.0 - EPS_BG, smooth=use_smooth_guard)

    ys_cap_hit = guard_meta["ys_cap_hit"]
    y_cond_sum = guard_meta["y_cond_sum"]

    # P3: Compute background total (no max, direct from guarded y_cond_sum)
    y_bg_total = 1.0 - y_cond_sum

    mask_bg = np.ones(Ns_g, dtype=bool)
    mask_bg[idxG] = False

    # P3: Fill background using unified function (fail-fast if farfield bg=0)
    y_bg = _fill_background(
        y_bg_total=y_bg_total,
        X_farfield_background=model.Xg_farfield,
        idx_bg=mask_bg,
        Ns_g=Ns_g,
    )

    # Calculate X_bg_norm for diagnostics (normalized farfield background)
    X_bg = np.where(mask_bg, model.Xg_farfield, 0.0)
    s_bg = float(np.sum(X_bg))
    X_bg_norm = X_bg / s_bg if s_bg > EPS else np.zeros(Ns_g, dtype=np.float64)

    # Calculate sum of partial pressures for diagnostics
    sum_partials = float(np.sum(p_partial))

    # Assemble full gas mole fractions
    y_all = np.zeros(Ns_g, dtype=np.float64)
    if idxG.size:
        y_all[idxG] = y_cond
    y_all[mask_bg] = y_bg[mask_bg]

    # P3: Finalize simplex (light correction only, no hard clip)
    y_all = _finalize_simplex(y_all, tol_neg=1e-14, tol_sum=1e-10)

    # Convert mole -> mass fractions
    Yg_eq = mole_to_mass(y_all, model.M_g)

    # P3: Defensive renormalize mass fractions (should be ~1 already)
    s_Yg = float(np.sum(Yg_eq))
    if abs(s_Yg - 1.0) > 1e-12:
        if s_Yg > EPS:
            Yg_eq /= s_Yg
        else:
            raise InterfaceEquilibriumError(
                f"Yg_eq sum near zero: {s_Yg:.6e} at Ts={Ts:.6g}, Pg={Pg:.6g}"
            )

    # P2: Build meta dict with source tracking (P2: no fallback, only coolprop or clausius)
    # P4: Extended to support "custom" source
    if all(src == "coolprop" for src in psat_sources):
        psat_source_summary = "coolprop"
    elif all(src == "clausius" for src in psat_sources):
        psat_source_summary = "clausius"
    elif all(src == "custom" for src in psat_sources):
        psat_source_summary = "custom"
    else:
        psat_source_summary = "mixed"
    # P2: reasons should always be empty (no fallback)
    fallback_reason_summary = ""

    # P0.1: Check finite validity
    finite_ok = (
        np.all(np.isfinite(psat))
        and np.all(np.isfinite(y_cond))
        and np.all(np.isfinite(Yg_eq))
    )

    # P2: Consistency checks
    y_all_sum = float(np.sum(y_all))
    if not np.isfinite(y_all_sum) or abs(y_all_sum - 1.0) > 1e-10:
        raise InterfaceEquilibriumError(
            f"interface y_sum != 1: {y_all_sum} at Ts={Ts:.6g}, Pg={Pg:.6g}"
        )
    y_all_min = float(np.min(y_all))
    if y_all_min < -1e-14:
        raise InterfaceEquilibriumError(
            f"interface y has negative component: min={y_all_min} at Ts={Ts:.6g}, Pg={Pg:.6g}"
        )

    # P2/P4: psat_over_P already calculated earlier for early guard triggering

    meta = {
        "psat_sources": psat_sources,  # List per species
        "psat_reasons": psat_reasons,  # List per species (P2: always empty)
        "psat_source": psat_source_summary,  # P2: "coolprop" | "clausius" | "mixed"
        "hvap_source": "not_computed",  # Placeholder (hvap not tracked in equilibrium.py)
        "fallback_reason": fallback_reason_summary,  # P2: always empty (no fallback)
        "finite_ok": finite_ok,
        "y_cond_sum": y_cond_sum,  # P2: Sum of condensable mole fractions
        "y_bg_total": y_bg_total,  # P2: Total background mole fraction
        "ys_cap_hit": ys_cap_hit,  # P2: Whether boiling guard was triggered
        "psat_over_P": psat_over_P,  # P2: Max psat/Pg ratio
    }

    return EquilibriumResult(
        Yg_eq=Yg_eq,
        y_all=y_all,
        y_cond=y_cond,
        psat=psat,
        X_liq=X_liq,
        x_cond=x_cond,
        p_partial=p_partial,
        sum_partials=sum_partials,
        y_bg_total=y_bg_total,
        X_bg_norm=X_bg_norm,
        mask_bg=mask_bg,
        meta=meta,
    )


def compute_interface_equilibrium(
    model: EquilibriumModel,
    Ts: float,
    Pg: float,
    Yl_face: np.ndarray,
    Yg_face: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute interface-equilibrium gas composition (legacy wrapper).

    Returns:
        Yg_eq : (Ns_g,) equilibrium gas mass fractions
        y_cond : (Nv,) condensable gas mole fractions
        psat : (Ns_l,) saturation pressures for all liquid species
    """
    res = compute_interface_equilibrium_full(model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face)
    return res.Yg_eq, res.y_cond, res.psat
