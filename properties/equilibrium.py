"""
Interface phase-equilibrium utilities (Raoult + psat) with interface-noncondensables background fill.

Responsibilities:
- Build an EquilibriumModel from CaseConfig and species data (indices, molar masses, farfield Y).
- Compute interface-equilibrium gas composition Yg_eq given Ts, Pg, Yl_face, Yg_face.
- Provide psat helpers (CoolProp when available, Clausius fallback).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import CoolProp.CoolProp as CP
except Exception:  # pragma: no cover - environment dependent
    CP = None

from core.types import CaseConfig

FloatArray = np.ndarray
EPS = 1e-30


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
    meta: Dict[str, any] = field(default_factory=dict)  # P0.1: source tracking metadata


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
    """Simple Clausius-Clapeyron fallback: p = p_ref * exp(-B(1/T - 1/T_ref)); crude placeholder."""
    if p_ref is None or p_ref <= 0.0:
        return 0.0
    # Placeholder constant slope; can be replaced with real parameters if available
    B = 2000.0
    val = p_ref * np.exp(-B * (1.0 / max(T, EPS) - 1.0 / max(T_ref, EPS)))
    return float(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))


def _compute_psat_single(
    fluid_label: str, T: float, psat_model: str, T_ref: float, p_ref: Optional[float]
) -> Tuple[float, str, str]:
    """
    Compute psat with CoolProp first (if requested/available), Clausius fallback.

    Returns:
        (value, source, reason)
        - value: psat in Pa
        - source: "coolprop" | "clausius" | "fallback"
        - reason: failure reason if fallback was used, else ""
    """
    if psat_model in ("coolprop", "auto"):
        val = _psat_coolprop_single(fluid_label, T)
        if val is not None:
            return (val, "coolprop", "")
        # CoolProp failed
        reason = f"CoolProp failed for {fluid_label} at T={T:.2f}K"
        if psat_model == "coolprop":
            warnings.warn(f"{reason}; falling back to Clausius.")
        # Fallback to Clausius
        val_clausius = _psat_clausius_single(fluid_label, T, T_ref, p_ref)
        return (val_clausius, "fallback", reason)
    # Direct Clausius (psat_model == "clausius")
    val = _psat_clausius_single(fluid_label, T, T_ref, p_ref)
    return (val, "clausius", "")


def _psat_vec_all_with_meta(
    model: EquilibriumModel, T: float
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute psat vector for all liquid species with source tracking.

    Returns:
        (psat, sources, reasons)
        - psat: (Ns_l,) saturation pressures [Pa]
        - sources: list of "coolprop"/"clausius"/"fallback" for each species
        - reasons: list of failure reasons (empty string if no fallback)
    """
    Ns_l = len(model.liq_names)
    psat = np.zeros(Ns_l, dtype=np.float64)
    sources: List[str] = []
    reasons: List[str] = []
    for i, name in enumerate(model.liq_names):
        p_ref = model.psat_ref.get(name)
        base_label = model.cp_fluids[i] if i < len(model.cp_fluids) else name
        # Respect backend if provided and not already namespaced
        if "::" in base_label:
            fluid_label = base_label
        else:
            fluid_label = f"{model.cp_backend}::{base_label}"
        val, src, reason = _compute_psat_single(fluid_label, T, model.psat_model, model.T_ref, p_ref)
        psat[i] = val
        sources.append(src)
        reasons.append(reason)
    psat = np.nan_to_num(psat, nan=0.0, posinf=0.0, neginf=0.0)
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

    Steps:
    1) reshape/clean inputs; renormalize Yl_face defensively.
    2) convert liquid mass -> mole fractions.
    3) build psat vector (clipped non-negative).
    4) Raoult partial pressures for condensables (with NaN/neg guards).
     5) cap total condensable partial pressure to 0.995*Pg.
     6) condensable gas mole fractions.
     7) choose background source mole fractions from interface gas; normalize on background set.
     8) allocate background total fraction = 1 - sum(y_cond).
     9) assemble full gas mole fractions; clip/optional renorm for numerical noise.
     10) convert mole -> mass fractions for gas.
    """
    Ns_g = len(model.M_g)
    Ns_l = len(model.M_l)

    Yl_face = np.asarray(Yl_face, dtype=np.float64).reshape(Ns_l)
    Yg_face = np.asarray(Yg_face, dtype=np.float64).reshape(Ns_g)

    yl_sum = float(np.sum(Yl_face))
    if yl_sum <= 0.0:
        Yl_face = np.zeros_like(Yl_face)
    else:
        Yl_face = Yl_face / yl_sum
    Yg_face = np.nan_to_num(Yg_face, nan=0.0, posinf=0.0, neginf=0.0)

    X_liq = mass_to_mole(Yl_face, model.M_l)

    # P0.1: Use meta-tracking version to capture psat source
    psat, psat_sources, psat_reasons = _psat_vec_all_with_meta(model, Ts)
    psat = np.clip(psat, 0.0, np.inf)

    idxL = model.idx_cond_l
    idxG = model.idx_cond_g
    x_cond = X_liq[idxL] if idxL.size else np.zeros(0, dtype=np.float64)
    p_partial = x_cond * psat[idxL] if idxL.size else np.zeros(0, dtype=np.float64)
    p_partial = np.nan_to_num(p_partial, nan=0.0, posinf=0.0, neginf=0.0)
    p_partial = np.clip(p_partial, 0.0, np.inf)
    sum_partials = float(np.sum(p_partial))

    Pg_safe = max(float(Pg), 1.0)
    cap = 0.995 * Pg_safe
    if sum_partials > cap and sum_partials > 0.0:
        p_partial *= cap / sum_partials
        sum_partials = cap

    y_cond = p_partial / Pg_safe if idxG.size else np.zeros(0, dtype=np.float64)

    mask_bg = np.ones(Ns_g, dtype=bool)
    mask_bg[idxG] = False

    X_g_face = mass_to_mole(Yg_face, model.M_g)
    X_source = X_g_face

    X_bg = np.where(mask_bg, X_source, 0.0)
    s_bg = float(np.sum(X_bg))
    if s_bg > EPS:
        X_bg_norm = X_bg / s_bg
    else:
        X_bg_norm = np.zeros_like(X_bg)
        if np.any(mask_bg):
            X_bg_norm[mask_bg] = 1.0 / np.sum(mask_bg)

    y_bg_total = max(1.0 - float(np.sum(y_cond)), 0.0)
    y_bg = y_bg_total * X_bg_norm

    y_all = np.zeros(Ns_g, dtype=np.float64)
    if idxG.size:
        y_all[idxG] = y_cond
    y_all[mask_bg] = y_bg[mask_bg]
    y_all = np.nan_to_num(y_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = np.clip(y_all, 0.0, 1.0)
    s_all = float(np.sum(y_all))
    if s_all > EPS and not np.isclose(s_all, 1.0, rtol=1e-12, atol=1e-12):
        y_all /= s_all

    Yg_eq = mole_to_mass(y_all, model.M_g)
    s_Yg = float(np.sum(Yg_eq))
    if s_Yg > EPS and not np.isclose(s_Yg, 1.0, rtol=1e-12, atol=1e-12):
        Yg_eq = Yg_eq / s_Yg

    # P0.1: Build meta dict with source tracking
    psat_source_summary = "coolprop"
    fallback_reason_summary = ""
    if any(src in ("fallback", "clausius") for src in psat_sources):
        if all(src == "clausius" for src in psat_sources):
            psat_source_summary = "clausius"
        else:
            psat_source_summary = "fallback"
        # Collect non-empty reasons
        reasons_list = [r for r in psat_reasons if r]
        if reasons_list:
            fallback_reason_summary = "; ".join(reasons_list)

    # P0.1: Check finite validity
    finite_ok = (
        np.all(np.isfinite(psat))
        and np.all(np.isfinite(y_cond))
        and np.all(np.isfinite(Yg_eq))
    )

    meta = {
        "psat_sources": psat_sources,  # List per species
        "psat_reasons": psat_reasons,  # List per species
        "psat_source": psat_source_summary,  # "coolprop" | "fallback" | "clausius" | "mixed"
        "hvap_source": "not_computed",  # Placeholder (hvap not tracked in equilibrium.py)
        "fallback_reason": fallback_reason_summary,
        "finite_ok": finite_ok,
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
