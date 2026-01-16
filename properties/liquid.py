"""
Liquid-phase property evaluation (MVP):
- Uses CoolProp pure-component properties at fixed P_inf.
- Mix rules: simple mass-fraction weighting for cp/k, harmonic for density.
- Extras returned: psat(Ts) and hvap(Ts) per liquid species for interface use.

Future extensions:
- Non-ideal mixture rules, temperature-dependent viscosity/thermal conductivity.
- Multicomponent diffusion (Maxwell–Stefan) if solve_Yl is enabled.
- More accurate latent heat/psat models (fits, EOS mixing).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

import numpy as np

try:
    import CoolProp.CoolProp as CP
except Exception as e:  # pragma: no cover - environment dependent
    CP = None
    _cp_import_error = e
else:
    _cp_import_error = None

from core.types import CaseConfig, Grid1D, State

FloatArray = np.ndarray
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LiquidPropertiesModel:
    backend: str
    fluids: Tuple[str, ...]
    liq_names: Tuple[str, ...]
    P_inf: float


def build_liquid_model(cfg: CaseConfig) -> LiquidPropertiesModel:
    """Construct liquid property model from config."""
    if CP is None:
        raise ImportError(f"CoolProp is required for liquid properties: {_cp_import_error}")

    eq_cfg = cfg.physics.interface.equilibrium
    backend = eq_cfg.coolprop.backend
    raw_fluids = tuple(eq_cfg.coolprop.fluids)
    liq_names = tuple(cfg.species.liq_species)
    if len(raw_fluids) != len(liq_names):
        raise ValueError("CoolProp fluids list must align with liquid species order.")
    fluids = tuple(
        f"{backend}::{name}" if backend and "::" not in name else name
        for name in raw_fluids
    )
    return LiquidPropertiesModel(
        backend=backend,
        fluids=fluids,
        liq_names=liq_names,
        P_inf=float(cfg.initial.P_inf),
    )


def _pure_liquid_props(fluid: str, T: float, P: float) -> Tuple[float, float, float, float]:
    """Return rho [kg/m3], cp [J/kg/K], k [W/m/K], h [J/kg] for pure fluid at T,P."""
    T_safe = _cp_safe_T(fluid, T)
    rho = float(CP.PropsSI("D", "T", T_safe, "P", P, fluid))
    cp = float(CP.PropsSI("Cpmass", "T", T_safe, "P", P, fluid))
    k = float(CP.PropsSI("L", "T", T_safe, "P", P, fluid))
    h = float(CP.PropsSI("H", "T", T_safe, "P", P, fluid))
    return rho, cp, k, h


def _psat_hvap(fluid: str, T: float) -> Tuple[float, float]:
    """Return psat [Pa] and hvap [J/kg] at temperature T."""
    T_safe = _cp_safe_T(fluid, T)
    try:
        psat = float(CP.PropsSI("P", "T", T_safe, "Q", 0, fluid))
        hV = float(CP.PropsSI("H", "T", T_safe, "Q", 1, fluid))
        hL = float(CP.PropsSI("H", "T", T_safe, "Q", 0, fluid))
    except Exception as exc:
        logger.warning(
            "CoolProp psat/hvap failed for %s at T=%.3f (T_safe=%.3f): %s",
            fluid,
            T,
            T_safe,
            exc,
        )
        return 0.0, 0.0
    hvap = max(hV - hL, 0.0)
    return max(psat, 0.0), hvap


def _cp_safe_T(fluid: str, T: float) -> float:
    """Clamp temperature to CoolProp Tmin to avoid domain errors."""
    try:
        tmin = float(CP.PropsSI("Tmin", fluid))
    except Exception:
        tmin = None
    if tmin is not None and np.isfinite(tmin) and T < tmin:
        return tmin + 1.0e-3
    return T


def compute_liquid_props(
    model: LiquidPropertiesModel, state: State, grid: Grid1D
) -> Tuple[Dict[str, FloatArray], Dict[str, FloatArray]]:
    """
    Compute liquid mixture properties (simple mixing rules) and interface psat/hvap.

    Contract:
    - state.Tl.shape == (grid.Nl,)
    - state.Yl.shape == (Nspec_l, grid.Nl) where Nspec_l == len(model.fluids)
    - For each cell, state.Yl[:, il] is already full and normalized; no renormalization is done here.

    Returns:
        core: {"rho_l", "cp_l", "k_l"}
        extra: {"psat_l", "hvap_l"}
    """
    Nl = grid.Nl
    Ns_l = state.Yl.shape[0]
    P = model.P_inf

    if state.Tl.shape[0] != Nl:
        raise ValueError(f"Tl length {state.Tl.shape[0]} does not match grid.Nl={Nl}.")
    if state.Yl.shape[1] != Nl:
        raise ValueError(f"Yl shape {state.Yl.shape} inconsistent with Nl={Nl} (expected (Nspec_l, Nl)).")
    if Ns_l != len(model.fluids):
        raise ValueError(
            f"Liquid species count {Ns_l} does not match CoolProp fluids list length {len(model.fluids)}."
        )

    rho_l = np.zeros(Nl, dtype=np.float64)
    cp_l = np.zeros(Nl, dtype=np.float64)
    k_l = np.zeros(Nl, dtype=np.float64)

    # pure component props at each cell temperature
    for il in range(Nl):
        T = float(state.Tl[il])
        Y = np.asarray(state.Yl[:, il], dtype=np.float64)

        sY = float(np.sum(Y))
        if not np.isfinite(sY) or sY <= 0.0:
            raise ValueError(f"Liquid mass fractions at cell {il} are invalid: sum(Y)={sY}")
        if abs(sY - 1.0) > 1e-6:
            raise ValueError(
                f"Liquid mass fractions at cell {il} are not normalized: sum(Y)={sY}. "
                "state.Yl must be full, normalized mass fractions."
            )

        rho_pure = np.zeros(Ns_l, dtype=np.float64)
        cp_pure = np.zeros(Ns_l, dtype=np.float64)
        k_pure = np.zeros(Ns_l, dtype=np.float64)
        for j, fluid in enumerate(model.fluids):
            r_j, cp_j, k_j, _ = _pure_liquid_props(fluid, T, P)
            rho_pure[j] = r_j
            cp_pure[j] = cp_j
            k_pure[j] = k_j

        # mixing rules
        inv_rho = np.sum(np.where(rho_pure > 0.0, Y / rho_pure, 0.0))
        rho_mix = 1.0 / max(inv_rho, 1e-30)
        cp_mix = float(np.sum(Y * cp_pure))
        k_mix = float(np.sum(Y * k_pure))

        if not np.isfinite(rho_mix) or rho_mix <= 0.0:
            raise ValueError(f"Non-physical liquid density at cell {il}: {rho_mix}")
        if not np.isfinite(cp_mix) or cp_mix <= 0.0:
            raise ValueError(f"Non-physical liquid cp at cell {il}: {cp_mix}")
        if not np.isfinite(k_mix) or k_mix <= 0.0:
            raise ValueError(f"Non-physical liquid k at cell {il}: {k_mix}")

        rho_l[il] = rho_mix
        cp_l[il] = cp_mix
        k_l[il] = k_mix

    # interface psat/hvap at Ts
    Ts = float(state.Ts)
    psat_l = np.zeros(Ns_l, dtype=np.float64)
    hvap_l = np.zeros(Ns_l, dtype=np.float64)
    for j, fluid in enumerate(model.fluids):
        p_j, hv_j = _psat_hvap(fluid, Ts)
        psat_l[j] = max(p_j, 0.0)
        hvap_l[j] = max(hv_j, 0.0)

    core = {"rho_l": rho_l, "cp_l": cp_l, "k_l": k_l}
    extra = {"psat_l": psat_l, "hvap_l": hvap_l}
    return core, extra
