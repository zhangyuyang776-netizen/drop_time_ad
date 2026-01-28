"""
Liquid-phase property evaluation (MVP):
- Uses p2db correlations for pure components.
- Mix rules: simple mass-fraction weighting for cp/k, harmonic for density.
- Extras returned: psat(Ts) and hvap(Ts) per liquid species for interface use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from core.types import CaseConfig, Grid1D, State
from properties.equilibrium import InterfaceEquilibriumError
from properties.p2_liquid_db import get_mixture_props, get_psat_hvap, load_liquid_db

FloatArray = np.ndarray


@dataclass(slots=True)
class LiquidPropertiesModel:
    backend: str
    fluids: Tuple[str, ...]
    liq_names: Tuple[str, ...]
    P_inf: float
    p2db: Any | None = None
    T_ref: float = 298.15


def build_liquid_model(cfg: CaseConfig) -> LiquidPropertiesModel:
    """Construct liquid property model from config."""
    liq_cfg = getattr(cfg.physics, "liquid", None)
    backend = str(getattr(liq_cfg, "backend", "p2db") or "p2db").lower()

    liq_names = tuple(cfg.species.liq_species)
    if backend == "p2db":
        db_file = str(getattr(liq_cfg, "db_file", "") or "")
        if not db_file:
            raise ValueError("physics.liquid.db_file must be set for backend='p2db'.")
        db_path = _resolve_db_path(cfg, db_file)
        p2db = load_liquid_db(db_path)
        t_ref = float(getattr(p2db, "meta", {}).get("T_ref", 298.15))
        return LiquidPropertiesModel(
            backend=backend,
            fluids=liq_names,
            liq_names=liq_names,
            P_inf=float(cfg.initial.P_inf),
            p2db=p2db,
            T_ref=t_ref,
        )

    raise ValueError(f"Unsupported liquid backend '{backend}'. Use backend='p2db'.")


def _resolve_db_path(cfg: CaseConfig, db_file: str | Path) -> Path:
    path = Path(db_file)
    if path.is_absolute():
        return path
    base = getattr(getattr(cfg, "paths", None), "mechanism_dir", None)
    if isinstance(base, Path):
        return (base / path).resolve()
    return path.resolve()


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
            f"Liquid species count {Ns_l} does not match configured liquid names length {len(model.fluids)}."
        )

    rho_l = np.zeros(Nl, dtype=np.float64)
    cp_l = np.zeros(Nl, dtype=np.float64)
    k_l = np.zeros(Nl, dtype=np.float64)

    if model.backend != "p2db":
        raise ValueError(f"Unsupported liquid backend '{model.backend}'. Use backend='p2db'.")
    if model.p2db is None:
        raise ValueError("p2db backend selected but database is not loaded.")
    # mixture properties at each cell temperature
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

        props, _ = get_mixture_props(
            model.p2db,
            model.liq_names,
            Y,
            T,
            P,
            T_ref=model.T_ref,
        )
        rho_l[il] = float(props["rho_l"])
        cp_l[il] = float(props["cp_l"])
        k_l[il] = float(props["k_l"])

    # interface psat/hvap at Ts
    Ts = float(state.Ts)
    if model.p2db is None:
        raise ValueError("p2db backend selected but database is not loaded.")
    psat_l, hvap_l = get_psat_hvap(model.p2db, model.liq_names, Ts)
    if not np.all(np.isfinite(psat_l)) or np.any(psat_l < 0.0):
        raise InterfaceEquilibriumError(f"Invalid psat values at Ts={Ts:.6g}.")
    if not np.all(np.isfinite(hvap_l)) or np.any(hvap_l <= 0.0):
        raise InterfaceEquilibriumError(f"Invalid hvap values at Ts={Ts:.6g}.")

    core = {"rho_l": rho_l, "cp_l": cp_l, "k_l": k_l}
    extra = {"psat_l": psat_l, "hvap_l": hvap_l}
    return core, extra
