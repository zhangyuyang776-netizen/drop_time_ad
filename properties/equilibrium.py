"""
Interface phase-equilibrium utilities (Raoult + psat) using the P2 liquid DB.

Runtime path is single-source: p2db only. No legacy branches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from core.logging_utils import is_root_rank
from core.types import CaseConfig
from properties.p2_liquid_db import interface_equilibrium as p2_interface_equilibrium
from properties.p2_liquid_db import load_liquid_db

FloatArray = np.ndarray
EPS = 1e-30
EPS_BG = 1e-5  # Minimum background gas mole fraction (boiling guard)

logger = logging.getLogger(__name__)


def decide_regime(
    sum_x_psat_over_P: float,
    prev: str | None = None,
    *,
    tol_enter: float = 1.0e-3,
    tol_exit: float = 5.0e-3,
) -> tuple[str, bool]:
    """
    Decide interface regime based on sum(x_i * psat_i / P) with hysteresis.

    Returns (regime, switched) where regime in {"evap", "sat"}.
    """
    s = float(sum_x_psat_over_P)
    if not np.isfinite(s):
        return (prev or "evap"), False
    prev_norm = str(prev or "evap").strip().lower()
    if prev_norm not in ("evap", "sat", "boil"):
        prev_norm = "evap"

    if prev_norm in ("sat", "boil"):
        if s <= 1.0 - float(tol_exit):
            return "evap", True
        return "sat", False

    if s >= 1.0 - float(tol_enter):
        return "sat", True
    return "evap", False


def _apply_boiling_guard(
    y_cond: np.ndarray, eps_bg: float = EPS_BG
) -> Tuple[np.ndarray, bool, float]:
    """
    Apply boiling guard to condensable mole fractions (pure function for testing).
    """
    y_cond = np.asarray(y_cond, dtype=np.float64).copy()
    y_cond_sum = float(np.sum(y_cond))
    ys_cap_hit = False

    if y_cond_sum > 1.0 - eps_bg:
        scale = (1.0 - eps_bg) / max(y_cond_sum, 1e-300)
        y_cond *= scale
        y_cond_sum = float(np.sum(y_cond))
        ys_cap_hit = True

    y_bg_total = max(1.0 - y_cond_sum, 0.0)
    return y_cond, ys_cap_hit, y_bg_total


def _apply_condensable_guard(
    y_cond: np.ndarray, boundary: float = 1.0 - EPS_BG, smooth: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y_cond = np.asarray(y_cond, dtype=np.float64).copy()
    y_cond_sum = float(np.sum(y_cond))
    ys_cap_hit = False
    guard_type = "smooth" if smooth else "hard"

    if smooth:
        smooth_start = 0.9 * boundary
        if y_cond_sum > smooth_start:
            if y_cond_sum > boundary:
                scale = boundary / max(y_cond_sum, 1e-300)
                ys_cap_hit = True
            else:
                transition = (y_cond_sum - smooth_start) / max(boundary - smooth_start, 1e-300)
                k = 0.5
                scale_reduction = k * transition
                target_sum = y_cond_sum * (1.0 - scale_reduction)
                scale = target_sum / max(y_cond_sum, 1e-300)
                ys_cap_hit = True if y_cond_sum > 0.95 * boundary else False
            y_cond *= scale
            y_cond_sum = float(np.sum(y_cond))
    else:
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
    y_all = np.asarray(y_all, dtype=np.float64).copy()
    y_min = float(np.min(y_all))
    y_sum = float(np.sum(y_all))

    if y_min < -tol_neg:
        raise InterfaceEquilibriumError(
            f"Large negative mole fraction detected: min={y_min:.6e} (tol={tol_neg:.6e})."
        )

    if y_min < 0:
        y_all = np.maximum(y_all, 0.0)
        y_sum = float(np.sum(y_all))

    if not np.isfinite(y_sum) or abs(y_sum - 1.0) > tol_sum:
        raise InterfaceEquilibriumError(
            f"Mole fraction sum far from 1.0: sum={y_sum:.15f} (tol={tol_sum:.6e})."
        )

    if abs(y_sum - 1.0) > 1e-15 and y_sum > EPS:
        y_all /= y_sum

    return y_all


class InterfaceEquilibriumError(RuntimeError):
    """Fail-fast error for interface equilibrium computations."""


def mass_to_mole(Y: FloatArray, M: FloatArray) -> FloatArray:
    Y = np.asarray(Y, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    denom = np.sum(Y / np.maximum(M, EPS))
    if denom <= 0.0:
        return np.zeros_like(Y)
    X = (Y / np.maximum(M, EPS)) / denom
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def mole_to_mass(X: FloatArray, M: FloatArray) -> FloatArray:
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
    gas_names: List[str]
    liq_names: List[str]
    idx_cond_l: np.ndarray
    idx_cond_g: np.ndarray
    M_g: np.ndarray
    M_l: np.ndarray
    Yg_farfield: np.ndarray
    Xg_farfield: np.ndarray
    p2db: Any
    Ts_guard_dT: float = 3.0
    Ts_guard_width_K: float = 0.5
    Ts_sat_eps_K: float = 0.01
    eps_bg: float = EPS_BG
    case_id: str = ""


@dataclass(slots=True)
class EquilibriumResult:
    Yg_eq: np.ndarray
    y_all: np.ndarray
    y_cond: np.ndarray
    psat: np.ndarray
    X_liq: np.ndarray
    x_cond: np.ndarray
    p_partial: np.ndarray
    sum_partials: float
    y_bg_total: float
    X_bg_norm: np.ndarray
    mask_bg: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


def _resolve_db_path(cfg: CaseConfig, db_file: str | Path) -> Path:
    path = Path(db_file)
    if path.is_absolute():
        return path
    base = getattr(getattr(cfg, "paths", None), "mechanism_dir", None)
    if isinstance(base, Path):
        return (base / path).resolve()
    return path.resolve()


def _reject_legacy_equilibrium(eq_cfg: Any) -> None:
    for key in ("sat_source", "psat_model"):
        if not hasattr(eq_cfg, key):
            continue
        val = str(getattr(eq_cfg, key) or "").strip().lower()
        if val and val not in ("p2db", "p2"):
            raise ValueError(
                f"{key}='{val}' is not supported; runtime only supports 'p2db'."
            )


def build_equilibrium_model(
    cfg: CaseConfig,
    Ns_g: int,
    Ns_l: int,
    M_g: np.ndarray,
    M_l: np.ndarray,
) -> EquilibriumModel:
    eq_cfg = cfg.physics.interface.equilibrium
    _reject_legacy_equilibrium(eq_cfg)

    sp_cfg = cfg.species
    liq2gas_map = dict(sp_cfg.liq2gas_map)

    gas_names = list(cfg.species.gas_species_full)
    liq_names = list(sp_cfg.liq_species)

    if len(gas_names) != Ns_g:
        raise ValueError(f"gas species length {len(gas_names)} != Ns_g {Ns_g}")
    if len(liq_names) != Ns_l:
        raise ValueError(f"liq species length {len(liq_names)} != Ns_l {Ns_l}")

    cond_gas = list(getattr(eq_cfg, "condensables_gas", []) or [])
    if len(cond_gas) == 0:
        cond_gas = list(liq2gas_map.values())

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

    Yg_far = np.zeros(Ns_g, dtype=np.float64)
    init_Yg = cfg.initial.Yg
    for name, frac in init_Yg.items():
        if name in gas_index:
            Yg_far[gas_index[name]] = float(frac)
    Yg_far = np.nan_to_num(Yg_far, nan=0.0, posinf=0.0, neginf=0.0)
    s_far = np.sum(Yg_far)
    if s_far > EPS:
        Yg_far /= s_far
    Xg_far = mass_to_mole(Yg_far, M_g)

    liq_cfg = getattr(getattr(cfg, "physics", None), "liquid", None)
    db_file = str(getattr(liq_cfg, "db_file", "") or "") if liq_cfg is not None else ""
    if not db_file:
        raise ValueError("p2db requires physics.liquid.db_file.")
    db_path = _resolve_db_path(cfg, db_file)
    p2db = load_liquid_db(db_path)
    for liq_name in liq_names:
        if not p2db.has_species(liq_name):
            raise InterfaceEquilibriumError(
                f"Liquid species '{liq_name}' not found in p2 liquid db "
                f"(loaded from {db_path}). Available: {p2db.list_species()}"
            )

    return EquilibriumModel(
        method=str(getattr(eq_cfg, "method", "raoult_psat")),
        gas_names=gas_names,
        liq_names=liq_names,
        idx_cond_l=idx_cond_l_arr,
        idx_cond_g=idx_cond_g_arr,
        M_g=np.asarray(M_g, dtype=np.float64),
        M_l=np.asarray(M_l, dtype=np.float64),
        Yg_farfield=Yg_far,
        Xg_farfield=Xg_far,
        p2db=p2db,
        Ts_guard_dT=float(getattr(eq_cfg, "Ts_guard_dT", 3.0)),
        Ts_guard_width_K=float(getattr(eq_cfg, "Ts_guard_width_K", 0.5)),
        # Hard cap very close to Tbub to keep psat(T)/P <= 1 in all trial states while still
        # allowing the "sat" regime to be reachable.
        Ts_sat_eps_K=float(getattr(eq_cfg, "Ts_sat_eps_K", 0.01)),
        eps_bg=float(getattr(eq_cfg, "eps_bg", EPS_BG)),
        case_id=str(getattr(getattr(cfg, "case", None), "id", "") or ""),
    )


def compute_interface_equilibrium_full(
    model: EquilibriumModel,
    Ts: float,
    Pg: float,
    Yl_face: np.ndarray,
    Yg_face: np.ndarray,
) -> EquilibriumResult:
    if model.p2db is None:
        raise InterfaceEquilibriumError("p2db is not loaded in equilibrium model.")

    p2 = p2_interface_equilibrium(
        P=float(Pg),
        Ts=float(Ts),
        Yl_face=Yl_face,
        Yg_far=model.Yg_farfield,
        liq_names=model.liq_names,
        gas_names=model.gas_names,
        idx_cond_l=model.idx_cond_l,
        idx_cond_g=model.idx_cond_g,
        M_l=model.M_l,
        M_g=model.M_g,
        db=model.p2db,
        Ts_guard_dT=float(model.Ts_guard_dT),
        Ts_guard_width_K=float(model.Ts_guard_width_K),
        Ts_sat_eps_K=float(model.Ts_sat_eps_K),
        eps_bg=float(model.eps_bg),
    )

    psat = np.asarray(p2["psat"], dtype=np.float64)
    y_cond = np.asarray(p2["y_cond"], dtype=np.float64)
    Yg_eq = np.asarray(p2["Yg_eq"], dtype=np.float64)
    X_liq = np.asarray(p2["x_l"], dtype=np.float64)
    x_cond = np.asarray(p2["x_cond"], dtype=np.float64)

    idxL = model.idx_cond_l
    p_partial = x_cond * psat[idxL] if idxL.size else np.zeros(0, dtype=np.float64)
    sum_partials = float(np.sum(p_partial)) if p_partial.size else 0.0

    mask_bg = np.ones(len(model.gas_names), dtype=bool)
    mask_bg[model.idx_cond_g] = False
    X_bg = np.where(mask_bg, model.Xg_farfield, 0.0)
    s_bg = float(np.sum(X_bg))
    X_bg_norm = X_bg / s_bg if s_bg > EPS else np.zeros_like(X_bg)

    y_all = mass_to_mole(Yg_eq, model.M_g)
    y_bg_total = float(p2.get("meta", {}).get("y_bg_total", 0.0))

    meta = dict(p2.get("meta", {}))
    meta.update(
        {
            "psat_sources": ["p2db"] * len(model.liq_names),
            "psat_source": "p2db",
            "hvap_source": "p2db",
            "latent_source": "p2db",
            "Ts_guard_dT": float(model.Ts_guard_dT),
            "Ts_guard_width_K": float(model.Ts_guard_width_K),
            "eps_bg": float(model.eps_bg),
            "case_id": model.case_id,
            "path": "full",
        }
    )

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


def interface_equilibrium(
    model: EquilibriumModel,
    Ts: float,
    Pg: float,
    Yl_face: np.ndarray,
    Yg_face: np.ndarray,
) -> EquilibriumResult:
    return compute_interface_equilibrium_full(model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face)


def compute_interface_equilibrium(
    model: EquilibriumModel,
    Ts: float,
    Pg: float,
    Yl_face: np.ndarray,
    Yg_face: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if is_root_rank():
        logger.warning("[EQPATH] legacy compute_interface_equilibrium called")
    res = interface_equilibrium(model, Ts=Ts, Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face)
    if isinstance(res.meta, dict):
        res.meta["path"] = "legacy"
        res.meta.setdefault("psat_source", "p2db")
        res.meta.setdefault("hvap_source", "p2db")
    return res.Yg_eq, res.y_cond, res.psat
