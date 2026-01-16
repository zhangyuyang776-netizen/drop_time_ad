from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from core.types import CaseConfig, Grid1D, Props, State
from properties.aggregator import build_props_from_state
from properties.gas import GasPropertiesModel, build_gas_model
from properties.liquid import LiquidPropertiesModel, build_liquid_model


@dataclass(frozen=True, slots=True)
class _ModelCacheKey:
    """Cache key for property models to avoid rebuilding every timestep."""

    gas_mech_path: str
    gas_phase: str
    P_inf: float

    coolprop_backend: str
    coolprop_fluids: Tuple[str, ...]
    liq_species: Tuple[str, ...]


_MODEL_CACHE: Dict[_ModelCacheKey, Tuple[GasPropertiesModel, LiquidPropertiesModel]] = {}


def _resolve_gas_mech_path(cfg: CaseConfig) -> str:
    """Return absolute path string used for model cache key."""
    mech_path = cfg.paths.gas_mech
    try:
        mech_path = cfg.paths.mechanism_dir / cfg.paths.gas_mech
    except Exception:
        pass
    return str(Path(mech_path).resolve())


def _get_gas_phase(cfg: CaseConfig) -> str:
    """Optional phase name if present in cfg; else empty string."""
    try:
        return str(getattr(cfg.species, "gas_mechanism_phase", "") or "")
    except Exception:
        return ""


def _make_model_cache_key(cfg: CaseConfig) -> _ModelCacheKey:
    gas_mech_path = _resolve_gas_mech_path(cfg)
    gas_phase = _get_gas_phase(cfg)
    P_inf = float(cfg.initial.P_inf)

    backend = ""
    fluids: Tuple[str, ...] = tuple()
    try:
        eq_cfg = cfg.physics.interface.equilibrium
        backend = str(eq_cfg.coolprop.backend or "")
        fluids = tuple(eq_cfg.coolprop.fluids)
    except Exception:
        backend = ""
        fluids = tuple()

    liq_species = tuple(getattr(cfg.species, "liq_species", []) or [])

    return _ModelCacheKey(
        gas_mech_path=gas_mech_path,
        gas_phase=gas_phase,
        P_inf=P_inf,
        coolprop_backend=backend,
        coolprop_fluids=fluids,
        liq_species=liq_species,
    )


def clear_models_cache() -> None:
    """Clear cached property models (useful in tests)."""
    _MODEL_CACHE.clear()


def get_or_build_models(cfg: CaseConfig) -> Tuple[GasPropertiesModel, LiquidPropertiesModel]:
    """Build (gas_model, liquid_model) once per cfg signature and reuse them across timesteps."""
    key = _make_model_cache_key(cfg)
    hit = _MODEL_CACHE.get(key)
    if hit is not None:
        return hit

    gas_model = build_gas_model(cfg)
    liq_model = build_liquid_model(cfg)
    _MODEL_CACHE[key] = (gas_model, liq_model)
    return gas_model, liq_model


def compute_props(cfg: CaseConfig, grid: Grid1D, state: State) -> Tuple[Props, Dict[str, Any]]:
    """
    Unified properties recomputation entry point with model caching.

    Returns:
        props: aggregated Props (including enthalpy/psat/hvap fields)
        extras: dict with raw extras from gas/liquid calculators
    """
    gas_model, liq_model = get_or_build_models(cfg)

    props, extras = build_props_from_state(
        cfg=cfg,
        grid=grid,
        state=state,
        gas_model=gas_model,
        liq_model=liq_model,
    )

    Ns_g = int(state.Yg.shape[0])
    Ns_l = int(state.Yl.shape[0]) if state.Yl is not None else 0
    props.validate_shapes(grid, Ns_g=Ns_g, Ns_l=Ns_l)

    return props, extras
