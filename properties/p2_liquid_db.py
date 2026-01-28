from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import re

import numpy as np

from properties.p2_mix_rules import eval_pure_props, mass_to_mole_fractions, mix_props, sanitize_mass_fractions
from properties.p2_pure_models import hvap as p2_hvap
from properties.p2_pure_models import psat as p2_psat
from properties.p2_safeguards import clip_temperature
from properties.p2_equilibrium import bubble_point_T

REQUIRED_PROPS = ("psat", "hvap", "rho", "cp", "k", "mu")
CACHE_DIGITS = 6
REQUIRED_PROP_FIELDS = ("Tmin", "Tmax", "source", "ref")


def _normalize_name(name: str) -> str:
    normalized = name.lower()
    normalized = re.sub(r"[-_\\s]+", "", normalized)
    return normalized


def _coerce_float(value: Any, field: str, species: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Invalid {field} for '{species}': {value!r}") from exc


def _lift_legacy_prop(spec: Mapping[str, Any], prop: str) -> Optional[Dict[str, Any]]:
    model_key = f"{prop}_model"
    if model_key not in spec:
        return None
    entry: Dict[str, Any] = {"model": spec[model_key]}
    for key, value in spec.items():
        if key.startswith(f"{prop}_") and key != model_key:
            entry[key[len(prop) + 1 :]] = value
    return entry


def _get_prop_entry(spec: Mapping[str, Any], prop: str, species: str) -> Dict[str, Any]:
    if prop in spec:
        entry = spec[prop]
    else:
        entry = _lift_legacy_prop(spec, prop)
    if entry is None:
        raise ValueError(f"Missing '{prop}' model for species '{species}'.")
    if not isinstance(entry, dict):
        raise ValueError(
            f"Invalid '{prop}' entry for '{species}': expected dict, got {type(entry)}"
        )
    model = entry.get("model", "")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Missing '{prop}.model' for species '{species}'.")
    for field in REQUIRED_PROP_FIELDS:
        if field not in entry:
            raise ValueError(f"Missing '{prop}.{field}' for species '{species}'.")
    Tmin = _coerce_float(entry["Tmin"], f"{prop}.Tmin", species)
    Tmax = _coerce_float(entry["Tmax"], f"{prop}.Tmax", species)
    if Tmin >= Tmax:
        raise ValueError(f"Invalid '{prop}.Tmin/Tmax' for species '{species}'.")
    source = entry.get("source")
    ref = entry.get("ref")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"Missing '{prop}.source' for species '{species}'.")
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError(f"Missing '{prop}.ref' for species '{species}'.")
    return dict(entry)


def _infer_t_valid(prop_entries: Mapping[str, Dict[str, Any]]) -> Optional[List[float]]:
    mins: List[float] = []
    maxs: List[float] = []
    for entry in prop_entries.values():
        if "Tmin" in entry and "Tmax" in entry:
            mins.append(float(entry["Tmin"]))
            maxs.append(float(entry["Tmax"]))
    if not mins or not maxs:
        return None
    return [min(mins), max(maxs)]


class LiquidPropsDB:
    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta
        self._species: Dict[str, Dict[str, Any]] = {}
        self._alias_map: Dict[str, str] = {}
        self._pure_cache: Dict[Tuple[str, float], Tuple[Dict[str, float], Dict[str, Any]]] = {}
        self._psat_cache: Dict[Tuple[str, float], float] = {}
        self._hvap_cache: Dict[Tuple[str, float], float] = {}
        self._tbub_cache: Dict[
            Tuple[float, Tuple[float, ...], Tuple[str, ...]], Tuple[float, bool]
        ] = {}

    def add_species(self, name: str, spec: Dict[str, Any], aliases: List[str]) -> None:
        if name in self._species:
            raise ValueError(f"Duplicate species entry: '{name}'.")
        self._species[name] = spec
        all_names = [name] + aliases
        for alias in all_names:
            normalized = _normalize_name(alias)
            existing = self._alias_map.get(normalized)
            if existing and existing != name:
                raise ValueError(
                    f"Alias collision: '{alias}' maps to '{existing}', cannot add to '{name}'."
                )
            self._alias_map[normalized] = name

    def get_params(self, name: str) -> Dict[str, Any]:
        normalized = _normalize_name(name)
        if normalized not in self._alias_map:
            raise KeyError(
                f"Species '{name}' not found. Available: {list(self._species.keys())}"
            )
        return self._species[self._alias_map[normalized]]

    def has_species(self, name: str) -> bool:
        return _normalize_name(name) in self._alias_map

    def list_species(self) -> List[str]:
        return list(self._species.keys())

    def canonical_name(self, name: str) -> str:
        normalized = _normalize_name(name)
        if normalized not in self._alias_map:
            raise KeyError(
                f"Species '{name}' not found. Available: {list(self._species.keys())}"
            )
        return self._alias_map[normalized]


def _pick_species_block(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if "species" in data and isinstance(data["species"], dict):
        return data["species"]
    if "liquids" in data and isinstance(data["liquids"], dict):
        return data["liquids"]
    raise ValueError("Invalid YAML: expected top-level 'species' or 'liquids' mapping.")


def load_liquid_db(yaml_path: str | Path) -> LiquidPropsDB:
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Liquid props db not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {yaml_path}: expected dict.")

    species_block = _pick_species_block(data)
    meta = {k: v for k, v in data.items() if k not in ("species", "liquids")}
    units = meta.get("units")
    if not isinstance(units, dict):
        raise ValueError("Missing top-level units mapping in liquid props DB.")
    m_unit = str(units.get("M", "")).strip().lower()
    if m_unit != "kg/mol":
        raise ValueError("Liquid props DB requires units.M = 'kg/mol'.")
    db = LiquidPropsDB(meta=meta)

    for name, raw in species_block.items():
        if not isinstance(raw, dict):
            raise ValueError(
                f"Invalid species data for '{name}': expected dict, got {type(raw)}"
            )

        spec = dict(raw)
        spec["name"] = name
        aliases = spec.get("aliases", [])
        if not isinstance(aliases, list):
            raise ValueError(
                f"Invalid aliases for '{name}': expected list, got {type(aliases)}"
            )

        if "M" in spec:
            spec["M"] = _coerce_float(spec["M"], "M", name)
        elif "MW" in spec:
            spec["M"] = _coerce_float(spec["MW"], "MW", name)
        elif "W" in spec:
            spec["M"] = _coerce_float(spec["W"], "W", name)
        else:
            raise ValueError(f"Missing molar mass (M/MW/W) for '{name}'.")
        if not (0.0 < spec["M"] < 1.0):
            raise ValueError(
                f"Invalid molar mass for '{name}': expected kg/mol, got {spec['M']}"
            )

        if "Tc" not in spec:
            raise ValueError(f"Missing Tc for '{name}'.")
        spec["Tc"] = _coerce_float(spec["Tc"], "Tc", name)
        if "Pc" not in spec:
            raise ValueError(f"Missing Pc for '{name}'.")
        spec["Pc"] = _coerce_float(spec["Pc"], "Pc", name)
        if "Tb_1atm" not in spec:
            raise ValueError(f"Missing Tb_1atm for '{name}'.")
        spec["Tb_1atm"] = _coerce_float(spec["Tb_1atm"], "Tb_1atm", name)

        prop_entries: Dict[str, Dict[str, Any]] = {}
        for prop in REQUIRED_PROPS:
            prop_entries[prop] = _get_prop_entry(spec, prop, name)

        t_valid = spec.get("T_valid")
        if t_valid is None:
            t_valid = _infer_t_valid(prop_entries)
        if t_valid is None or not isinstance(t_valid, (list, tuple)) or len(t_valid) != 2:
            raise ValueError(f"Missing T_valid for '{name}'.")
        t_low = _coerce_float(t_valid[0], "T_valid[0]", name)
        t_high = _coerce_float(t_valid[1], "T_valid[1]", name)
        if t_low >= t_high:
            raise ValueError(f"Invalid T_valid range for '{name}': {t_valid}")
        spec["T_valid"] = [t_low, t_high]

        for prop, entry in prop_entries.items():
            spec[prop] = entry

        db.add_species(name=name, spec=spec, aliases=aliases)

    return db


def get_pure_props(
    db: LiquidPropsDB,
    species: str,
    T: float,
    *,
    T_ref: float | None = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    spec = db.get_params(species)
    canon = db.canonical_name(species)
    t_ref = float(T_ref) if T_ref is not None else float(getattr(db, "meta", {}).get("T_ref", 298.15))
    key = (canon, _round_key(float(T)))
    hit = db._pure_cache.get(key)
    if hit is not None:
        props, meta = hit
        return dict(props), dict(meta)
    props, meta = eval_pure_props(float(T), spec, T_ref=t_ref)
    db._pure_cache[key] = (dict(props), dict(meta))
    return dict(props), dict(meta)


def get_mixture_props(
    db: LiquidPropsDB,
    species_list: Sequence[str],
    Y_l: Sequence[float],
    T: float,
    P: float,
    *,
    T_ref: float | None = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    return mix_props(float(T), float(P), Y_l, species_list, db, T_ref=T_ref)


def get_psat_hvap(
    db: LiquidPropsDB,
    species_list: Sequence[str],
    T: float,
) -> Tuple[np.ndarray, np.ndarray]:
    psat = np.zeros(len(species_list), dtype=np.float64)
    hvap = np.zeros(len(species_list), dtype=np.float64)
    for i, name in enumerate(species_list):
        spec = db.get_params(name)
        Tc = float(spec["Tc"])
        Pc = spec.get("Pc")
        Tmin_psat, Tmax_psat = _prop_bounds(spec, "psat")
        Tmin_hvap, Tmax_hvap = _prop_bounds(spec, "hvap")
        Ts_psat, _, _ = clip_temperature(float(T), Tmin_psat, Tmax_psat)
        Ts_hvap, _, _ = clip_temperature(float(T), Tmin_hvap, Tmax_hvap)
        key_psat = (db.canonical_name(name), _round_key(Ts_psat))
        key_hvap = (db.canonical_name(name), _round_key(Ts_hvap))
        if key_psat in db._psat_cache:
            p_val = db._psat_cache[key_psat]
        else:
            p_val = float(p2_psat(Ts_psat, spec["psat"], Tc=Tc, Pc=Pc))
            db._psat_cache[key_psat] = p_val
        if key_hvap in db._hvap_cache:
            h_val = db._hvap_cache[key_hvap]
        else:
            h_val = float(p2_hvap(Ts_hvap, spec["hvap"], Tc=Tc))
            db._hvap_cache[key_hvap] = h_val
        psat[i] = p_val
        hvap[i] = h_val
    return psat, hvap


def get_tbub(
    db: LiquidPropsDB,
    P: float,
    x_l: Sequence[float],
    liq_names: Sequence[str],
) -> Tuple[float, Dict[str, Any]]:
    x = np.asarray(x_l, dtype=np.float64)
    key = (_round_key(float(P)), tuple(round(float(v), CACHE_DIGITS) for v in x), tuple(liq_names))
    hit = db._tbub_cache.get(key)
    if hit is not None:
        Tbub, bracketed = hit
        return Tbub, {"bracketed": bracketed}

    specs = [db.get_params(name) for name in liq_names]
    tmins: List[float] = []
    tmaxs: List[float] = []
    psat_funcs = []
    for spec in specs:
        Tmin, Tmax = _prop_bounds(spec, "psat")
        tmins.append(Tmin)
        tmaxs.append(Tmax)
        Tc = float(spec["Tc"])
        Pc = spec.get("Pc")

        def _fn(T: float, spec=spec, Tc=Tc, Pc=Pc) -> float:
            return float(p2_psat(T, spec["psat"], Tc=Tc, Pc=Pc))

        psat_funcs.append(_fn)

    T_lo = float(max(tmins))
    T_hi = float(min(tmaxs))
    Tbub, meta = bubble_point_T(float(P), x, psat_funcs, T_lo, T_hi)
    bracketed = bool(meta.get("bracketed", False))
    db._tbub_cache[key] = (float(Tbub), bracketed)
    return float(Tbub), {"bracketed": bracketed}


def interface_equilibrium(
    db: LiquidPropsDB,
    P: float,
    Ts: float,
    Yl_face: Sequence[float],
    Yg_far: Sequence[float],
    *,
    liq_names: Sequence[str],
    gas_names: Sequence[str],
    idx_cond_l: Sequence[int],
    idx_cond_g: Sequence[int],
    M_l: Sequence[float],
    M_g: Sequence[float],
    Ts_guard_dT: float = 3.0,
    Ts_guard_width_K: float = 0.5,
    Ts_sat_eps_K: float = 0.01,
    eps_bg: float = 1.0e-12,
) -> Dict[str, Any]:
    from properties.p2_equilibrium import interface_equilibrium as p2_interface_equilibrium

    Yl, _ = sanitize_mass_fractions(Yl_face)
    M_l_arr = np.asarray(M_l, dtype=np.float64)
    x_l, _ = mass_to_mole_fractions(Yl, M_l_arr)
    Tbub, tbub_meta = get_tbub(db, P, x_l, liq_names)

    return p2_interface_equilibrium(
        P=float(P),
        Ts=float(Ts),
        Yl_face=Yl_face,
        Yg_far=Yg_far,
        liq_names=liq_names,
        gas_names=gas_names,
        idx_cond_l=idx_cond_l,
        idx_cond_g=idx_cond_g,
        M_l=M_l,
        M_g=M_g,
        db=db,
        Ts_guard_dT=Ts_guard_dT,
        Ts_guard_width_K=Ts_guard_width_K,
        Ts_sat_eps_K=Ts_sat_eps_K,
        eps_bg=eps_bg,
        tbub_override=Tbub,
        tbub_bracketed=bool(tbub_meta.get("bracketed", False)),
    )
def _round_key(value: float, *, digits: int = CACHE_DIGITS) -> float:
    if not np.isfinite(value):
        raise ValueError(f"Invalid cache key value: {value!r}")
    return round(float(value), digits)


def _prop_bounds(spec: Mapping[str, Any], prop: str) -> Tuple[float, float]:
    entry = spec[prop]
    if isinstance(entry, dict) and "Tmin" in entry and "Tmax" in entry:
        return float(entry["Tmin"]), float(entry["Tmax"])
    t_valid = spec.get("T_valid")
    if not t_valid or len(t_valid) != 2:
        raise ValueError(f"Missing T_valid for species '{spec.get('name', '?')}'.")
    return float(t_valid[0]), float(t_valid[1])
