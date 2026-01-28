#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from properties.p2_liquid_db import load_liquid_db
from properties.p2_pure_models import cp_l, hvap as model_hvap, k_l, psat as model_psat, rho_l


BASELINE_PROP_KEYS = {
    "psat": "psat_l",
    "hvap": "hvap_l",
    "rho": "rho_l",
    "cp": "cp_l",
    "k": "k_l",
}


def _slugify(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out)


def _flatten_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim <= 1:
        return arr
    return np.mean(arr.reshape(arr.shape[0], -1), axis=1)


def _estimate_tb(T: np.ndarray, psat: np.ndarray, pressure: float) -> float | None:
    if T.size == 0 or psat.size == 0:
        return None
    idx = np.where(psat >= pressure)[0]
    if idx.size == 0:
        return None
    j = int(idx[0])
    if j == 0:
        return float(T[0])
    t0, t1 = float(T[j - 1]), float(T[j])
    p0, p1 = float(psat[j - 1]), float(psat[j])
    if p1 == p0:
        return float(t1)
    return t0 + (pressure - p0) * (t1 - t0) / (p1 - p0)


def _collect_baseline_series(
    npz: np.lib.npyio.NpzFile,
    species_slug: str,
    pressures: Iterable[float],
    key: str,
    *,
    tb_map: Dict[float, float | None],
    t_guard: float,
    extra_ref: Dict[str, np.ndarray] | None,
    extra_prop: str,
) -> Tuple[np.ndarray, np.ndarray]:
    temps: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for P in pressures:
        prefix = f"{species_slug}__P{int(round(float(P)))}Pa"
        t_key = f"{prefix}__T"
        v_key = f"{prefix}__{key}"
        if t_key not in npz or v_key not in npz:
            continue
        T = np.asarray(npz[t_key], dtype=np.float64)
        V = _flatten_series(np.asarray(npz[v_key], dtype=np.float64))
        if T.size == 0 or V.size == 0:
            continue
        tb = tb_map.get(float(P))
        if tb is not None:
            T_eff = T[T <= tb - t_guard]
            V_eff = V[: T_eff.size]
            if T_eff.size == 0:
                continue
            temps.append(T_eff)
            vals.append(V_eff)
        else:
            temps.append(T)
            vals.append(V)
    if extra_ref is not None:
        t_key = f"ref__{species_slug}__T"
        v_key = f"ref__{species_slug}__{extra_prop}"
        if t_key in extra_ref and v_key in extra_ref:
            temps.append(np.asarray(extra_ref[t_key], dtype=np.float64))
            vals.append(np.asarray(extra_ref[v_key], dtype=np.float64))
    if not temps:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.concatenate(temps), np.concatenate(vals)


def _filter_series(T: np.ndarray, V: np.ndarray, Tmin: float, Tmax: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(T) & np.isfinite(V)
    mask &= T >= Tmin
    mask &= T <= Tmax
    mask &= V > 0.0
    return T[mask], V[mask]


def _fit_yaws_ln(T: np.ndarray, P: np.ndarray, *, E: float) -> Dict[str, float]:
    X = np.column_stack(
        [
            np.ones_like(T),
            1.0 / T,
            np.log(T),
            np.power(T, E),
        ]
    )
    y = np.log(P)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"A": float(coeffs[0]), "B": float(coeffs[1]), "C": float(coeffs[2]), "D": float(coeffs[3]), "E": float(E)}


def _fit_watson(T: np.ndarray, Hvap: np.ndarray, *, Tc: float, Tref: float) -> Dict[str, float]:
    term_den = 1.0 - Tref / Tc
    if term_den <= 0.0:
        raise ValueError("Invalid Watson reference: Tref >= Tc.")
    ratio = (1.0 - T / Tc) / term_den
    mask = ratio > 0.0
    T_use = T[mask]
    Hvap_use = Hvap[mask]
    ratio_use = ratio[mask]
    if T_use.size < 2:
        raise ValueError("Not enough points for Watson fit.")
    x = np.log(ratio_use)
    y = np.log(Hvap_use)
    X = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"Tref": float(Tref), "Hvap_Tref": float(np.exp(coeffs[0])), "n": float(coeffs[1])}


def _fit_rackett(T: np.ndarray, rho: np.ndarray, *, Tc: float) -> Dict[str, float]:
    Tr = T / Tc
    exponent = -np.power(1.0 - Tr, 2.0 / 7.0)
    X = np.column_stack([np.ones_like(exponent), exponent])
    y = np.log(rho)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"rho_c": float(np.exp(coeffs[0])), "Z_RA": float(np.exp(coeffs[1]))}


def _fit_poly(T: np.ndarray, V: np.ndarray, order: int) -> list[float]:
    X = np.column_stack([np.power(T, i) for i in range(order + 1)])
    coeffs, *_ = np.linalg.lstsq(X, V, rcond=None)
    return [float(c) for c in coeffs]


def _rel_stats(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    rel = (pred - obs) / obs
    return {
        "rms_rel": float(np.sqrt(np.mean(rel**2))),
        "max_rel": float(np.max(np.abs(rel))),
    }


def _eval_prop(prop: str, T: np.ndarray, entry: Dict[str, Any], *, Tc: float, Pc: float | None) -> np.ndarray:
    if prop == "psat":
        return np.asarray(model_psat(T, entry, Tc=Tc, Pc=Pc), dtype=np.float64)
    if prop == "hvap":
        return np.asarray(model_hvap(T, entry, Tc=Tc), dtype=np.float64)
    if prop == "rho":
        return np.asarray(rho_l(T, entry, Tc=Tc), dtype=np.float64)
    if prop == "cp":
        return np.asarray(cp_l(T, entry), dtype=np.float64)
    if prop == "k":
        return np.asarray(k_l(T, entry), dtype=np.float64)
    raise ValueError(f"Unsupported property '{prop}'.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit liquid property DB coefficients to CoolProp baseline.")
    ap.add_argument("--baseline", required=True, help="Path to baseline_coolprop.npz")
    ap.add_argument("--db", required=True, help="Path to liquid_props_db.yaml")
    ap.add_argument("--out", default="", help="Output YAML path (default: *_fitted.yaml)")
    ap.add_argument("--summary", default="", help="Optional summary JSON path.")
    ap.add_argument("--tb-guard", type=float, default=0.5, help="Temperature margin below Tb for fitting.")
    ap.add_argument("--extra-ref", default="", help="Optional extra reference NPZ with ref__<species>__* keys.")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    db_path = Path(args.db)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    npz = np.load(baseline_path, allow_pickle=True)
    pressures = np.asarray(npz["meta_pressures_pa"], dtype=np.float64)
    species_list = [str(s) for s in np.asarray(npz["meta_species"]).tolist()]

    db = load_liquid_db(db_path)
    with db_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    species_block = raw.get("liquids") or raw.get("species")
    if not isinstance(species_block, dict):
        raise ValueError("DB missing 'liquids'/'species' block.")

    extra_ref = None
    extra_max: Dict[str, float] = {}
    if args.extra_ref:
        extra_path = Path(args.extra_ref)
        if not extra_path.exists():
            raise FileNotFoundError(f"Extra reference not found: {extra_path}")
        extra_ref = dict(np.load(extra_path, allow_pickle=True))
        for key, arr in extra_ref.items():
            if not key.startswith("ref__") or not key.endswith("__T"):
                continue
            parts = key.split("__")
            if len(parts) < 3:
                continue
            slug = parts[1]
            try:
                extra_max[slug] = max(extra_max.get(slug, -np.inf), float(np.max(arr)))
            except Exception:
                continue

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for baseline_species in species_list:
        canon = db.canonical_name(baseline_species)
        spec = species_block[canon]
        Tc = float(spec["Tc"])
        Pc = float(spec.get("Pc", 0.0)) if "Pc" in spec else None
        spec_summary: Dict[str, Dict[str, float]] = {}

        slug = _slugify(baseline_species)
        tb_map: Dict[float, float | None] = {}
        for P in pressures:
            prefix = f"{slug}__P{int(round(float(P)))}Pa"
            t_key = f"{prefix}__T"
            p_key = f"{prefix}__psat_l"
            if t_key not in npz or p_key not in npz:
                tb_map[float(P)] = None
                continue
            T_psat = np.asarray(npz[t_key], dtype=np.float64)
            psat_vals = _flatten_series(np.asarray(npz[p_key], dtype=np.float64))
            tb_map[float(P)] = _estimate_tb(T_psat, psat_vals, float(P))

        for prop, key in BASELINE_PROP_KEYS.items():
            entry = dict(spec[prop])
            Tmin = float(entry["Tmin"])
            Tmax = float(entry["Tmax"])
            if extra_ref is not None:
                extra_key = f"ref__{slug}__{prop}"
                extra_T_key = f"ref__{slug}__T"
                if extra_key in extra_ref and extra_T_key in extra_ref:
                    Tmax = max(Tmax, float(np.max(extra_ref[extra_T_key])))
                    entry["Tmax"] = float(Tmax)
            T, V = _collect_baseline_series(
                npz,
                slug,
                pressures,
                key,
                tb_map=tb_map,
                t_guard=float(args.tb_guard),
                extra_ref=extra_ref,
                extra_prop=prop,
            )
            T, V = _filter_series(T, V, Tmin, Tmax)
            if T.size < 5:
                continue

            model = str(entry.get("model", "")).strip().lower()
            if prop == "psat" and model in ("yaws_ln", "yaws"):
                coeffs = entry.get("coeffs", {})
                E = float(coeffs.get("E", 1.0)) if isinstance(coeffs, dict) else 1.0
                entry["coeffs"] = _fit_yaws_ln(T, V, E=E)
            elif prop == "hvap" and model == "watson":
                Tref = float(entry.get("Tref", spec.get("Tb_1atm", Tmin)))
                entry.update(_fit_watson(T, V, Tc=Tc, Tref=Tref))
            elif prop == "rho" and model == "rackett":
                entry["coeffs"] = _fit_rackett(T, V, Tc=Tc)
            elif prop == "cp" and model in ("poly4", "poly"):
                entry["coeffs"] = _fit_poly(T, V, order=4 if model == "poly4" else 3)
            elif prop == "k" and model in ("poly2", "poly"):
                entry["coeffs"] = _fit_poly(T, V, order=2 if model == "poly2" else 3)
            else:
                continue

            entry["source"] = "fitted_to_coolprop_baseline"
            entry["ref"] = str(baseline_path)
            spec[prop] = entry

            pred = _eval_prop(prop, T, entry, Tc=Tc, Pc=Pc)
            spec_summary[prop] = _rel_stats(pred, V)

            if spec_summary:
                summary[canon] = spec_summary

    out_path = Path(args.out) if args.out else db_path.with_name(db_path.stem + "_fitted.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            meta_summary = {
                "extra_ref": str(Path(args.extra_ref)) if args.extra_ref else "",
                "tb_guard": float(args.tb_guard),
            }
            payload = {"meta": meta_summary, "fit": summary}
            json.dump(payload, f, indent=2, sort_keys=True)

    print(f"[OK] wrote: {out_path}")
    if args.summary:
        print(f"[OK] wrote: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
