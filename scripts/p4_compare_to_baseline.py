#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from properties.p2_liquid_db import load_liquid_db
from properties.p2_pure_models import cp_l, hvap as model_hvap, k_l, mu_l, psat as model_psat, rho_l


BASELINE_PROP_KEYS = {
    "psat": "psat_l",
    "hvap": "hvap_l",
    "rho": "rho_l",
    "cp": "cp_l",
    "mu": "mu_l",
    "k": "k_l",
}

DEFAULT_THRESHOLDS = {
    "psat": {"rms": 0.20, "max": 0.50},
    "hvap": {"rms": 0.10, "max": 0.20},
    "rho": {"rms": 0.15, "max": 0.30},
    "cp": {"rms": 0.15, "max": 0.30},
    "mu": {"rms": 0.15, "max": 0.30},
    "k": {"rms": 0.15, "max": 0.30},
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


def _series_for(
    npz: np.lib.npyio.NpzFile, slug: str, pressure: float, key: str
) -> Tuple[np.ndarray, np.ndarray] | None:
    prefix = f"{slug}__P{int(round(float(pressure)))}Pa"
    t_key = f"{prefix}__T"
    v_key = f"{prefix}__{key}"
    if t_key not in npz or v_key not in npz:
        return None
    temps = np.asarray(npz[t_key], dtype=np.float64)
    vals = _flatten_series(np.asarray(npz[v_key], dtype=np.float64))
    return temps, vals


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


def _filter_series(T: np.ndarray, V: np.ndarray, Tmin: float, Tmax: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(T) & np.isfinite(V)
    mask &= T >= float(Tmin)
    mask &= T <= float(Tmax)
    mask &= V > 0.0
    return T[mask], V[mask]


def _eval_prop(prop: str, T: np.ndarray, spec: Dict[str, object]) -> np.ndarray:
    Tc = float(spec["Tc"])
    Pc = float(spec["Pc"]) if "Pc" in spec else None
    entry = spec[prop]
    if prop == "psat":
        return np.asarray(model_psat(T, entry, Tc=Tc, Pc=Pc), dtype=np.float64)
    if prop == "hvap":
        return np.asarray(model_hvap(T, entry, Tc=Tc), dtype=np.float64)
    if prop == "rho":
        return np.asarray(rho_l(T, entry, Tc=Tc), dtype=np.float64)
    if prop == "cp":
        return np.asarray(cp_l(T, entry), dtype=np.float64)
    if prop == "mu":
        return np.asarray(mu_l(T, entry), dtype=np.float64)
    if prop == "k":
        return np.asarray(k_l(T, entry), dtype=np.float64)
    raise ValueError(f"Unsupported property '{prop}'.")


def _rel_stats(pred: np.ndarray, obs: np.ndarray) -> Tuple[float, float]:
    rel = (pred - obs) / obs
    rms = float(np.sqrt(np.mean(rel**2)))
    max_rel = float(np.max(np.abs(rel)))
    return rms, max_rel


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare P2 property DB against CoolProp baseline.")
    ap.add_argument("--baseline", required=True, help="Path to baseline_coolprop.npz")
    ap.add_argument("--db", required=True, help="Path to liquid_props_db.yaml")
    ap.add_argument("--out", default="out/p4_baseline_compare", help="Output directory")
    ap.add_argument("--tb-guard", type=float, default=0.5, help="Temperature margin below Tb for comparisons.")
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "p2_vs_coolprop.csv"
    summary_path = out_dir / "summary.json"

    rows: List[List[str]] = []
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    per_prop: Dict[str, Dict[str, int]] = {}

    for species in species_list:
        if not db.has_species(species):
            for prop in BASELINE_PROP_KEYS:
                rows.append([species, "", prop, "0", "0", "", "", "SKIP", "missing_in_db"])
                counts["SKIP"] += 1
                per_prop.setdefault(prop, {"PASS": 0, "FAIL": 0, "SKIP": 0})["SKIP"] += 1
            continue

        spec = db.get_params(species)
        slug = _slugify(species)
        tb_map: Dict[float, float | None] = {}
        for pressure in pressures:
            psat_series = _series_for(npz, slug, pressure, "psat_l")
            if psat_series is None:
                tb_map[float(pressure)] = None
                continue
            T_psat, psat_vals = psat_series
            tb_map[float(pressure)] = _estimate_tb(T_psat, psat_vals, float(pressure))

        for pressure in pressures:
            for prop, base_key in BASELINE_PROP_KEYS.items():
                per_prop.setdefault(prop, {"PASS": 0, "FAIL": 0, "SKIP": 0})
                series = _series_for(npz, slug, pressure, base_key)
                if series is None:
                    rows.append([species, str(float(pressure)), prop, "0", "0", "", "", "SKIP", "missing_baseline"])
                    counts["SKIP"] += 1
                    per_prop[prop]["SKIP"] += 1
                    continue

                T_raw, V_raw = series
                entry = spec[prop]
                Tmin = float(entry["Tmin"])
                Tmax = float(entry["Tmax"])
                tb = tb_map.get(float(pressure))
                if tb is not None:
                    Tmax = min(Tmax, float(tb) - float(args.tb_guard))
                T, V = _filter_series(T_raw, V_raw, Tmin, Tmax)
                if T.size == 0:
                    rows.append([species, str(float(pressure)), prop, str(T_raw.size), "0", "", "", "SKIP", "no_valid_points"])
                    counts["SKIP"] += 1
                    per_prop[prop]["SKIP"] += 1
                    continue

                pred = _eval_prop(prop, T, spec)
                rms, max_rel = _rel_stats(pred, V)
                thr = DEFAULT_THRESHOLDS.get(prop, {})
                ok = rms <= float(thr.get("rms", 1.0e9)) and max_rel <= float(thr.get("max", 1.0e9))
                status = "PASS" if ok else "FAIL"
                counts[status] += 1
                per_prop[prop][status] += 1

                rows.append(
                    [
                        species,
                        str(float(pressure)),
                        prop,
                        str(T_raw.size),
                        str(T.size),
                        f"{rms:.6g}",
                        f"{max_rel:.6g}",
                        status,
                        "",
                    ]
                )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["species", "pressure_pa", "property", "n_total", "n_used", "rms_rel", "max_rel", "status", "note"])
        w.writerows(rows)

    summary = {
        "baseline": str(baseline_path),
        "db": str(db_path),
        "thresholds": DEFAULT_THRESHOLDS,
        "counts": counts,
        "per_property": per_prop,
        "PASS": counts["FAIL"] == 0,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
