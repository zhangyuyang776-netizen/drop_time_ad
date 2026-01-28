#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from properties.p2_liquid_db import load_liquid_db
from properties.p2_pure_models import k_l as model_k, rho_l as model_rho


def _slugify(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out)


def _load_baseline_series(
    npz: np.lib.npyio.NpzFile, slug: str, pressure_pa: float
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    prefix = f"{slug}__P{int(round(float(pressure_pa)))}Pa"
    t_key = f"{prefix}__T"
    if t_key not in npz:
        raise KeyError(f"Missing baseline key: {t_key}")
    temps = np.asarray(npz[t_key], dtype=np.float64)
    series: Dict[str, np.ndarray] = {}
    for prop in ("rho_l", "k_l"):
        key = f"{prefix}__{prop}"
        if key not in npz:
            raise KeyError(f"Missing baseline key: {key}")
        arr = np.asarray(npz[key], dtype=np.float64)
        if arr.ndim > 1:
            arr = np.mean(arr.reshape(arr.shape[0], -1), axis=1)
        series[prop] = arr
    return temps, series


def _interp(T_grid: np.ndarray, values: np.ndarray, T: float) -> float:
    return float(np.interp(float(T), T_grid, values))


def main() -> int:
    ap = argparse.ArgumentParser(description="Check rho/k ratio vs CoolProp baseline for a species.")
    ap.add_argument("--baseline", default="baseline/coolprop/baseline_coolprop.npz")
    ap.add_argument("--db", default="code/properties/data/liquid_props_db_fitted.yaml")
    ap.add_argument("--species", default="n-Heptane")
    ap.add_argument("--pressure", type=float, default=101325.0)
    ap.add_argument("--temps", default="300,400")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    db = load_liquid_db(args.db)
    spec = db.get_params(args.species)
    Tc = float(spec["Tc"])

    npz = np.load(str(baseline_path), allow_pickle=True)
    slug = _slugify(args.species)
    T_grid, series = _load_baseline_series(npz, slug, args.pressure)

    temps = [float(x) for x in args.temps.split(",") if x.strip()]
    print(f"species: {args.species}")
    print(f"pressure_pa: {float(args.pressure)}")
    print("T(K)  rho_base  rho_p2  rho_ratio  k_base  k_p2  k_ratio")
    for T in temps:
        rho_base = _interp(T_grid, series["rho_l"], T)
        k_base = _interp(T_grid, series["k_l"], T)
        rho_p2 = float(model_rho(T, spec["rho"], Tc=Tc))
        k_p2 = float(model_k(T, spec["k"]))
        rho_ratio = rho_p2 / rho_base if rho_base else float("nan")
        k_ratio = k_p2 / k_base if k_base else float("nan")
        print(
            f"{T:5.1f}  {rho_base:8.4g}  {rho_p2:8.4g}  {rho_ratio:9.4g}  "
            f"{k_base:8.4g}  {k_p2:8.4g}  {k_ratio:8.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
