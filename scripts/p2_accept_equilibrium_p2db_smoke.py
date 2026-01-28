#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from driver.run_evap_case import _load_case_config, _maybe_fill_gas_species
from properties.equilibrium import build_equilibrium_model, interface_equilibrium
from properties.gas import build_gas_model
from properties.p2_liquid_db import load_liquid_db


def _build_mass_fractions(names: List[str], values: Dict[str, float]) -> np.ndarray:
    Y = np.zeros(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        if name in values:
            Y[i] = float(values[name])
    s = float(np.sum(Y))
    if s <= 0.0:
        if Y.size:
            Y[0] = 1.0
        return Y
    return Y / s


def _load_p2db(cfg) -> Any:
    liq_cfg = getattr(cfg.physics, "liquid", None)
    db_file = str(getattr(liq_cfg, "db_file", "") or "")
    if not db_file:
        raise ValueError("physics.liquid.db_file is required for p2db acceptance.")
    path = Path(db_file)
    if not path.is_absolute():
        base = getattr(cfg.paths, "mechanism_dir", None)
        if isinstance(base, Path):
            path = (base / path).resolve()
        else:
            path = path.resolve()
    return load_liquid_db(path)


def _liq_molar_masses(cfg, p2db) -> np.ndarray:
    liq_names = list(cfg.species.liq_species)
    mw_map = dict(getattr(cfg.species, "mw_kg_per_mol", {}))
    out = np.zeros(len(liq_names), dtype=np.float64)
    for i, name in enumerate(liq_names):
        if name in mw_map:
            out[i] = float(mw_map[name])
        else:
            out[i] = float(p2db.get_params(name)["M"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="P2 acceptance smoke for p2db equilibrium.")
    ap.add_argument("case_yaml", help="Path to case YAML file.")
    ap.add_argument("--out", default="out/p2_accept", help="Output directory.")
    ap.add_argument("--n-samples", type=int, default=50, help="Number of Ts samples.")
    args = ap.parse_args()

    cfg = _load_case_config(args.case_yaml)
    gas_model = build_gas_model(cfg)
    _maybe_fill_gas_species(cfg, gas_model)

    p2db = _load_p2db(cfg)

    Ns_g = len(cfg.species.gas_species_full)
    Ns_l = len(cfg.species.liq_species)
    M_g = np.asarray(gas_model.gas.molecular_weights, dtype=np.float64) / 1000.0
    M_l = _liq_molar_masses(cfg, p2db)

    eq_model = build_equilibrium_model(cfg, Ns_g=Ns_g, Ns_l=Ns_l, M_g=M_g, M_l=M_l)

    liq_names = list(cfg.species.liq_species)
    specs = [p2db.get_params(name) for name in liq_names]
    tmins = []
    tmaxs = []
    for spec in specs:
        psat_entry = spec.get("psat", {})
        if "Tmin" in psat_entry and "Tmax" in psat_entry:
            tmins.append(float(psat_entry["Tmin"]))
            tmaxs.append(float(psat_entry["Tmax"]))
        else:
            t_valid = spec.get("T_valid", [250.0, 600.0])
            tmins.append(float(t_valid[0]))
            tmaxs.append(float(t_valid[1]))
    T_lo = max(tmins)
    T_hi = min(tmaxs)
    temps = np.linspace(T_lo, T_hi, max(int(args.n_samples), 2))

    P = float(getattr(cfg.initial, "P_inf", 101325.0))
    Yl_face = _build_mass_fractions(liq_names, dict(getattr(cfg.initial, "Yl", {})))
    Yg_far = _build_mass_fractions(list(cfg.species.gas_species_full), dict(getattr(cfg.initial, "Yg", {})))

    n_guard_active = 0
    n_clamp_active = 0
    any_nan = False
    max_sum_y_cond = 0.0
    min_margin = None

    for Ts in temps:
        res = interface_equilibrium(
            eq_model,
            Ts=float(Ts),
            Pg=P,
            Yl_face=Yl_face,
            Yg_face=Yg_far,
        )
        meta = res.meta or {}
        guard_active = bool(meta.get("guard_active", False))
        clamp_active = bool(meta.get("clamp_active", False))
        n_guard_active += int(guard_active)
        n_clamp_active += int(clamp_active)

        if not (np.all(np.isfinite(res.Yg_eq)) and np.all(np.isfinite(res.y_cond))):
            any_nan = True

        y_sum = float(np.sum(res.y_cond)) if res.y_cond.size else 0.0
        max_sum_y_cond = max(max_sum_y_cond, y_sum)

        Tbub = meta.get("Tbub", float("nan"))
        Ts_eff = meta.get("Ts_eff", float("nan"))
        Ts_guard_dT = float(meta.get("Ts_guard_dT", getattr(eq_model, "Ts_guard_dT", 3.0)))
        if np.isfinite(Tbub) and np.isfinite(Ts_eff):
            margin = float(Tbub) - Ts_guard_dT - float(Ts_eff)
            min_margin = margin if min_margin is None else min(min_margin, margin)
        else:
            any_nan = True

    eps_bg = float(getattr(eq_model, "eps_bg", 1.0e-12))
    ok = True
    if any_nan:
        ok = False
    if max_sum_y_cond > 1.0 - eps_bg + 1.0e-12:
        ok = False
    if min_margin is None or min_margin < -1.0e-12:
        ok = False

    summary = {
        "ok": ok,
        "n_samples": int(len(temps)),
        "n_guard_active": int(n_guard_active),
        "n_clamp_active": int(n_clamp_active),
        "any_nan": bool(any_nan),
        "max_sum_y_cond": float(max_sum_y_cond),
        "min_Ts_eff_margin": float(min_margin) if min_margin is not None else float("nan"),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[OK] wrote: {out_path}")
    if not ok:
        print("[FAIL] p2 acceptance failed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
