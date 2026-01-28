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

from properties.p2_liquid_db import get_tbub, load_liquid_db, interface_equilibrium
from properties.p2_pure_models import psat as model_psat


BACKGROUND_GAS = "N2"
BACKGROUND_MW = 0.028
EPS = 1.0e-12


def _mole_to_mass(x: np.ndarray, M: np.ndarray) -> np.ndarray:
    numer = x * M
    denom = float(np.sum(numer))
    if denom <= 0.0 or not np.isfinite(denom):
        out = np.zeros_like(x)
        if out.size:
            out[0] = 1.0
        return out
    return numer / denom


def _canonicalize(db, name: str) -> str:
    return db.canonical_name(name)


def _build_gas_arrays(db, liq_names: List[str]) -> Tuple[List[str], List[int], np.ndarray]:
    gas_names = list(liq_names) + [BACKGROUND_GAS]
    idx_cond_g = list(range(len(liq_names)))
    M_g = np.zeros(len(gas_names), dtype=np.float64)
    for i, name in enumerate(gas_names):
        if name == BACKGROUND_GAS:
            M_g[i] = BACKGROUND_MW
        else:
            spec = db.get_params(name)
            M_g[i] = float(spec["M"])
    return gas_names, idx_cond_g, M_g


def _psat_max(db, liq_names: List[str], x_l_mole: np.ndarray) -> Tuple[float, float]:
    specs = [db.get_params(name) for name in liq_names]
    t_hi_candidates = []
    for spec in specs:
        Tmax = float(spec["psat"]["Tmax"])
        Tc = float(spec["Tc"])
        t_hi_candidates.append(min(Tmax, Tc - 1.0e-6))
    T_hi = float(min(t_hi_candidates))
    psat_vals = []
    for spec in specs:
        Tc = float(spec["Tc"])
        Pc = spec.get("Pc")
        psat_vals.append(float(model_psat(T_hi, spec["psat"], Tc=Tc, Pc=Pc)))
    psat_vals = np.asarray(psat_vals, dtype=np.float64)
    p_max = float(np.sum(x_l_mole * psat_vals))
    return p_max, T_hi


def _eval_ok(
    eq: Dict[str, object],
    *,
    Ts_guard_dT: float,
    eps_bg: float,
) -> Tuple[bool, str, Dict[str, float]]:
    meta = eq.get("meta", {}) if isinstance(eq.get("meta"), dict) else {}
    Ts_eff = float(eq.get("Ts_eff", float("nan")))
    Tbub = float(eq.get("Tbub", float("nan")))
    y_cond = np.asarray(eq.get("y_cond", []), dtype=np.float64)
    Yg_eq = np.asarray(eq.get("Yg_eq", []), dtype=np.float64)
    y_sum = float(meta.get("y_cond_sum", float(np.sum(y_cond))))

    metrics = {
        "Ts_eff": Ts_eff,
        "Tbub": Tbub,
        "sum_y_cond": y_sum,
    }

    finite_ok = (
        np.isfinite(Ts_eff)
        and np.isfinite(Tbub)
        and np.isfinite(y_sum)
        and np.all(np.isfinite(y_cond))
        and np.all(np.isfinite(Yg_eq))
    )
    if not finite_ok:
        return False, "nonfinite", metrics

    if np.any(y_cond < -EPS) or np.any(Yg_eq < -EPS):
        return False, "negative_fraction", metrics

    margin = float(Tbub - Ts_guard_dT - Ts_eff)
    if margin < -1.0e-12:
        return False, "guard_margin", metrics

    if y_sum > 1.0 - eps_bg + 1.0e-12:
        return False, "sum_y_exceed", metrics

    s_Y = float(np.sum(Yg_eq))
    if not np.isfinite(s_Y) or abs(s_Y - 1.0) > 5.0e-6:
        return False, "yg_norm", metrics

    return True, "", metrics


def _run_case(
    *,
    db,
    mixture_id: str,
    liq_names: List[str],
    x_l_mole: np.ndarray,
    pressures: Iterable[float],
    dT_list: Iterable[float],
    Ts_guard_dT: float,
    eps_bg: float,
    rows: List[List[str]],
    summary: Dict[str, float | int | bool],
) -> None:
    M_l = np.array([float(db.get_params(n)["M"]) for n in liq_names], dtype=np.float64)
    Yl_face = _mole_to_mass(x_l_mole, M_l)

    gas_names, idx_cond_g, M_g = _build_gas_arrays(db, liq_names)
    idx_cond_l = list(range(len(liq_names)))
    Yg_far = np.zeros(len(gas_names), dtype=np.float64)
    Yg_far[-1] = 1.0

    p_max, _ = _psat_max(db, liq_names, x_l_mole)
    for P in pressures:
        if not np.isfinite(p_max) or p_max <= 0.0 or float(P) > p_max:
            summary["n_skipped"] += len(list(dT_list))
            for dT in dT_list:
                rows.append(
                    [
                        mixture_id,
                        ",".join(liq_names),
                        f"{float(P):.6g}",
                        f"{float('nan'):.6g}",
                        f"{float('nan'):.6g}",
                        f"{float('nan'):.6g}",
                        f"{Ts_guard_dT:.6g}",
                        f"{float('nan'):.6g}",
                        "",
                        "",
                        "",
                        "",
                        "False",
                        "p_out_of_range",
                    ]
                )
            continue

        Tbub, tb_meta = get_tbub(db, float(P), x_l_mole, liq_names)
        bracketed = bool(tb_meta.get("bracketed", False))
        if not np.isfinite(Tbub) or not bracketed:
            summary["n_skipped"] += len(list(dT_list))
            for dT in dT_list:
                rows.append(
                    [
                        mixture_id,
                        ",".join(liq_names),
                        f"{float(P):.6g}",
                        f"{float('nan'):.6g}",
                        f"{Tbub:.6g}",
                        f"{float('nan'):.6g}",
                        f"{Ts_guard_dT:.6g}",
                        f"{float('nan'):.6g}",
                        "",
                        "",
                        "",
                        "",
                        "False",
                        "tbub_unbracketed",
                    ]
                )
            continue

        for dT in dT_list:
            Ts = float(Tbub) + float(dT)
            eq = interface_equilibrium(
                db=db,
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
                Ts_guard_dT=Ts_guard_dT,
                eps_bg=eps_bg,
            )
            meta = eq.get("meta", {}) if isinstance(eq.get("meta"), dict) else {}
            ok, reason, metrics = _eval_ok(eq, Ts_guard_dT=Ts_guard_dT, eps_bg=eps_bg)

            summary["n_samples"] += 1
            summary["n_fail"] += int(not ok)
            summary["any_nan"] = summary["any_nan"] or (reason == "nonfinite")
            summary["max_sum_y_cond"] = max(
                float(summary["max_sum_y_cond"]), float(metrics.get("sum_y_cond", 0.0))
            )

            rows.append(
                [
                    mixture_id,
                    ",".join(liq_names),
                    f"{float(P):.6g}",
                    f"{float(Ts):.6g}",
                    f"{float(metrics['Tbub']):.6g}",
                    f"{float(metrics['Ts_eff']):.6g}",
                    f"{Ts_guard_dT:.6g}",
                    f"{float(metrics['sum_y_cond']):.6g}",
                    str(bool(meta.get("guard_active", False))),
                    str(bool(meta.get("cond_scaled", False))),
                    str(bool(meta.get("bg_floor_applied", False))),
                    str(bool(meta.get("bg_floor_relaxed", False))),
                    str(bool(ok)),
                    reason,
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="P5 stress test near boiling for interface equilibrium.")
    ap.add_argument("--db", default="code/properties/data/liquid_props_db_fitted.yaml")
    ap.add_argument("--out", default="out/p5_stress_boil_zone")
    ap.add_argument("--pressures", default="101325,506625,1013250", help="Comma-separated pressures [Pa]")
    ap.add_argument("--dT", default="1,5,20", help="Comma-separated Ts offsets above Tb/Tbub [K]")
    ap.add_argument("--ts-guard", type=float, default=3.0)
    ap.add_argument("--eps-bg", type=float, default=1.0e-5)
    args = ap.parse_args()

    db = load_liquid_db(args.db)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pressures = [float(p.strip()) for p in args.pressures.split(",") if p.strip()]
    dT_list = [float(x.strip()) for x in args.dT.split(",") if x.strip()]

    rows: List[List[str]] = []
    summary: Dict[str, float | int | bool] = {
        "ok": True,
        "n_samples": 0,
        "n_fail": 0,
        "n_skipped": 0,
        "any_nan": False,
        "max_sum_y_cond": 0.0,
    }

    # Pure components
    for name in ["NC7H16", "NC12H26"]:
        canon = _canonicalize(db, name)
        _run_case(
            db=db,
            mixture_id=canon,
            liq_names=[canon],
            x_l_mole=np.array([1.0], dtype=np.float64),
            pressures=pressures,
            dT_list=dT_list,
            Ts_guard_dT=float(args.ts_guard),
            eps_bg=float(args.eps_bg),
            rows=rows,
            summary=summary,
        )

    # Mixtures
    nc7 = _canonicalize(db, "NC7H16")
    nc12 = _canonicalize(db, "NC12H26")
    for x_nc7 in [0.1, 0.5, 0.9]:
        x_l = np.array([float(x_nc7), float(1.0 - x_nc7)], dtype=np.float64)
        mix_id = f"{nc7}:{x_nc7:.2f}_{nc12}:{1.0-x_nc7:.2f}"
        _run_case(
            db=db,
            mixture_id=mix_id,
            liq_names=[nc7, nc12],
            x_l_mole=x_l,
            pressures=pressures,
            dT_list=dT_list,
            Ts_guard_dT=float(args.ts_guard),
            eps_bg=float(args.eps_bg),
            rows=rows,
            summary=summary,
        )

    summary["ok"] = summary["n_fail"] == 0 and (not summary["any_nan"])

    csv_path = out_dir / "stress_boil_zone_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mixture_id",
                "liq_names",
                "pressure_pa",
                "Ts",
                "Tbub",
                "Ts_eff",
                "Ts_guard_dT",
                "sum_y_cond",
                "guard_active",
                "cond_scaled",
                "bg_floor_applied",
                "bg_floor_relaxed",
                "ok",
                "reason",
            ]
        )
        w.writerows(rows)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {summary_path}")
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
