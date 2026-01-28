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

    specs = [db.get_params(name) for name in liq_names]
    t_hi_candidates = []
    for spec in specs:
        Tmax = float(spec["psat"]["Tmax"])
        Tc = float(spec["Tc"])
        t_hi_candidates.append(min(Tmax, Tc - 1.0e-6))
    T_hi = float(min(t_hi_candidates))

    psat_tmax = []
    for spec in specs:
        Tc = float(spec["Tc"])
        psat_val = float(model_psat(T_hi, spec["psat"], Tc=Tc, Pc=float(spec["Pc"])))
        psat_tmax.append(psat_val)
    psat_tmax = np.asarray(psat_tmax, dtype=np.float64)

    for P in pressures:
        p_max = float(np.sum(x_l_mole * psat_tmax))
        if not np.isfinite(p_max) or p_max <= 0.0 or float(P) > p_max:
            summary["n_out_of_range"] += 1
            for dT in dT_list:
                summary["n_samples"] += 1
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
                        "False",
                        "False",
                        "",
                        "",
                        "",
                        "p_out_of_range",
                        "False",
                        f"{float('nan'):.6g}",
                    ]
                )
            continue

        Tbub, tb_meta = get_tbub(db, float(P), x_l_mole, liq_names)
        bracketed = bool(tb_meta.get("bracketed", False))
        if not np.isfinite(Tbub):
            summary["n_bracket_failed"] += 1
            for dT in dT_list:
                Ts = float("nan")
                summary["n_samples"] += 1
                summary["n_violations"] += 1
                rows.append(
                    [
                        mixture_id,
                        ",".join(liq_names),
                        f"{float(P):.6g}",
                        f"{Ts:.6g}",
                        f"{Tbub:.6g}",
                        f"{float('nan'):.6g}",
                        f"{Ts_guard_dT:.6g}",
                        f"{float('nan'):.6g}",
                        "False",
                        "False",
                        "",
                        "",
                        "",
                        "tbub_nan",
                        str(bracketed),
                        f"{float('nan'):.6g}",
                    ]
                )
            continue

        for dT in dT_list:
            Ts = float(Tbub) + float(dT) if np.isfinite(Tbub) else float("nan")
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
            meta = eq.get("meta", {})
            Ts_eff = float(eq.get("Ts_eff", float("nan")))
            Tbub_used = float(eq.get("Tbub", float("nan")))
            y_sum = float(meta.get("y_cond_sum", float("nan")))
            guard_active = bool(meta.get("guard_active", False))
            clamp_active = bool(meta.get("clamp_active", False))
            fallback = str(meta.get("fallback_reason", ""))
            psat_source = str(meta.get("psat_source", ""))
            hvap_source = str(meta.get("hvap_source", ""))
            latent_source = str(meta.get("latent_source", hvap_source))
            margin = float(Tbub_used - Ts_guard_dT - Ts_eff) if np.isfinite(Tbub_used) else float("nan")

            finite_ok = np.isfinite(Ts_eff) and np.isfinite(Tbub_used) and np.isfinite(y_sum)
            summary["any_nan"] = summary["any_nan"] or not finite_ok
            summary["n_samples"] += 1
            summary["n_guard_active"] += int(guard_active)
            summary["n_clamp_active"] += int(clamp_active)
            summary["max_sum_y_cond"] = max(float(summary["max_sum_y_cond"]), y_sum)
            summary["min_Ts_eff_margin"] = min(float(summary["min_Ts_eff_margin"]), margin)

            ok_margin = np.isfinite(margin) and margin >= -1.0e-12
            ok_sum = np.isfinite(y_sum) and y_sum <= 1.0 - eps_bg + 1.0e-12
            if not (finite_ok and ok_margin and ok_sum):
                summary["n_violations"] += 1

            rows.append(
                [
                    mixture_id,
                    ",".join(liq_names),
                    f"{float(P):.6g}",
                    f"{Ts:.6g}",
                    f"{Tbub_used:.6g}",
                    f"{Ts_eff:.6g}",
                    f"{Ts_guard_dT:.6g}",
                    f"{y_sum:.6g}",
                    str(guard_active),
                    str(clamp_active),
                    psat_source,
                    hvap_source,
                    latent_source,
                    fallback,
                    str(bracketed),
                    f"{margin:.6g}",
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Stress test interface guard near Tb/Tbub.")
    ap.add_argument("--db", default="code/properties/data/liquid_props_db_fitted.yaml")
    ap.add_argument("--out", default="out/p4_stress_guard")
    ap.add_argument("--pressures", default="101325,506625,1013250", help="Comma-separated pressures [Pa]")
    ap.add_argument("--dT", default="1,5,20", help="Comma-separated Ts offsets above Tb/Tbub [K]")
    ap.add_argument("--ts-guard", type=float, default=0.5)
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
        "n_guard_active": 0,
        "n_clamp_active": 0,
        "n_bracket_failed": 0,
        "n_out_of_range": 0,
        "any_nan": False,
        "max_sum_y_cond": 0.0,
        "min_Ts_eff_margin": float("inf"),
        "n_violations": 0,
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

    summary["ok"] = (
        (summary["n_violations"] == 0)
        and (not summary["any_nan"])
        and (summary["n_bracket_failed"] == 0)
    )

    csv_path = out_dir / "stress_guard_report.csv"
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
                "clamp_active",
                "psat_source",
                "hvap_source",
                "latent_source",
                "fallback_reason",
                "tb_bracketed",
                "Ts_eff_margin",
            ]
        )
        w.writerows(rows)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
