#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def _latest_run_dir(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _parse_bool(value: str) -> Optional[bool]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return None


def _parse_interface_diag(
    path: Path,
    *,
    eps_bg: float,
    ts_guard_dT: float,
    expect_source: str,
) -> Dict[str, object]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    any_nan = False
    n_failed_eq = 0
    n_source_mismatch = 0
    n_sum_y_exceed = 0
    n_margin_violation = 0
    n_finite_false = 0
    n_penalty = 0
    n_lambda_cap_ts = 0
    max_ts_trial_minus_upper = float("-inf")
    max_sum_y = 0.0
    min_margin = float("inf")
    min_Ts = float("inf")
    max_Ts = float("-inf")
    min_Ts_eff = float("inf")
    max_Ts_eff = float("-inf")
    min_Tbub = float("inf")
    max_Tbub = float("-inf")
    min_energy_res = float("inf")
    max_energy_res = float("-inf")

    for row in rows:
        eq_source = str(row.get("eq_source", ""))
        if eq_source and eq_source != "computed":
            n_failed_eq += 1

        psat_source = str(row.get("psat_source", ""))
        hvap_source = str(row.get("hvap_source", ""))
        latent_source = str(row.get("latent_source", ""))
        if psat_source or hvap_source or latent_source:
            sources = [psat_source, hvap_source, latent_source]
            if expect_source != "any":
                if not all(s == expect_source for s in sources):
                    n_source_mismatch += 1
            elif not (psat_source == hvap_source == latent_source):
                n_source_mismatch += 1

        sum_y = _to_float(row.get("sum_y_cond", ""))
        if np.isfinite(sum_y):
            max_sum_y = max(max_sum_y, sum_y)
            if sum_y > 1.0 - eps_bg + 1.0e-12:
                n_sum_y_exceed += 1
        else:
            any_nan = True

        Ts = _to_float(row.get("Ts", ""))
        Ts_eff = _to_float(row.get("Ts_eff", ""))
        Tbub = _to_float(row.get("Tbub", ""))
        if np.isfinite(Ts):
            min_Ts = min(min_Ts, Ts)
            max_Ts = max(max_Ts, Ts)
        if np.isfinite(Ts_eff):
            min_Ts_eff = min(min_Ts_eff, Ts_eff)
            max_Ts_eff = max(max_Ts_eff, Ts_eff)
        if np.isfinite(Tbub):
            min_Tbub = min(min_Tbub, Tbub)
            max_Tbub = max(max_Tbub, Tbub)

        if np.isfinite(Ts_eff) and np.isfinite(Tbub):
            margin = float(Tbub - ts_guard_dT - Ts_eff)
            min_margin = min(min_margin, margin)
            if margin < -1.0e-12:
                n_margin_violation += 1
        else:
            any_nan = True

        energy_res = _to_float(row.get("energy_res", ""))
        if np.isfinite(energy_res):
            min_energy_res = min(min_energy_res, energy_res)
            max_energy_res = max(max_energy_res, energy_res)

        finite_ok = _parse_bool(row.get("finite_ok", ""))
        if finite_ok is False:
            n_finite_false += 1

        pen_raw = row.get("n_penalty_residual", "")
        try:
            n_penalty += int(float(pen_raw)) if pen_raw != "" else 0
        except Exception:
            pass

        cap_raw = row.get("n_lambda_cap_ts", "")
        try:
            n_lambda_cap_ts += int(float(cap_raw)) if cap_raw != "" else 0
        except Exception:
            pass

        delta_raw = row.get("max_Ts_trial_minus_upper", "")
        try:
            delta = float(delta_raw)
            if np.isfinite(delta):
                max_ts_trial_minus_upper = max(max_ts_trial_minus_upper, delta)
        except Exception:
            pass

    if min_Ts == float("inf"):
        min_Ts = float("nan")
    if max_Ts == float("-inf"):
        max_Ts = float("nan")
    if min_Ts_eff == float("inf"):
        min_Ts_eff = float("nan")
    if max_Ts_eff == float("-inf"):
        max_Ts_eff = float("nan")
    if min_Tbub == float("inf"):
        min_Tbub = float("nan")
    if max_Tbub == float("-inf"):
        max_Tbub = float("nan")
    if min_margin == float("inf"):
        min_margin = float("nan")
    if max_ts_trial_minus_upper == float("-inf"):
        max_ts_trial_minus_upper = float("nan")
    if min_energy_res == float("inf"):
        min_energy_res = float("nan")
    if max_energy_res == float("-inf"):
        max_energy_res = float("nan")

    return {
        "n_rows": len(rows),
        "any_nan": any_nan,
        "n_failed_eq": n_failed_eq,
        "n_source_mismatch": n_source_mismatch,
        "n_sum_y_exceed": n_sum_y_exceed,
        "n_margin_violation": n_margin_violation,
        "n_finite_false": n_finite_false,
        "n_penalty_residual": n_penalty,
        "n_lambda_cap_ts": n_lambda_cap_ts,
        "max_sum_y_cond": max_sum_y,
        "min_margin": min_margin,
        "max_Ts_trial_minus_upper": max_ts_trial_minus_upper,
        "min_Ts": min_Ts,
        "max_Ts": max_Ts,
        "min_Ts_eff": min_Ts_eff,
        "max_Ts_eff": max_Ts_eff,
        "min_Tbub": min_Tbub,
        "max_Tbub": max_Tbub,
        "min_energy_res": min_energy_res,
        "max_energy_res": max_energy_res,
    }


def _iter_run_dirs(runs: Iterable[str], case_dirs: Iterable[str]) -> List[Path]:
    run_dirs: List[Path] = []
    for raw in runs:
        if raw:
            run_dirs.append(Path(raw))
    for raw in case_dirs:
        if not raw:
            continue
        base = Path(raw)
        latest = _latest_run_dir(base)
        if latest is not None:
            run_dirs.append(latest)
    return run_dirs


def main() -> int:
    ap = argparse.ArgumentParser(description="P5 run health summary for interface diagnostics.")
    ap.add_argument("--runs", default="", help="Comma-separated run directories (with interface_diag.csv).")
    ap.add_argument(
        "--case-dirs",
        default="",
        help="Comma-separated case output dirs; latest run dir is used.",
    )
    ap.add_argument("--out", default="out/p5_run_health", help="Output directory.")
    ap.add_argument("--eps-bg", type=float, default=1.0e-5)
    ap.add_argument("--ts-guard", type=float, default=3.0)
    ap.add_argument(
        "--expect-source",
        default="p2db",
        help="Expected source for psat/hvap/latent (use 'any' to skip source check).",
    )
    args = ap.parse_args()

    run_dirs = _iter_run_dirs(
        [p.strip() for p in args.runs.split(",") if p.strip()],
        [p.strip() for p in args.case_dirs.split(",") if p.strip()],
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[List[str]] = []
    summary: Dict[str, object] = {"PASS": True, "runs": {}}

    for run_dir in run_dirs:
        diag_path = run_dir / "interface_diag.csv"
        run_key = str(run_dir)
        if not diag_path.exists():
            summary["runs"][run_key] = {"ok": False, "error": "missing_interface_diag"}
            summary["PASS"] = False
            rows.append([run_key, "False", "missing_interface_diag"])
            continue

        diag = _parse_interface_diag(
            diag_path,
            eps_bg=float(args.eps_bg),
            ts_guard_dT=float(args.ts_guard),
            expect_source=str(args.expect_source),
        )
        ok = True
        ok = ok and (not diag["any_nan"])
        ok = ok and (diag["n_failed_eq"] == 0)
        ok = ok and (diag["n_source_mismatch"] == 0)
        ok = ok and (diag["n_sum_y_exceed"] == 0)
        ok = ok and (diag["n_margin_violation"] == 0)
        ok = ok and (diag["n_finite_false"] == 0)
        ok = ok and (
            (not np.isfinite(diag["max_Ts_trial_minus_upper"]))
            or (diag["max_Ts_trial_minus_upper"] <= 1.0e-12)
        )

        result = dict(diag)
        result["ok"] = ok
        result["run_dir"] = run_key
        summary["runs"][run_key] = result
        summary["PASS"] = summary["PASS"] and ok

        rows.append(
            [
                run_key,
                str(ok),
                f"{int(diag['n_rows'])}",
                str(diag["any_nan"]),
                f"{int(diag['n_failed_eq'])}",
                f"{int(diag['n_source_mismatch'])}",
                f"{int(diag['n_sum_y_exceed'])}",
                f"{int(diag['n_margin_violation'])}",
                f"{int(diag['n_finite_false'])}",
                f"{int(diag['n_penalty_residual'])}",
                f"{int(diag['n_lambda_cap_ts'])}",
                f"{float(diag['max_sum_y_cond']):.6g}",
                f"{float(diag['min_margin']):.6g}",
                f"{float(diag['max_Ts_trial_minus_upper']):.6g}",
                f"{float(diag['min_Ts']):.6g}",
                f"{float(diag['max_Ts']):.6g}",
                f"{float(diag['min_Ts_eff']):.6g}",
                f"{float(diag['max_Ts_eff']):.6g}",
                f"{float(diag['min_Tbub']):.6g}",
                f"{float(diag['max_Tbub']):.6g}",
                f"{float(diag['min_energy_res']):.6g}",
                f"{float(diag['max_energy_res']):.6g}",
            ]
        )

    csv_path = out_dir / "run_health.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_dir",
                "ok",
                "n_rows",
                "any_nan",
                "n_failed_eq",
                "n_source_mismatch",
                "n_sum_y_exceed",
                "n_margin_violation",
                "n_finite_false",
                "n_penalty_residual",
                "n_lambda_cap_ts",
                "max_sum_y_cond",
                "min_margin",
                "max_Ts_trial_minus_upper",
                "min_Ts",
                "max_Ts",
                "min_Ts_eff",
                "max_Ts_eff",
                "min_Tbub",
                "max_Tbub",
                "min_energy_res",
                "max_energy_res",
            ]
        )
        w.writerows(rows)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {summary_path}")
    return 0 if summary["PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
