#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driver.run_evap_case import run_case


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _load_case_meta(cfg_path: Path) -> Dict[str, object]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML: {cfg_path}")
    case_id = raw.get("case", {}).get("id", cfg_path.stem)
    paths = raw.get("paths", {}) or {}
    output_root = paths.get("output_root", cfg_path.parent / ".." / "out")
    return {
        "case_id": case_id,
        "output_root": _resolve_path(cfg_path.parent, output_root),
    }


def _prepare_case_yaml(
    src_path: Path,
    out_dir: Path,
    *,
    max_steps: int,
    db_path: Optional[str],
    backend: Optional[str],
) -> Path:
    raw = yaml.safe_load(src_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML: {src_path}")

    time_cfg = raw.setdefault("time", {})
    t0 = float(time_cfg.get("t0", 0.0))
    dt = float(time_cfg.get("dt", 1.0e-6))
    time_cfg["t_end"] = t0 + dt * float(max_steps)
    time_cfg["max_steps"] = None

    nonlinear = raw.setdefault("nonlinear", {})
    nonlinear["enabled"] = False
    if backend:
        nonlinear["backend"] = str(backend)

    if backend:
        solver = raw.setdefault("solver", {})
        linear = solver.setdefault("linear", {})
        linear["backend"] = str(backend)

    if db_path:
        physics = raw.setdefault("physics", {})
        liquid = physics.setdefault("liquid", {})
        liquid["db_file"] = str(db_path)

    paths = raw.setdefault("paths", {})
    out_abs = str(out_dir.resolve())
    paths["output_root"] = out_abs
    paths["case_dir"] = out_abs
    if "mechanism_dir" in paths:
        mech_dir = _resolve_path(src_path.parent, paths["mechanism_dir"])
        paths["mechanism_dir"] = str(mech_dir)

    tmp_dir = out_dir / "_tmp_cases"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / src_path.name
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    return tmp_path


def _latest_run_dir(output_root: Path, case_id: str) -> Optional[Path]:
    base = output_root / case_id
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_interface_diag(path: Path) -> Dict[str, object]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    any_nan = False
    n_failed_eq = 0
    n_source_mismatch = 0
    n_sum_y_exceed = 0
    max_sum_y = 0.0

    for row in rows:
        eq_source = str(row.get("eq_source", ""))
        if eq_source and eq_source != "computed":
            n_failed_eq += 1

        psat_source = str(row.get("psat_source", ""))
        hvap_source = str(row.get("hvap_source", ""))
        latent_source = str(row.get("latent_source", ""))
        if psat_source or hvap_source or latent_source:
            if not (psat_source == hvap_source == latent_source == "p2db"):
                n_source_mismatch += 1

        sum_y_raw = row.get("sum_y_cond", "")
        try:
            sum_y = float(sum_y_raw) if sum_y_raw != "" else float("nan")
        except Exception:
            sum_y = float("nan")
        if np.isfinite(sum_y):
            max_sum_y = max(max_sum_y, sum_y)
            eps_bg_raw = row.get("eps_bg", "")
            try:
                eps_bg = float(eps_bg_raw) if eps_bg_raw != "" else 1.0e-5
            except Exception:
                eps_bg = 1.0e-5
            if sum_y > 1.0 - eps_bg + 1.0e-12:
                n_sum_y_exceed += 1
        else:
            any_nan = True

        for key in ("Ts_eff", "Tbub"):
            try:
                val = float(row.get(key, "nan"))
            except Exception:
                val = float("nan")
            if not np.isfinite(val):
                any_nan = True

    return {
        "n_rows": len(rows),
        "any_nan": any_nan,
        "n_failed_eq": n_failed_eq,
        "n_source_mismatch": n_source_mismatch,
        "n_sum_y_exceed": n_sum_y_exceed,
        "max_sum_y_cond": max_sum_y,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run P4 regression cases and summarize interface diagnostics.")
    ap.add_argument(
        "--cases",
        default="code/cases/template_single_nc12.yaml,code/cases/template_multiliq_nc7_nc12.yaml",
        help="Comma-separated list of case YAMLs.",
    )
    ap.add_argument("--max-steps", type=int, default=50, help="Maximum timesteps per case.")
    ap.add_argument("--backend", default=None, help="Optional backend override for case config.")
    ap.add_argument("--out", default="out/p4_regression", help="Output directory for summary.json")
    ap.add_argument("--db", default="", help="Optional liquid db_file override")
    args = ap.parse_args()

    case_paths = [Path(p.strip()) for p in args.cases.split(",") if p.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {"cases": {}, "PASS": True}

    for case_path in case_paths:
        if not case_path.exists():
            summary["cases"][str(case_path)] = {"ok": False, "error": "case_not_found"}
            summary["PASS"] = False
            continue

        temp_case = _prepare_case_yaml(
            case_path,
            out_dir,
            max_steps=int(args.max_steps),
            db_path=args.db or None,
            backend=args.backend or None,
        )
        meta = _load_case_meta(temp_case)
        ret = run_case(str(temp_case), backend=None)
        run_dir = _latest_run_dir(Path(meta["output_root"]), str(meta["case_id"]))
        case_result: Dict[str, object] = {
            "return_code": ret,
            "run_dir": str(run_dir) if run_dir else "",
        }

        ok = ret == 0 and run_dir is not None
        diag_path = run_dir / "interface_diag.csv" if run_dir else None
        if diag_path is None or not diag_path.exists():
            case_result["ok"] = False
            case_result["error"] = "missing_interface_diag"
            summary["PASS"] = False
            summary["cases"][str(case_path)] = case_result
            continue

        diag = _parse_interface_diag(diag_path)
        case_result.update(diag)
        ok = ok and (not diag["any_nan"])
        ok = ok and (diag["n_failed_eq"] == 0)
        ok = ok and (diag["n_source_mismatch"] == 0)
        ok = ok and (diag["n_sum_y_exceed"] == 0)
        case_result["ok"] = ok
        summary["PASS"] = summary["PASS"] and ok
        summary["cases"][str(case_path)] = case_result

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[OK] wrote: {summary_path}")
    return 0 if summary["PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
