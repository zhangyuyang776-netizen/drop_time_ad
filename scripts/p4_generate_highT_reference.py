#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from properties.p2_liquid_db import load_liquid_db
from properties.p2_pure_models import cp_l, hvap as model_hvap, k_l, psat as model_psat, rho_l


PROP_EVAL = {
    "psat": lambda T, spec: model_psat(T, spec["psat"], Tc=float(spec["Tc"]), Pc=float(spec["Pc"])),
    "hvap": lambda T, spec: model_hvap(T, spec["hvap"], Tc=float(spec["Tc"])),
    "rho": lambda T, spec: rho_l(T, spec["rho"], Tc=float(spec["Tc"])),
    "cp": lambda T, spec: cp_l(T, spec["cp"]),
    "k": lambda T, spec: k_l(T, spec["k"]),
}


def _slugify(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out)


def _load_baseline_max_T(npz: np.lib.npyio.NpzFile, slug: str) -> float | None:
    key = None
    for k in npz.files:
        if k.startswith(f"{slug}__P") and k.endswith("__T"):
            key = k
            break
    if key is None:
        return None
    return float(np.max(np.asarray(npz[key], dtype=np.float64)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate high-T reference data using correlation models.")
    ap.add_argument("--baseline", required=True, help="Path to baseline_coolprop.npz")
    ap.add_argument("--db", required=True, help="Path to liquid_props_db_fitted.yaml")
    ap.add_argument("--out", default="out/p4_highT_reference/ref_highT.npz")
    ap.add_argument("--props", default="psat,hvap", help="Comma-separated props to generate")
    ap.add_argument("--extend-dT", type=float, default=80.0, help="Target extension above baseline Tmax")
    ap.add_argument("--tc-margin", type=float, default=1.0, help="Max temperature = Tc - margin")
    ap.add_argument("--nT", type=int, default=10, help="Number of samples in high-T range")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    db_path = Path(args.db)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    props = [p.strip() for p in args.props.split(",") if p.strip()]
    for prop in props:
        if prop not in PROP_EVAL:
            raise ValueError(f"Unsupported prop '{prop}'.")

    db = load_liquid_db(db_path)
    npz = np.load(baseline_path, allow_pickle=True)
    species_list = [str(s) for s in np.asarray(npz["meta_species"]).tolist()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, np.ndarray] = {}
    meta: Dict[str, object] = {
        "baseline": str(baseline_path),
        "db": str(db_path),
        "props": props,
        "extend_dT": float(args.extend_dT),
        "tc_margin": float(args.tc_margin),
        "nT": int(args.nT),
        "species": species_list,
    }

    for species in species_list:
        if not db.has_species(species):
            continue
        spec = db.get_params(species)
        slug = _slugify(species)
        Tmax_base = _load_baseline_max_T(npz, slug)
        if Tmax_base is None:
            continue
        Tc = float(spec["Tc"])
        Tmax_target = min(float(Tmax_base + args.extend_dT), float(Tc - args.tc_margin))
        if Tmax_target <= Tmax_base:
            continue
        temps = np.linspace(float(Tmax_base), float(Tmax_target), int(args.nT), dtype=np.float64)
        data[f"ref__{slug}__T"] = temps
        for prop in props:
            vals = PROP_EVAL[prop](temps, spec)
            data[f"ref__{slug}__{prop}"] = np.asarray(vals, dtype=np.float64)

    np.savez(out_path, **data)
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"[OK] wrote: {out_path}")
    print(f"[OK] wrote: {out_path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
