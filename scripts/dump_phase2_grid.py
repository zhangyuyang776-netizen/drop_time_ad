from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.grid import build_grid, rebuild_grid_phase2
from driver.run_evap_case import _load_case_config


def dump_grid(step_prefix: Path, grid, *, iface: int, j_anchor: int | None, plan: dict) -> None:
    r_f = np.asarray(grid.r_f, dtype=float)
    r_c = np.asarray(grid.r_c, dtype=float)
    Nf = r_f.size
    Nl = int(grid.Nl)

    faces_path = step_prefix.with_name(f"{step_prefix.name}_faces.csv")
    with faces_path.open("w", encoding="utf-8") as f:
        f.write("face_id,r_f,segment,is_interface,is_anchor,is_far_fixed\n")
        for i in range(Nf):
            seg = "liq" if i <= iface else "gas"
            is_iface = int(i == iface)
            is_anchor = int(j_anchor is not None and i == j_anchor)
            is_far = int(j_anchor is not None and i >= j_anchor)
            f.write(f"{i},{r_f[i]:.16e},{seg},{is_iface},{is_anchor},{is_far}\n")

    cells_path = step_prefix.with_name(f"{step_prefix.name}_cells.csv")
    with cells_path.open("w", encoding="utf-8") as f:
        f.write("cell_id,rL,rR,r_c,dr,segment\n")
        for c in range(r_c.size):
            rL, rR = r_f[c], r_f[c + 1]
            seg = "liq" if c < Nl else "gas"
            f.write(f"{c},{rL:.16e},{rR:.16e},{r_c[c]:.16e},{(rR - rL):.16e},{seg}\n")

    plan_path = step_prefix.with_name(f"{step_prefix.name}_plan.json")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_yaml", type=str)
    ap.add_argument("--out", type=str, default="out/grid_phase2_debug")
    ap.add_argument("--nsteps", type=int, default=5)
    ap.add_argument("--shrink_frac", type=float, default=1.0e-3)
    args = ap.parse_args()

    cfg = _load_case_config(args.case_yaml)
    grid0 = build_grid(cfg)
    Rd0 = float(cfg.geometry.a0)
    out_dir = Path(args.out) / Path(args.case_yaml).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    plan0 = {"step": 0, "Rd": Rd0, "note": "initial build"}
    dump_grid(out_dir / "step_000", grid0, iface=grid0.Nl, j_anchor=None, plan=plan0)

    grid_ref = grid0
    Rd = Rd0
    for s in range(1, args.nsteps + 1):
        Rd_new = Rd * (1.0 - args.shrink_frac)
        grid_new, remap_plan = rebuild_grid_phase2(cfg, Rd_new, grid_ref=grid_ref)

        plan = {
            "step": s,
            "Rd_old": Rd,
            "Rd_new": Rd_new,
            "Nl": grid_new.Nl,
            "Ng": grid_new.Ng,
        }
        j_anchor = None
        if remap_plan is not None:
            plan.update(
                {
                    "liq_remap": remap_plan.liq_remap,
                    "gas_remap_cells": remap_plan.gas_remap_cells,
                    "gas_copy_cells": remap_plan.gas_copy_cells,
                    "k": remap_plan.k,
                    "j_anchor": remap_plan.j_anchor,
                    "dr_if_target": remap_plan.dr_if_target,
                    "q_band": remap_plan.q_band,
                    "q_liq": remap_plan.q_liq,
                }
            )
            j_anchor = remap_plan.j_anchor

        dump_grid(out_dir / f"step_{s:03d}", grid_new, iface=grid_new.Nl, j_anchor=j_anchor, plan=plan)
        grid_ref = grid_new
        Rd = Rd_new

    print(f"[OK] dumped to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
