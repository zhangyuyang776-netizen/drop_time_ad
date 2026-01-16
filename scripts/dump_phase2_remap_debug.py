from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.grid import build_grid, rebuild_grid_phase2
from core.interface_postcorrect import post_correct_interface_after_remap
from core.layout import build_layout
from core.preprocess import preprocess_case
from core.remap import remap_state_to_new_grid
from core.types import State
from driver.run_evap_case import _load_case_config
from properties.compute_props import compute_props


def _dump_grid(prefix: Path, grid) -> None:
    r_f = np.asarray(grid.r_f, dtype=np.float64)
    r_c = np.asarray(grid.r_c, dtype=np.float64)
    V_c = np.asarray(grid.V_c, dtype=np.float64)
    A_f = np.asarray(grid.A_f, dtype=np.float64)
    Nl = int(grid.Nl)

    faces_path = prefix.with_name(f"{prefix.name}_faces.csv")
    with faces_path.open("w", encoding="utf-8") as f:
        f.write("face_id,r_f,A_f,is_interface\n")
        for i in range(r_f.size):
            is_iface = int(i == grid.iface_f)
            f.write(f"{i},{r_f[i]:.16e},{A_f[i]:.16e},{is_iface}\n")

    cells_path = prefix.with_name(f"{prefix.name}_cells.csv")
    with cells_path.open("w", encoding="utf-8") as f:
        f.write("cell_id,rL,rR,r_c,V_c,dr,segment\n")
        for c in range(r_c.size):
            rL = r_f[c]
            rR = r_f[c + 1]
            seg = "liq" if c < Nl else "gas"
            f.write(f"{c},{rL:.16e},{rR:.16e},{r_c[c]:.16e},{V_c[c]:.16e},{(rR - rL):.16e},{seg}\n")


def _build_dummy_state(grid, layout, cfg) -> State:
    Ng = int(grid.Ng)
    Nl = int(grid.Nl)
    Ns_g = int(getattr(layout, "Ns_g_full", 0)) or 1
    Ns_l = int(getattr(layout, "Ns_l_full", 0)) or 1

    Tg = np.linspace(900.0, 1000.0, Ng, dtype=np.float64)
    Tl = np.linspace(300.0, 400.0, Nl, dtype=np.float64)

    Yg = np.full((Ns_g, Ng), 1.0 / Ns_g, dtype=np.float64)
    Yl = np.full((Ns_l, Nl), 1.0 / Ns_l, dtype=np.float64)

    return State(Tg=Tg, Yg=Yg, Tl=Tl, Yl=Yl, Ts=300.0, mpp=0.0, Rd=float(cfg.geometry.a0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_yaml", type=str)
    ap.add_argument("--out", type=str, default="out/remap_debug")
    ap.add_argument("--nsteps", type=int, default=1)
    ap.add_argument("--shrink_frac", type=float, default=1.0e-3)
    args = ap.parse_args()

    cfg = _load_case_config(args.case_yaml)
    cfg = preprocess_case(cfg)
    grid_old = build_grid(cfg)
    layout = build_layout(cfg, grid_old)
    state_old = _build_dummy_state(grid_old, layout, cfg)
    print(
        f"[INFO] Ns_g_full={len(cfg.species.gas_species_full)} "
        f"Ns_l_full={len(cfg.species.liq_species)}"
    )

    out_dir = Path(args.out) / Path(args.case_yaml).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    Rd_old = float(cfg.geometry.a0)
    summary_rows = []
    for step in range(1, args.nsteps + 1):
        Rd_new = Rd_old * (1.0 - args.shrink_frac)
        grid_new, remap_plan = rebuild_grid_phase2(cfg, Rd_new, grid_old)

        iface_fill = {"Ts": float(state_old.Tg[0]), "Yg_sat": state_old.Yg[:, 0].copy()}
        state_new, stats = remap_state_to_new_grid(
            state_old,
            grid_old,
            grid_new,
            cfg,
            layout,
            remap_plan=remap_plan,
            iface_fill=iface_fill,
            return_stats=True,
        )
        props_new, _ = compute_props(cfg, grid_new, state_new)
        state_pc, evap_pc, pc_stats = post_correct_interface_after_remap(
            cfg=cfg,
            grid=grid_new,
            layout=layout,
            state=state_new,
            iface_fill=iface_fill,
            evaporation=None,
            props=props_new,
            remap_stats=stats,
            tol=float(getattr(cfg.remap, "post_correct_tol", 1.0e-10)),
            rtol=float(getattr(cfg.remap, "post_correct_rtol", 1.0e-3)),
            tol_skip_abs=float(getattr(cfg.remap, "post_correct_tol_skip_abs", 1.0e-12)),
            eps_rd_rel=float(getattr(cfg.remap, "post_correct_eps_rd_rel", 1.0e-6)),
            skip_if_uncovered_zero=bool(
                getattr(cfg.remap, "post_correct_skip_if_uncovered_zero", True)
            ),
            improve_min=float(getattr(cfg.remap, "post_correct_improve_min", 1.0e-3)),
            max_iter=int(getattr(cfg.remap, "post_correct_max_iter", 12)),
            damping=float(getattr(cfg.remap, "post_correct_damping", 0.7)),
            fd_eps_T=float(getattr(cfg.remap, "post_correct_fd_eps_T", 1.0e-3)),
            fd_eps_m=float(getattr(cfg.remap, "post_correct_fd_eps_m", 1.0e-6)),
            debug=False,
        )
        state_new = state_pc
        props_new, _ = compute_props(cfg, grid_new, state_new)

        prefix = out_dir / f"step_{step:03d}"
        _dump_grid(prefix.with_name(f"{prefix.name}_grid_old"), grid_old)
        _dump_grid(prefix.with_name(f"{prefix.name}_grid_new"), grid_new)

        plan = asdict(remap_plan) if remap_plan is not None else {"note": "no remap plan"}
        plan.update({"Rd_old": Rd_old, "Rd_new": Rd_new})
        prefix.with_name(f"{prefix.name}_remap_plan.json").write_text(
            json.dumps(plan, indent=2), encoding="utf-8"
        )

        np.savez(
            prefix.with_name(f"{prefix.name}_field_old.npz"),
            Tg=state_old.Tg,
            Tl=state_old.Tl,
            Yg=state_old.Yg,
            Yl=state_old.Yl,
            Ts=state_old.Ts,
            mpp=state_old.mpp,
            Rd=state_old.Rd,
        )
        np.savez(
            prefix.with_name(f"{prefix.name}_field_new.npz"),
            Tg=state_new.Tg,
            Tl=state_new.Tl,
            Yg=state_new.Yg,
            Yl=state_new.Yl,
            Ts=state_new.Ts,
            mpp=state_new.mpp,
            Rd=state_new.Rd,
        )

        iface_pre = {
            "Ts": pc_stats.Ts_old,
            "mpp": pc_stats.mpp_old,
            "res0": pc_stats.res0,
            "abs_tol": float(getattr(cfg.remap, "post_correct_tol", 1.0e-10)),
            "improve_min": float(getattr(cfg.remap, "post_correct_improve_min", 1.0e-3)),
            "Rd": float(state_new.Rd),
            "grid_tag": "new",
        }
        prefix.with_name(f"{prefix.name}_iface_pre.json").write_text(
            json.dumps(iface_pre, indent=2), encoding="utf-8"
        )
        iface_post = {
            "Ts": pc_stats.Ts_new,
            "mpp": pc_stats.mpp_new,
            "res_best": pc_stats.res_best,
            "improved_ratio": pc_stats.improved_ratio,
            "iters": pc_stats.iters,
            "status": pc_stats.status,
            "accepted": pc_stats.accepted,
            "reason": pc_stats.reason,
        }
        prefix.with_name(f"{prefix.name}_iface_post.json").write_text(
            json.dumps(iface_post, indent=2), encoding="utf-8"
        )

        uncovered = {
            "uncovered_cells_gas": int(stats.get("uncovered_cells_gas", 0)),
            "uncovered_frac_max_gas": float(stats.get("uncovered_frac_max_gas", 0.0)),
            "uncovered_frac_p95_gas": float(stats.get("uncovered_frac_p95_gas", 0.0)),
        }
        V3_new = stats.get("V3_new_gas", None)
        V3_ov = stats.get("V3_ov_gas", None)
        uncovered_frac = stats.get("uncovered_frac_gas", None)
        if V3_new is not None and V3_ov is not None and uncovered_frac is not None and len(V3_new) > 0:
            uncovered.update(
                {
                    "gas0_uncovered_frac": float(uncovered_frac[0]),
                    "gas0_V3_new": float(V3_new[0]),
                    "gas0_V3_ov": float(V3_ov[0]),
                }
            )
        prefix.with_name(f"{prefix.name}_uncovered.json").write_text(
            json.dumps(uncovered, indent=2), encoding="utf-8"
        )

        if uncovered["uncovered_frac_max_gas"] > 0.5:
            print(
                "[WARN] uncovered_frac_max_gas=%.3f exceeds 0.5; check mesh or step size."
                % uncovered["uncovered_frac_max_gas"]
            )
        if uncovered["uncovered_cells_gas"] > 0:
            if abs(state_new.Tg[0] - state_old.Tg[0]) > 1.0e-12:
                raise AssertionError("Step2 check failed: Tg_new[0] diluted despite Ts fill.")
            if np.max(np.abs(state_new.Yg[:, 0] - state_old.Yg[:, 0])) > 1.0e-12:
                raise AssertionError("Step2 check failed: Yg_new[:,0] diluted despite Yg_sat fill.")

        summary_rows.append(
            {
                "step": step,
                "Rd_old": Rd_old,
                "Rd_new": Rd_new,
                "uncovered_frac_gas0": float(uncovered.get("gas0_uncovered_frac", 0.0)),
                "uncovered_frac_max_gas": uncovered["uncovered_frac_max_gas"],
                "iface_res_ratio": pc_stats.improved_ratio,
                "iters": pc_stats.iters,
                "status": pc_stats.status,
                "accepted": pc_stats.accepted,
            }
        )

        grid_old = grid_new
        state_old = state_new
        Rd_old = Rd_new

    statuses = [str(r.get("status", "")) for r in summary_rows]
    improved_count = int(sum(1 for s in statuses if s == "IMPROVED"))
    no_improve_count = int(sum(1 for s in statuses if s == "NO_IMPROVEMENT"))
    skipped_count = int(sum(1 for s in statuses if s == "SKIPPED"))
    error_count = int(sum(1 for s in statuses if s == "ERROR"))
    res_ratios = np.array([r["iface_res_ratio"] for r in summary_rows], dtype=np.float64)
    res_ratios_f = res_ratios[np.isfinite(res_ratios)]
    hard_pass = True
    summary = {
        "steps": summary_rows,
        "postcorrect_improved_count": improved_count,
        "postcorrect_no_improvement_count": no_improve_count,
        "postcorrect_skipped_count": skipped_count,
        "postcorrect_error_count": error_count,
        "iface_res_ratio_p50": float(np.percentile(res_ratios_f, 50)) if res_ratios_f.size else float("nan"),
        "iface_res_ratio_p95": float(np.percentile(res_ratios_f, 95)) if res_ratios_f.size else float("nan"),
        "iters_p95": float(np.percentile([r["iters"] for r in summary_rows], 95))
        if summary_rows
        else 0.0,
        "hard_pass": hard_pass,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if error_count > 0:
        print(f"[WARN] post-correct ERROR count={error_count}; see summary.json in {out_dir.resolve()}")

    print(f"[OK] dumped to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
