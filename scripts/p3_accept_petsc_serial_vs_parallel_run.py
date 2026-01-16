from __future__ import annotations

import argparse
import json
import os
import pprint
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _ensure_unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    parent = base.parent
    stem = base.name
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx:02d}"
        if not candidate.exists():
            return candidate
        idx += 1


def _import_mpi():
    from mpi4py import MPI  # type: ignore

    return MPI


def _is_root_rank() -> bool:
    try:
        MPI = _import_mpi()
        return int(MPI.COMM_WORLD.Get_rank()) == 0
    except Exception:
        return True


def _mpi_comm_size_rank() -> Tuple[int, int]:
    try:
        MPI = _import_mpi()
        comm = MPI.COMM_WORLD
        return int(comm.Get_size()), int(comm.Get_rank())
    except Exception:
        return 1, 0


def _allreduce_max(value: float) -> float:
    try:
        MPI = _import_mpi()
        comm = MPI.COMM_WORLD
        return float(comm.allreduce(float(value), op=MPI.MAX))
    except Exception:
        return float(value)


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="P3 acceptance run: PETSc MPI serial (n=1) vs parallel (n=4)."
    )
    parser.add_argument("case", help="Path to case YAML (e.g. cases/p3_accept_single_petsc_mpi_schur.yaml)")
    parser.add_argument(
        "--out",
        default=None,
        help="Override output root (default: paths.output_root from YAML).",
    )
    args, unknown = parser.parse_known_args(argv)
    return args, unknown


def _load_case_id_and_out_root(cfg_path: Path, override_out: str | None) -> tuple[str, Path]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    case_id = str(raw.get("case", {}).get("id", "case"))
    base = cfg_path.parent
    out_raw = override_out if override_out is not None else raw.get("paths", {}).get("output_root", str(ROOT / "out"))
    out_root = _resolve_path(base, out_raw)
    return case_id, out_root


def _read_last_row_csv(path: Path) -> dict[str, float]:
    import csv

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    last = rows[-1]
    out: dict[str, float] = {}
    for k, v in last.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = float("nan")
    return out


def main() -> int:
    args, petsc_args = _parse_args()

    # Prevent PETSc from treating our CLI flags (e.g. --out) as PETSc options.
    sys.argv = [sys.argv[0]] + list(petsc_args)

    cfg_path = Path(args.case).expanduser().resolve()
    if not cfg_path.exists():
        print(f"[ERROR] case YAML not found: {cfg_path}", file=sys.stderr)
        return 2

    size, rank = _mpi_comm_size_rank()
    case_id, out_root = _load_case_id_and_out_root(cfg_path, args.out)

    run_dir_base = out_root / "p3_stage3_serial_vs_parallel" / case_id / f"n{size}"
    run_dir: Path
    mpi_comm = None
    try:
        MPI = _import_mpi()
        mpi_comm = MPI.COMM_WORLD
    except Exception:
        mpi_comm = None

    # Root rank decides the run_dir (unique suffix if needed) and broadcasts it.
    if rank == 0:
        run_dir = _ensure_unique_dir(run_dir_base)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(cfg_path, run_dir / "config.yaml")
        except Exception:
            pass
    else:
        run_dir = run_dir_base

    if mpi_comm is not None:
        run_dir_str = mpi_comm.bcast(str(run_dir) if rank == 0 else None, root=0)
        run_dir = Path(str(run_dir_str))

    # Sync before compute so timing covers the same work on all ranks.
    try:
        if mpi_comm is not None:
            mpi_comm.Barrier()
    except Exception:
        pass

    # Load config and run the time loop (all ranks must participate).
    from driver import run_scipy_case as driver
    from core.grid import rebuild_grid_phase2, rebuild_grid_with_Rd
    from core.grid import build_grid
    from core.layout import build_layout
    from core.interface_postcorrect import post_correct_interface_after_remap
    from core.remap import remap_state_to_new_grid
    from physics.initial import build_initial_state_erfc
    from properties.compute_props import compute_props, get_or_build_models
    from solvers.timestepper import advance_one_step_scipy

    t_total_start = time.perf_counter()
    t_init_start = time.perf_counter()

    cfg = driver._load_case_config(str(cfg_path))

    # Disable timestepper scalar writer (MPI-unsafe + lacks integrated diagnostics);
    # we write a root-only acceptance CSV instead.
    try:
        cfg.io.scalars_write_every = 0
        cfg.io.fields.scalars = []
    except Exception:
        pass

    gas_model, liq_model = get_or_build_models(cfg)
    driver._maybe_fill_gas_species(cfg, gas_model)

    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)
    state = build_initial_state_erfc(cfg, grid, gas_model, liq_model)
    props, _ = compute_props(cfg, grid, state)

    t_init_end = time.perf_counter()
    init_time_local = t_init_end - t_init_start

    dump_diag = os.environ.get("DROPLET_P3_ACCEPT_DUMP_DIAG", "").strip().lower() not in ("", "0", "false", "no")

    # Root-only scalars writer with cumulative mass closure.
    scalars_path = run_dir / "scalars" / "scalars.csv"
    m_evap_cum = 0.0
    m_dot_prev: float | None = None
    rho_l0 = float(np.mean(props.rho_l)) if hasattr(props, "rho_l") else float("nan")
    m_drop0 = (4.0 / 3.0) * float(np.pi) * rho_l0 * float(state.Rd) ** 3 if np.isfinite(rho_l0) else float("nan")

    scalar_fields = [
        "step",
        "t",
        "Ts",
        "Rd",
        "mpp",
        "mass_balance_rd",
        "m_drop0",
        "m_drop",
        "m_evap_cum",
        "mass_closure_err",
    ]

    io_time_local = 0.0
    if _is_root_rank():
        io_start = time.perf_counter()
        scalars_path.parent.mkdir(parents=True, exist_ok=True)
        with scalars_path.open("w", encoding="utf-8", newline="") as f:
            import csv

            csv.writer(f).writerow(scalar_fields)
        io_time_local += time.perf_counter() - io_start

    # MPI distribution sanity log (printed once per rank).
    layout_logged = False
    mpi_layout_path = run_dir / "mpi_layout.txt"

    t = float(cfg.time.t0)
    step_id = 0
    max_steps = int(getattr(cfg.time, "max_steps", 100) or 100)

    solve_time_local = 0.0
    while t < float(cfg.time.t_end):
        step_id += 1
        if step_id > max_steps:
            break

        solve_start = time.perf_counter()
        res = advance_one_step_scipy(cfg, grid, layout, state, props, t)
        solve_time_local += time.perf_counter() - solve_start
        if not res.success:
            if _is_root_rank():
                (run_dir / "FAILED.txt").write_text(str(res.message or "step failed"), encoding="utf-8")
            return 2
        if getattr(getattr(cfg, "nonlinear", None), "enabled", False) and not bool(res.diag.nonlinear_converged):
            if _is_root_rank():
                (run_dir / "FAILED.txt").write_text("nonlinear not converged", encoding="utf-8")
            return 2
        if (not getattr(getattr(cfg, "nonlinear", None), "enabled", False)) and not bool(res.diag.linear_converged):
            if _is_root_rank():
                (run_dir / "FAILED.txt").write_text("linear not converged", encoding="utf-8")
            return 2

        state = res.state_new
        props = res.props_new
        t = float(res.diag.t_new)

        # Remesh on Rd changes (same as driver).
        if cfg.physics.include_Rd and bool(getattr(cfg.geometry, "use_moving_grid", True)):
            try:
                remap_plan = None
                if bool(getattr(cfg.geometry.mesh, "phase2_enabled", False)):
                    grid_new, remap_plan = rebuild_grid_phase2(cfg, float(state.Rd), grid_ref=grid)
                else:
                    grid_new = rebuild_grid_with_Rd(cfg, float(state.Rd), grid_ref=grid)
                iface_fill = {"Ts": float(state.Ts), "Yg_sat": state.Yg[:, 0].copy()}
                state, remap_stats = remap_state_to_new_grid(
                    state,
                    grid,
                    grid_new,
                    cfg,
                    layout,
                    remap_plan=remap_plan,
                    iface_fill=iface_fill,
                    return_stats=True,
                )
                props, _ = compute_props(cfg, grid_new, state)
                if bool(getattr(cfg.remap, "post_correct_interface", False)):
                    state, evap_post, pc_stats = post_correct_interface_after_remap(
                        cfg=cfg,
                        grid=grid_new,
                        layout=layout,
                        state=state,
                        iface_fill=iface_fill,
                        evaporation=None,
                        props=props,
                        remap_stats=remap_stats if isinstance(remap_stats, dict) else None,
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
                        debug=bool(os.getenv("DROPLET_REMAP_DEBUG", "0") == "1"),
                    )
                    props, _ = compute_props(cfg, grid_new, state)
                grid = grid_new
            except Exception as exc:
                if _is_root_rank():
                    (run_dir / "FAILED.txt").write_text(f"remesh failed: {exc}", encoding="utf-8")
                return 2

        # ---- MPI distribution verification (structure only, no convergence assumptions) ----
        if not layout_logged:
            nl_info = {}
            try:
                nl_info = (res.diag.extra or {}).get("nonlinear", {}) or {}
            except Exception:
                nl_info = {}
            nl_extra = nl_info.get("extra", {}) or {}

            comm_size = nl_extra.get("comm_size", None)
            comm_kind = nl_extra.get("comm_kind", None)
            x_local = nl_extra.get("X_local_size", None)
            x_range = nl_extra.get("X_ownership_range", None)
            global_size = int(layout.n_dof())

            # Per-rank single-line print (as requested).
            print(
                f"[P3 ACCEPT MPI] rank={rank} comm_size={comm_size} comm_kind={comm_kind} "
                f"global_size={global_size} X_local_size={x_local} X_ownership_range={x_range}",
                flush=True,
            )

            # Root gathers and writes a summary that shows coverage of the global range.
            if mpi_comm is not None:
                try:
                    all_info = mpi_comm.allgather(
                        {
                            "rank": rank,
                            "comm_size": comm_size,
                            "comm_kind": comm_kind,
                            "global_size": global_size,
                            "X_local_size": x_local,
                            "X_ownership_range": x_range,
                        }
                    )
                except Exception:
                    all_info = None
            else:
                all_info = [
                    {
                        "rank": rank,
                        "comm_size": comm_size,
                        "comm_kind": comm_kind,
                        "global_size": global_size,
                        "X_local_size": x_local,
                        "X_ownership_range": x_range,
                    }
                ]

            if _is_root_rank():
                try:
                    io_start = time.perf_counter()
                    lines = []
                    lines.append("[P3 ACCEPT MPI] layout distribution summary")
                    for info in sorted(all_info or [], key=lambda d: int(d.get("rank", 0))):
                        lines.append(
                            f"  rank={info.get('rank')} comm_size={info.get('comm_size')} comm_kind={info.get('comm_kind')} "
                            f"global_size={info.get('global_size')} X_local_size={info.get('X_local_size')} "
                            f"X_ownership_range={info.get('X_ownership_range')}"
                        )
                    # Attempt to check coverage if all ranks reported ownership ranges.
                    ranges: list[tuple[int, int]] = []
                    for info in (all_info or []):
                        r = info.get("X_ownership_range", None)
                        if isinstance(r, (list, tuple)) and len(r) == 2:
                            try:
                                lo = int(r[0])
                                hi = int(r[1])
                                ranges.append((lo, hi))
                            except Exception:
                                pass
                    if ranges:
                        ranges_sorted = sorted(ranges)
                        lines.append(f"  ownership_ranges_sorted={ranges_sorted}")
                        ok_cover = True
                        if ranges_sorted[0][0] != 0:
                            ok_cover = False
                        for (lo0, hi0), (lo1, _hi1) in zip(ranges_sorted, ranges_sorted[1:]):
                            if hi0 != lo1:
                                ok_cover = False
                        if ranges_sorted[-1][1] != global_size:
                            ok_cover = False
                        lines.append(f"  ownership_range_covers_global={ok_cover}")

                    mpi_layout_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    io_time_local += time.perf_counter() - io_start
                except Exception:
                    pass

            # Optional dump for deeper debugging (can be large).
            if dump_diag:
                try:
                    (run_dir / f"nonlinear_extra_step1_rank{rank}.txt").write_text(
                        pprint.pformat(nl_extra, width=120),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                try:
                    (run_dir / f"diag_extra_step1_rank{rank}.txt").write_text(
                        pprint.pformat(res.diag.extra, width=120),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

            layout_logged = True

        # Root-only cumulative scalars.
        if _is_root_rank():
            Rd = float(state.Rd)
            Ts = float(state.Ts)
            mpp = float(state.mpp)
            mass_balance_rd = res.diag.mass_balance_rd
            try:
                mass_balance_rd = float(mass_balance_rd) if mass_balance_rd is not None else float("nan")
            except Exception:
                mass_balance_rd = float("nan")

            rho_l_mean = float(np.mean(props.rho_l)) if hasattr(props, "rho_l") else float("nan")
            m_drop = (4.0 / 3.0) * float(np.pi) * rho_l_mean * Rd**3 if np.isfinite(rho_l_mean) else float("nan")
            A_if = 4.0 * float(np.pi) * Rd * Rd
            m_dot = A_if * mpp
            dt = float(res.diag.dt)

            if np.isfinite(m_dot) and np.isfinite(dt) and dt > 0.0:
                if m_dot_prev is None:
                    m_evap_cum += m_dot * dt
                else:
                    m_evap_cum += 0.5 * (m_dot_prev + m_dot) * dt
                m_dot_prev = m_dot

            mass_closure_err = m_drop + m_evap_cum - m_drop0

            row = [
                float(step_id),
                float(t),
                Ts,
                Rd,
                mpp,
                mass_balance_rd,
                float(m_drop0),
                float(m_drop),
                float(m_evap_cum),
                float(mass_closure_err),
            ]
            io_start = time.perf_counter()
            with scalars_path.open("a", encoding="utf-8", newline="") as f:
                import csv

                csv.writer(f).writerow(row)
            io_time_local += time.perf_counter() - io_start

    t_total_end = time.perf_counter()
    total_time_local = t_total_end - t_total_start

    init_time_max = _allreduce_max(init_time_local)
    solve_time_max = _allreduce_max(solve_time_local)
    io_time_max = _allreduce_max(io_time_local)
    other_time_local = max(
        0.0,
        total_time_local - init_time_local - solve_time_local - io_time_local,
    )
    other_time_max = _allreduce_max(other_time_local)
    total_time_max = _allreduce_max(total_time_local)

    if _is_root_rank():
        last = _read_last_row_csv(scalars_path)
        summary: dict[str, Any] = {
            "case_id": case_id,
            "nproc": size,
            "dt": float(cfg.time.dt),
            "t_end": float(cfg.time.t_end),
            "max_steps": max_steps,
            "steps_ran": int(step_id),
            "final_t": float(t),
            "wall_time_sec_max": float(total_time_max),
            "init_time_sec_max": float(init_time_max),
            "solve_time_sec_max": float(solve_time_max),
            "io_time_sec_max": float(io_time_max),
            "other_time_sec_max": float(other_time_max),
            "final_scalars": last,
            "paths": {
                "run_dir": str(run_dir),
                "scalars_csv": str(scalars_path),
                "config_yaml": str(run_dir / "config.yaml"),
            },
            "cfg": {
                "nonlinear_backend": str(getattr(getattr(cfg, "nonlinear", None), "backend", "")),
                "petsc_jacobian_mode": str(getattr(getattr(cfg, "petsc", None), "jacobian_mode", "")),
                "pc_type": str(getattr(getattr(getattr(cfg, "solver", None), "linear", None), "pc_type", "")),
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "[P3 ACCEPT TIMING] "
            f"init={init_time_max:.3f}s solve={solve_time_max:.3f}s "
            f"io={io_time_max:.3f}s other={other_time_max:.3f}s total={total_time_max:.3f}s",
            flush=True,
        )
        print(f"[INFO] P3 acceptance run complete (n={size}): {run_dir}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
