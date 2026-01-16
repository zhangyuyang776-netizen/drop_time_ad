#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_mpi4py():
    try:
        from mpi4py import MPI
    except Exception as exc:
        print(f"[ERROR] mpi4py is required: {exc}", file=sys.stderr)
        raise SystemExit(1)
    return MPI


def _import_petsc():
    try:
        from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

        bootstrap_mpi_before_petsc()
        from petsc4py import PETSc
    except Exception as exc:
        print(f"[ERROR] petsc4py is required: {exc}", file=sys.stderr)
        raise SystemExit(1)
    return PETSc


def _import_chemistry():
    try:
        import cantera  # noqa: F401
        import CoolProp  # noqa: F401
    except Exception as exc:
        print(f"[ERROR] chemistry backends are required: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _normalize_prefix(prefix: str) -> str:
    p = str(prefix or "")
    if p and not p.endswith("_"):
        p += "_"
    return p


def _set_max_outer_iter(cfg, max_outer_iter: int) -> None:
    if hasattr(cfg, "solver") and hasattr(cfg.solver, "nonlinear"):
        cfg.solver.nonlinear.max_outer_iter = int(max_outer_iter)
    if hasattr(cfg, "nonlinear"):
        cfg.nonlinear.max_outer_iter = int(max_outer_iter)


def _configure_case(
    cfg,
    fs_type: str,
    *,
    options_prefix: str,
    max_outer_iter: int,
    ksp_max_it: int,
    schur_fact_type: str,
    schur_sub_pc_type: str,
) -> None:
    cfg.nonlinear.enabled = True
    cfg.nonlinear.backend = "petsc_mpi"
    cfg.nonlinear.verbose = False
    _set_max_outer_iter(cfg, max_outer_iter)

    cfg.petsc.options_prefix = str(options_prefix)
    cfg.petsc.jacobian_mode = "mf"
    cfg.petsc.ksp_type = "gmres"
    cfg.petsc.max_it = int(ksp_max_it)

    if hasattr(cfg, "solver") and hasattr(cfg.solver, "linear"):
        cfg.solver.linear.pc_type = "fieldsplit"
        fs_cfg: Dict[str, Any] = {
            "type": fs_type,
            "scheme": "bulk_iface",
        }
        if fs_type == "schur":
            fs_cfg["schur_fact_type"] = str(schur_fact_type or "lower")
            sub_pc = str(schur_sub_pc_type or "")
            if sub_pc:
                fs_cfg["subsolvers"] = {
                    "bulk": {"asm_sub_pc_type": sub_pc},
                    "iface": {"asm_sub_pc_type": sub_pc},
                }
        cfg.solver.linear.fieldsplit = fs_cfg

    cfg.io.write_every = 10**9
    cfg.io.scalars_write_every = 10**9
    cfg.io.formats = []
    cfg.io.fields.scalars = []
    cfg.io.fields.gas = []
    cfg.io.fields.liquid = []
    cfg.io.fields.interface = []


def _build_case(
    tmp_path: Path,
    nproc: int,
    rank: int,
    *,
    yml: Path,
    Nl: int,
    Ng: int,
    fs_type: str,
    options_prefix: str,
    max_outer_iter: int,
    ksp_max_it: int,
    schur_fact_type: str,
    schur_sub_pc_type: str,
) -> Tuple[Any, Any, Any, np.ndarray]:
    try:
        from driver.run_scipy_case import _load_case_config, _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config, _maybe_fill_gas_species  # type: ignore  # noqa: E402

    if not yml.exists():
        raise FileNotFoundError(f"Case yaml not found: {yml}")

    cfg = _load_case_config(str(yml))
    cfg.geometry.N_liq = int(Nl)
    cfg.geometry.N_gas = int(Ng)
    cfg.geometry.mesh.enforce_interface_continuity = False

    _configure_case(
        cfg,
        fs_type,
        options_prefix=options_prefix,
        max_outer_iter=max_outer_iter,
        ksp_max_it=ksp_max_it,
        schur_fact_type=schur_fact_type,
        schur_sub_pc_type=schur_sub_pc_type,
    )

    cfg.paths.output_root = tmp_path
    cfg.paths.case_dir = tmp_path / f"case_rank_{rank:03d}"
    cfg.paths.case_dir.mkdir(parents=True, exist_ok=True)

    from core.grid import build_grid  # noqa: E402
    from core.layout import build_layout  # noqa: E402
    from physics.initial import build_initial_state_erfc  # noqa: E402
    from properties.compute_props import get_or_build_models, compute_props  # noqa: E402
    from solvers.nonlinear_context import build_nonlinear_context_for_step  # noqa: E402

    gas_model, liq_model = get_or_build_models(cfg)
    _maybe_fill_gas_species(cfg, gas_model)

    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)
    state0 = build_initial_state_erfc(cfg, grid, gas_model, liq_model)
    try:
        props0, _ = compute_props(cfg, grid, state0, gas_model, liq_model)
    except TypeError:
        props0, _ = compute_props(cfg, grid, state0)

    ctx, u0 = build_nonlinear_context_for_step(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state0,
        props_old=props0,
        t_old=float(cfg.time.t0),
        dt=float(cfg.time.dt),
    )
    return cfg, layout, ctx, u0


def _run_once(
    petsc_comm,
    mpi_comm,
    *,
    scenario: str,
    yml: Path,
    fs_type: str,
    Nl: int,
    Ng: int,
    max_outer_iter: int,
    ksp_max_it: int,
    out_root: Path,
    options_prefix: str,
    schur_precondition: str,
    schur_fact_type: str,
    schur_sub_pc_type: str,
    run_tag: str,
) -> Dict[str, Any]:
    rank = int(mpi_comm.Get_rank())
    size = int(mpi_comm.Get_size())

    run_root = out_root / scenario / fs_type / f"n{size}" / run_tag / f"rank_{rank:03d}"
    run_root.mkdir(parents=True, exist_ok=True)

    cfg, layout, ctx, u0 = _build_case(
        run_root,
        size,
        rank,
        yml=yml,
        Nl=Nl,
        Ng=Ng,
        fs_type=fs_type,
        options_prefix=options_prefix,
        max_outer_iter=max_outer_iter,
        ksp_max_it=ksp_max_it,
        schur_fact_type=schur_fact_type,
        schur_sub_pc_type=schur_sub_pc_type,
    )
    if fs_type == "schur" and schur_precondition:
        try:
            from petsc4py import PETSc

            prefix = _normalize_prefix(cfg.petsc.options_prefix)
            opts = PETSc.Options(prefix or "")
            try:
                opts.setValue("pc_fieldsplit_schur_precondition", schur_precondition)
            except Exception:
                opts["pc_fieldsplit_schur_precondition"] = schur_precondition
        except Exception:
            pass

    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_snes_parallel import solve_nonlinear_petsc_parallel  # noqa: E402

    mgr = build_dm(cfg, layout, comm=petsc_comm)
    ld = LayoutDistributed.build(petsc_comm, mgr, layout)
    ctx.meta["dm"] = mgr.dm
    ctx.meta["dm_manager"] = mgr
    ctx.meta["layout_dist"] = ld

    t0 = time.perf_counter()
    result = solve_nonlinear_petsc_parallel(ctx, u0)
    wall = float(time.perf_counter() - t0)

    diag = result.diag
    extra = diag.extra or {}

    metrics: Dict[str, Any] = {
        "scenario": scenario,
        "fs_type": fs_type,
        "nproc": size,
        "Nl": int(Nl),
        "Ng": int(Ng),
        "snes_iter": int(diag.n_iter),
        "ksp_its_total": int(extra.get("ksp_its_total", -1)),
        "wall_time": wall,
        "converged": bool(diag.converged),
    }
    return metrics


def _allreduce_max(mpi_comm, value):
    MPI = _import_mpi4py()
    return mpi_comm.allreduce(value, op=MPI.MAX)


def _allreduce_min(mpi_comm, value):
    MPI = _import_mpi4py()
    return mpi_comm.allreduce(value, op=MPI.MIN)


def _median(values, *, default: float = float("nan")) -> float:
    if not values:
        return default
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _write_csv_header_if_needed(path: Path, fieldnames: Tuple[str, ...]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _ensure_unique_out_root(base: Path) -> Path:
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


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Benchmark fieldsplit additive vs schur.")
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("DROPLET_BENCH_REPEAT", "7")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("DROPLET_BENCH_WARMUP", "1")))
    parser.add_argument("--max-outer", type=int, default=int(os.environ.get("DROPLET_BENCH_MAX_OUTER", "10")))
    parser.add_argument("--ksp-max-it", type=int, default=int(os.environ.get("DROPLET_BENCH_KSP_MAX_IT", "50")))
    parser.add_argument("--out", default=os.environ.get("DROPLET_BENCH_OUT", str(ROOT / "out" / "bench_fieldsplit_additive_schur")))
    parser.add_argument("--medium-nl", type=int, default=int(os.environ.get("DROPLET_BENCH_MEDIUM_NL", "16")))
    parser.add_argument("--medium-ng", type=int, default=int(os.environ.get("DROPLET_BENCH_MEDIUM_NG", "80")))
    parser.add_argument(
        "--schur-precondition",
        default=os.environ.get("DROPLET_BENCH_SCHUR_PRECONDITION", "selfp"),
    )
    parser.add_argument(
        "--schur-fact-type",
        default=os.environ.get("DROPLET_BENCH_SCHUR_FACT_TYPE", "lower"),
    )
    parser.add_argument(
        "--schur-sub-pc-type",
        default=os.environ.get("DROPLET_BENCH_SCHUR_SUB_PC_TYPE", "ilu"),
    )
    args, unknown = parser.parse_known_args(argv)
    return args, unknown


def main() -> int:
    args, petsc_args = _parse_args()
    sys.argv = [sys.argv[0]] + list(petsc_args)

    MPI = _import_mpi4py()
    PETSc = _import_petsc()
    _import_chemistry()

    mpi_comm = MPI.COMM_WORLD
    petsc_comm = PETSc.COMM_WORLD
    rank = int(mpi_comm.Get_rank())
    size = int(mpi_comm.Get_size())

    if int(petsc_comm.getSize()) != size or int(petsc_comm.getRank()) != rank:
        print("[ERROR] PETSc COMM_WORLD mismatch with MPI COMM_WORLD.", file=sys.stderr)
        return 2

    out_root = _ensure_unique_out_root(Path(args.out))
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "bench_fieldsplit_additive_schur.csv"

    max_outer_iter = int(args.max_outer)
    ksp_max_it = int(args.ksp_max_it)
    repeat = int(args.repeat)
    warmup = int(args.warmup)
    schur_precondition = str(args.schur_precondition or "")
    schur_fact_type = str(args.schur_fact_type or "lower")
    schur_sub_pc_type = str(args.schur_sub_pc_type or "")

    scenarios = [
        {
            "name": "single_small",
            "yml": ROOT / "cases" / "step4_2_evap_withYg.yaml",
            "Nl": 4,
            "Ng": 8,
        },
        {
            "name": "single_medium",
            "yml": ROOT / "cases" / "step4_2_evap_withYg.yaml",
            "Nl": int(args.medium_nl),
            "Ng": int(args.medium_ng),
        },
        {
            "name": "multiliq_small",
            "yml": ROOT / "cases" / "step5_multiliq_evap.yaml",
            "Nl": 4,
            "Ng": 12,
        },
    ]

    rows = []
    for scn in scenarios:
        for fs_type in ("additive", "schur"):
            options_prefix = f"bench_{scn['name']}_{fs_type}_"
            run_metrics = []
            for rep_idx in range(warmup + repeat):
                mpi_comm.barrier()
                run_tag = (
                    f"warmup_{rep_idx + 1}"
                    if rep_idx < warmup
                    else f"rep_{rep_idx - warmup + 1:02d}"
                )
                metrics = _run_once(
                    petsc_comm,
                    mpi_comm,
                    scenario=scn["name"],
                    yml=scn["yml"],
                    fs_type=fs_type,
                    Nl=scn["Nl"],
                    Ng=scn["Ng"],
                    max_outer_iter=max_outer_iter,
                    ksp_max_it=ksp_max_it,
                    out_root=out_root,
                    options_prefix=options_prefix,
                    schur_precondition=schur_precondition if fs_type == "schur" else "",
                    schur_fact_type=schur_fact_type,
                    schur_sub_pc_type=schur_sub_pc_type,
                    run_tag=run_tag,
                )
                metrics["snes_iter"] = int(_allreduce_max(mpi_comm, metrics["snes_iter"]))
                metrics["ksp_its_total"] = int(_allreduce_max(mpi_comm, metrics["ksp_its_total"]))
                metrics["wall_time"] = float(_allreduce_max(mpi_comm, metrics["wall_time"]))
                metrics["converged"] = bool(_allreduce_min(mpi_comm, int(metrics["converged"])))
                if rep_idx >= warmup:
                    run_metrics.append(metrics)

            snes_iters = [m["snes_iter"] for m in run_metrics]
            ksp_iters = [m["ksp_its_total"] for m in run_metrics if m["ksp_its_total"] >= 0]
            wall_times = [m["wall_time"] for m in run_metrics]
            converged_all = all(m["converged"] for m in run_metrics)
            converged_count = sum(1 for m in run_metrics if m["converged"])

            summary = {
                "scenario": scn["name"],
                "fs_type": fs_type,
                "nproc": size,
                "Nl": int(scn["Nl"]),
                "Ng": int(scn["Ng"]),
                "repeat": repeat,
                "warmup": warmup,
                "snes_iter_median": _median(snes_iters, default=float("nan")),
                "ksp_its_total_median": _median(ksp_iters, default=float("nan")),
                "wall_time_median": _median(wall_times, default=float("nan")),
                "snes_iter_min": float(min(snes_iters)) if snes_iters else float("nan"),
                "ksp_its_total_min": float(min(ksp_iters)) if ksp_iters else float("nan"),
                "wall_time_min": float(min(wall_times)) if wall_times else float("nan"),
                "converged_all": bool(converged_all),
                "converged_count": int(converged_count),
                "schur_precondition": schur_precondition if fs_type == "schur" else "",
                "schur_fact_type": schur_fact_type if fs_type == "schur" else "",
                "schur_sub_pc_type": schur_sub_pc_type if fs_type == "schur" else "",
            }
            if rank == 0:
                rows.append(summary)
                print(
                    "[INFO] %s %s: snes_med=%.2f ksp_med=%.2f wall_med=%.3fs conv=%s (%d/%d)"
                    % (
                        summary["scenario"],
                        summary["fs_type"],
                        summary["snes_iter_median"],
                        summary["ksp_its_total_median"],
                        summary["wall_time_median"],
                        summary["converged_all"],
                        summary["converged_count"],
                        repeat,
                    )
                )
        mpi_comm.barrier()

    if rank == 0:
        fieldnames = (
            "scenario",
            "fs_type",
            "nproc",
            "Nl",
            "Ng",
            "repeat",
            "warmup",
            "snes_iter_median",
            "ksp_its_total_median",
            "wall_time_median",
            "snes_iter_min",
            "ksp_its_total_min",
            "wall_time_min",
            "converged_all",
            "converged_count",
            "schur_precondition",
            "schur_fact_type",
            "schur_sub_pc_type",
        )
        _write_csv_header_if_needed(csv_path, fieldnames)
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in rows:
                writer.writerow(row)
        print(f"[INFO] Wrote benchmark results: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
