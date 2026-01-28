#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Acceptance check: mfpc_sparse_fd-only PETSc MPI solve (raw settings from compare script).

Usage:
  mpiexec -n 2 python code/scripts/accept_p2_mfpc_only_raw.py
"""

from __future__ import annotations

import json
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
    except Exception as exc:
        print(f"[ERROR] cantera is required: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _norm_inf(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.max(np.abs(x))) if x.size else 0.0


def _build_case(
    tmp_path: Path,
    nproc: int,
    rank: int,
    *,
    Nl: int,
    Ng: int,
    max_outer_iter: int,
    petsc_max_it: int,
) -> Tuple[Any, Any, Any, np.ndarray]:
    try:
        from driver.run_scipy_case import _load_case_config, _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config, _maybe_fill_gas_species  # type: ignore  # noqa: E402

    yml = ROOT / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        raise FileNotFoundError(f"Case yaml not found: {yml}")

    cfg = _load_case_config(str(yml))
    cfg.geometry.N_liq = int(Nl)
    cfg.geometry.N_gas = int(Ng)
    cfg.geometry.mesh.enforce_interface_continuity = False

    cfg.nonlinear.enabled = True
    cfg.nonlinear.backend = "petsc_mpi"
    cfg.nonlinear.verbose = False
    cfg.nonlinear.max_outer_iter = int(max_outer_iter)

    cfg.petsc.jacobian_mode = "mfpc_sparse_fd"
    cfg.petsc.ksp_type = "gmres"
    cfg.petsc.pc_type = "asm"
    cfg.petsc.max_it = int(petsc_max_it)

    cfg.paths.output_root = tmp_path
    cfg.paths.case_dir = tmp_path / f"case_rank_{rank:03d}"
    cfg.paths.case_dir.mkdir(parents=True, exist_ok=True)
    cfg.io.write_every = 10**9
    cfg.io.scalars_write_every = 10**9
    cfg.io.formats = []
    cfg.io.fields.scalars = []
    cfg.io.fields.gas = []
    cfg.io.fields.liquid = []
    cfg.io.fields.interface = []

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
    out_root: Path,
    Nl: int,
    Ng: int,
    max_outer_iter: int,
    petsc_max_it: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    rank = int(mpi_comm.Get_rank())
    size = int(mpi_comm.Get_size())

    run_root = out_root / f"p2_mfpc_only_raw_n{size}" / f"rank_{rank:03d}"
    run_root.mkdir(parents=True, exist_ok=True)

    cfg, layout, ctx, u0 = _build_case(
        run_root,
        size,
        rank,
        Nl=Nl,
        Ng=Ng,
        max_outer_iter=max_outer_iter,
        petsc_max_it=petsc_max_it,
    )

    from parallel.dm_manager import build_dm  # noqa: E402
    from core.layout_dist import LayoutDistributed  # noqa: E402
    from solvers.petsc_snes_parallel import solve_nonlinear_petsc_parallel  # noqa: E402
    from assembly.residual_global import residual_only  # noqa: E402

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

    res_norm_inf = float(diag.res_norm_inf)
    res_norm_2 = float(diag.res_norm_2)
    res_check = None
    if rank == 0:
        res_check = _norm_inf(residual_only(result.u, ctx))

    metrics: Dict[str, Any] = {
        "jacobian_mode": str(extra.get("jacobian_mode", "mfpc_sparse_fd")),
        "converged": bool(diag.converged),
        "n_iter": int(diag.n_iter),
        "res_norm_inf": res_norm_inf,
        "res_norm_2": res_norm_2,
        "res_norm_inf_check": res_check,
        "ksp_its_total": int(extra.get("ksp_its_total", -1)),
        "time_wall": wall,
        "snes_type": str(extra.get("snes_type", "")),
        "linesearch_type": str(extra.get("linesearch_type", "")),
        "ksp_type": str(extra.get("ksp_type", "")),
        "pc_type": str(extra.get("pc_type", "")),
        "KSP_A_is_P": bool(extra.get("KSP_A_is_P", False)),
        "KSP_A_type": str(extra.get("KSP_A_type", "")),
        "KSP_P_type": str(extra.get("KSP_P_type", "")),
    }

    return metrics, np.asarray(result.u, dtype=np.float64)


def _allreduce_max(mpi_comm, value):
    MPI = _import_mpi4py()
    return mpi_comm.allreduce(value, op=MPI.MAX)


def main() -> int:
    MPI = _import_mpi4py()
    PETSc = _import_petsc()
    _import_chemistry()

    mpi_comm = MPI.COMM_WORLD
    petsc_comm = PETSc.COMM_WORLD
    rank = int(mpi_comm.Get_rank())
    size = int(mpi_comm.Get_size())

    assert int(petsc_comm.getSize()) == size, "PETSc COMM_WORLD size mismatch"
    assert int(petsc_comm.getRank()) == rank, "PETSc COMM_WORLD rank mismatch"

    Nl = int(os.environ.get("DROPLET_P2_ACC_NL", "16"))
    Ng = int(os.environ.get("DROPLET_P2_ACC_NG", "64"))
    max_outer = int(os.environ.get("DROPLET_P2_ACC_MAX_OUTER", "30"))
    max_it = int(os.environ.get("DROPLET_P2_ACC_MAX_IT", "50"))

    res_tol = float(os.environ.get("DROPLET_P2_ACC_RES_TOL", "1e-7"))
    res_check_tol = float(os.environ.get("DROPLET_P2_ACC_RES_CHECK_TOL", "1e-6"))

    out_root = Path(os.environ.get("DROPLET_P2_ACC_OUT", str(ROOT / "out" / "acceptance")))
    out_root.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(
            "[INFO] P2 mfpc-only raw: n=%d Nl=%d Ng=%d max_outer=%d max_it=%d"
            % (size, Nl, Ng, max_outer, max_it)
        )

    mpi_comm.barrier()

    metrics, u = _run_once(
        petsc_comm,
        mpi_comm,
        out_root=out_root,
        Nl=Nl,
        Ng=Ng,
        max_outer_iter=max_outer,
        petsc_max_it=max_it,
    )

    mpi_comm.barrier()

    metrics["res_norm_inf_max"] = float(_allreduce_max(mpi_comm, metrics["res_norm_inf"]))
    metrics["ksp_its_total_max"] = int(_allreduce_max(mpi_comm, metrics["ksp_its_total"]))
    metrics["time_wall_max"] = float(_allreduce_max(mpi_comm, metrics["time_wall"]))

    fail = False
    if not metrics["converged"]:
        fail = True
    if metrics["res_norm_inf_max"] > res_tol:
        fail = True
    if metrics.get("res_norm_inf_check") is not None and metrics["res_norm_inf_check"] > res_check_tol:
        fail = True
    if str(metrics.get("jacobian_mode", "")) != "mfpc_sparse_fd":
        fail = True

    if rank == 0:
        print("[INFO] mfpc: res_inf=%.3e ksp_its=%d wall=%.2fs" % (
            metrics["res_norm_inf_max"],
            metrics["ksp_its_total_max"],
            metrics["time_wall_max"],
        ))
        out_json = out_root / f"p2_mfpc_only_raw_n{size}.json"
        payload = {
            "nproc": size,
            "Nl": Nl,
            "Ng": Ng,
            "max_outer": max_outer,
            "max_it": max_it,
            "res_tol": res_tol,
            "res_check_tol": res_check_tol,
            "mfpc_sparse_fd": metrics,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Wrote mfpc-only report: {out_json}")

    fail_flag = mpi_comm.bcast(bool(fail), root=0)
    return 1 if fail_flag else 0


if __name__ == "__main__":
    raise SystemExit(main())
