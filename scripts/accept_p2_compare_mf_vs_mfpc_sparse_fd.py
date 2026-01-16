#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Acceptance check: compare mf vs mfpc_sparse_fd for PETSc MPI backend.

Usage (example):
  mpiexec -n 2 python code/scripts/accept_p2_compare_mf_vs_mfpc_sparse_fd.py
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
        import CoolProp  # noqa: F401
    except Exception as exc:
        print(f"[ERROR] chemistry backends are required: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _norm_inf(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.max(np.abs(x))) if x.size else 0.0


def _build_case(
    tmp_path: Path,
    nproc: int,
    rank: int,
    *,
    jacobian_mode: str,
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

    cfg.petsc.jacobian_mode = str(jacobian_mode)
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
    jacobian_mode: str,
    out_root: Path,
    Nl: int,
    Ng: int,
    max_outer_iter: int,
    petsc_max_it: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    rank = int(mpi_comm.Get_rank())
    size = int(mpi_comm.Get_size())

    run_root = out_root / f"p2_accept_n{size}" / jacobian_mode / f"rank_{rank:03d}"
    run_root.mkdir(parents=True, exist_ok=True)

    cfg, layout, ctx, u0 = _build_case(
        run_root,
        size,
        rank,
        jacobian_mode=jacobian_mode,
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
        "jacobian_mode": str(extra.get("jacobian_mode", jacobian_mode)),
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

    du_rtol = float(os.environ.get("DROPLET_P2_ACC_DU_RTOL", "1e-7"))
    res_tol = float(os.environ.get("DROPLET_P2_ACC_RES_TOL", "1e-7"))
    res_check_tol = float(os.environ.get("DROPLET_P2_ACC_RES_CHECK_TOL", "1e-6"))

    ksp_ratio = float(os.environ.get("DROPLET_P2_ACC_KSP_RATIO", "0.8"))
    ksp_slack = int(os.environ.get("DROPLET_P2_ACC_KSP_SLACK", "2"))

    out_root = Path(os.environ.get("DROPLET_P2_ACC_OUT", str(ROOT / "out" / "acceptance")))
    out_root.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"[INFO] P2 acceptance: n={size} Nl={Nl} Ng={Ng} max_outer={max_outer} max_it={max_it}")

    mpi_comm.barrier()

    metrics_mf, u_mf = _run_once(
        petsc_comm,
        mpi_comm,
        jacobian_mode="mf",
        out_root=out_root,
        Nl=Nl,
        Ng=Ng,
        max_outer_iter=max_outer,
        petsc_max_it=max_it,
    )

    mpi_comm.barrier()

    metrics_mfpc, u_mfpc = _run_once(
        petsc_comm,
        mpi_comm,
        jacobian_mode="mfpc_sparse_fd",
        out_root=out_root,
        Nl=Nl,
        Ng=Ng,
        max_outer_iter=max_outer,
        petsc_max_it=max_it,
    )

    mpi_comm.barrier()

    # Reduce key metrics across ranks for robust reporting.
    metrics_mf["res_norm_inf_max"] = float(_allreduce_max(mpi_comm, metrics_mf["res_norm_inf"]))
    metrics_mfpc["res_norm_inf_max"] = float(_allreduce_max(mpi_comm, metrics_mfpc["res_norm_inf"]))
    metrics_mf["ksp_its_total_max"] = int(_allreduce_max(mpi_comm, metrics_mf["ksp_its_total"]))
    metrics_mfpc["ksp_its_total_max"] = int(_allreduce_max(mpi_comm, metrics_mfpc["ksp_its_total"]))
    metrics_mf["time_wall_max"] = float(_allreduce_max(mpi_comm, metrics_mf["time_wall"]))
    metrics_mfpc["time_wall_max"] = float(_allreduce_max(mpi_comm, metrics_mfpc["time_wall"]))

    # Correctness checks.
    fail = False
    if not (metrics_mf["converged"] and metrics_mfpc["converged"]):
        fail = True
    if metrics_mf["res_norm_inf_max"] > res_tol or metrics_mfpc["res_norm_inf_max"] > res_tol:
        fail = True
    if metrics_mf.get("res_norm_inf_check") is not None and metrics_mf["res_norm_inf_check"] > res_check_tol:
        fail = True
    if metrics_mfpc.get("res_norm_inf_check") is not None and metrics_mfpc["res_norm_inf_check"] > res_check_tol:
        fail = True
    if str(metrics_mf.get("jacobian_mode", "")) != "mf":
        fail = True
    if str(metrics_mfpc.get("jacobian_mode", "")) != "mfpc_sparse_fd":
        fail = True
    if not metrics_mfpc.get("KSP_A_is_P", False):
        fail = True

    du_inf = _allreduce_max(mpi_comm, _norm_inf(u_mfpc - u_mf))
    u_ref = _allreduce_max(mpi_comm, _norm_inf(u_mf))
    u_ref = max(1.0, float(u_ref))
    du_ratio = float(du_inf / u_ref)
    if du_ratio > du_rtol:
        fail = True

    # Performance advisory (rank0 only).
    if rank == 0:
        its_mf = metrics_mf["ksp_its_total_max"]
        its_pc = metrics_mfpc["ksp_its_total_max"]
        if its_mf < 0 or its_pc < 0:
            print("[WARN] ksp_its_total missing; skip performance advisory")
        elif its_mf >= 30 and its_pc > int(ksp_ratio * its_mf):
            print(
                f"[WARN] KSP its not improved enough: mf={its_mf} mfpc={its_pc} "
                f"(ratio={its_pc/its_mf:.3f}, target<={ksp_ratio})"
            )
        elif its_mf < 30 and its_pc > its_mf + ksp_slack:
            print(
                f"[WARN] KSP its worse than expected: mf={its_mf} mfpc={its_pc} "
                f"(slack={ksp_slack})"
            )

        print("[INFO] mf  : res_inf=%.3e ksp_its=%d wall=%.2fs" % (
            metrics_mf["res_norm_inf_max"],
            metrics_mf["ksp_its_total_max"],
            metrics_mf["time_wall_max"],
        ))
        print("[INFO] mfpc: res_inf=%.3e ksp_its=%d wall=%.2fs" % (
            metrics_mfpc["res_norm_inf_max"],
            metrics_mfpc["ksp_its_total_max"],
            metrics_mfpc["time_wall_max"],
        ))
        print("[INFO] du_inf/||u_mf||_inf = %.3e" % du_ratio)

        out_json = out_root / f"p2_mf_vs_mfpc_sparse_fd_n{size}.json"
        payload = {
            "nproc": size,
            "Nl": Nl,
            "Ng": Ng,
            "max_outer": max_outer,
            "max_it": max_it,
            "du_rtol": du_rtol,
            "res_tol": res_tol,
            "res_check_tol": res_check_tol,
            "mf": metrics_mf,
            "mfpc_sparse_fd": metrics_mfpc,
            "du_inf": du_inf,
            "du_ratio": du_ratio,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Wrote acceptance report: {out_json}")

    # Broadcast failure to all ranks.
    fail_flag = mpi_comm.bcast(bool(fail), root=0)
    return 1 if fail_flag else 0


if __name__ == "__main__":
    raise SystemExit(main())
