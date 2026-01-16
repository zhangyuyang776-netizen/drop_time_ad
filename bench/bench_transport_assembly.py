from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _petsc_or_die():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

    bootstrap_mpi_before_petsc()
    from petsc4py import PETSc
    return PETSc


def _build_case_with_Ng(Ng: int):
    try:
        from driver.run_scipy_case import _load_case_config  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _load_case_config  # type: ignore  # noqa: E402

    repo = Path(__file__).resolve().parents[1]
    yml = repo / "cases" / "step4_2_evap_withYg.yaml"
    if not yml.exists():
        raise FileNotFoundError(f"Case yaml not found: {yml}")

    cfg = _load_case_config(str(yml))
    cfg.geometry.N_liq = 1
    cfg.geometry.N_gas = Ng
    cfg.geometry.mesh.enforce_interface_continuity = False
    cfg.physics.include_Ts = False
    cfg.physics.include_mpp = False
    cfg.physics.include_Rd = False
    cfg.physics.solve_Yl = False

    solver_cfg = getattr(cfg, "solver", None)
    linear_cfg = getattr(solver_cfg, "linear", None)
    if linear_cfg is not None:
        linear_cfg.backend = "petsc"
        linear_cfg.assembly_mode = "native_aij"

    cfg.paths.output_root = ROOT
    cfg.paths.case_dir = ROOT / "bench"
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

    gas_model, liq_model = get_or_build_models(cfg)
    try:
        from driver.run_scipy_case import _maybe_fill_gas_species  # noqa: E402
    except ModuleNotFoundError:
        from run_scipy_case import _maybe_fill_gas_species  # type: ignore  # noqa: E402
    _maybe_fill_gas_species(cfg, gas_model)

    grid = build_grid(cfg)
    layout = build_layout(cfg, grid)
    state0 = build_initial_state_erfc(cfg, grid, gas_model, liq_model)
    try:
        props0, _ = compute_props(cfg, grid, state0, gas_model, liq_model)
    except TypeError:
        props0, _ = compute_props(cfg, grid, state0)

    return cfg, grid, layout, state0, props0


def bench_one(build_fn, *, reps: int, **kwargs):
    build_fn(**kwargs)
    t0 = time.perf_counter()
    A = None
    for _ in range(reps):
        A, _b = build_fn(**kwargs)
    t = (time.perf_counter() - t0) / reps
    info = A.getInfo() if A is not None else {}
    n = A.getSize()[0] if A is not None else 0
    nz = float(info.get("nz_used", 0.0))
    return t, nz, nz / max(n, 1)


def main():
    PETSc = _petsc_or_die()
    from assembly.build_system_petsc import (  # noqa: E402
        build_transport_system_petsc_bridge as build_bridge,
        build_transport_system_petsc_native as build_native,
    )

    Ng_list = [20, 200, 500]
    reps = 5

    print("Ng, N_dof, t_bridge, t_native, speedup(native/bridge), avg_nz_native")
    for Ng in Ng_list:
        cfg, grid, layout, state0, props0 = _build_case_with_Ng(Ng)
        dt = float(cfg.time.dt)
        comm = PETSc.COMM_SELF

        tracemalloc.start()

        t_b, _nz_b, _avg_b = bench_one(
            build_bridge,
            reps=reps,
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state0,
            props=props0,
            dt=dt,
            state_guess=state0,
            eq_result=None,
            return_diag=False,
            comm=comm,
        )
        t_n, nz_n, avg_n = bench_one(
            build_native,
            reps=reps,
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_old=state0,
            props=props0,
            dt=dt,
            state_guess=state0,
            eq_result=None,
            return_diag=False,
            comm=comm,
        )

        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        N = layout.n_dof()
        speedup = t_n / max(t_b, 1.0e-30)
        print(
            f"{Ng}, {N}, {t_b:.6g}, {t_n:.6g}, {speedup:.3f}, {avg_n:.2f} | "
            f"py_mem_peak={peak/1e6:.1f}MB"
        )


if __name__ == "__main__":
    main()
