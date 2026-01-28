"""
Unified driver to run evaporation cases across SciPy / PETSc serial / PETSc MPI.

Responsibilities:
- Phase-A parse to decide PETSc/MPI bootstrap.
- Load CaseConfig from YAML.
- Build grid/layout/initial state/initial properties.
- Advance in time by repeatedly calling advance_one_step_scipy.
- Log per-step summaries; stop early on failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from core.grid import build_grid, rebuild_grid_phase2, rebuild_grid_with_Rd
from core.interface_postcorrect import post_correct_interface_after_remap
from core.remap import remap_state_to_new_grid
from core.layout import build_layout
from core.logging_utils import get_log_level_from_env, is_root_rank, setup_logging
from core.types import (
    CaseChecks,
    CaseConfig,
    CaseConventions,
    CaseLiquid,
    CaseDiscretization,
    CaseEquilibrium,
    CaseGeometry,
    CaseIO,
    CaseIOFields,
    CaseInitial,
    CaseInterface,
    CaseMesh,
    CaseMeta,
    CaseNonlinear,
    CaseOutput,
    CasePaths,
    CasePETSc,
    CaseSolver,
    CasePhysics,
    CaseSpecies,
    CaseTime,
    LinearSolverConfig,
    Grid1D,
    State,
    CaseRemap,
)
from properties.compute_props import compute_props, get_or_build_models
from physics.initial import build_initial_state_erfc
from properties.equilibrium import build_equilibrium_model
from properties.gas import GasPropertiesModel
from solvers.linear_types import LinearSolverConfig as LinearSolverConfigTyped
from solvers.timestepper import StepResult, advance_one_step_scipy

logger = logging.getLogger(__name__)

_WRITERS_MODULE_NAME = "io_writers_cached_driver"
_ROOT_RANK: Optional[int] = None


# -----------------------------------------------------------------------------
# Backend helpers (Phase A)
# -----------------------------------------------------------------------------
_BACKEND_ALIAS = {
    "petsc": "petsc_serial",
    "snes": "petsc_serial",
    "petsc_snes": "petsc_serial",
    "petsc_serial": "petsc_serial",
    "petsc_parallel": "petsc_mpi",
    "snes_parallel": "petsc_mpi",
    "petsc_mpi": "petsc_mpi",
    "scipy": "scipy",
}


def _normalize_backend(backend: Optional[str]) -> Optional[str]:
    """Normalize backend names to scipy/petsc_serial/petsc_mpi when possible."""
    if backend is None:
        return None
    val = str(backend).strip().lower()
    return _BACKEND_ALIAS.get(val, val)


def _load_case_backend_phase_a(cfg_path: str) -> dict:
    """Lightweight YAML load to decide backend/bootstrapping."""
    cfg_file = Path(cfg_path).expanduser().resolve()
    raw = yaml.safe_load(_read_yaml_text(cfg_file)) or {}
    nonlinear_raw = raw.get("nonlinear", {}) or {}
    solver_raw = raw.get("solver", {}) or {}
    linear_raw = solver_raw.get("linear", {}) or {}
    petsc_raw = raw.get("petsc", {}) or {}
    case_raw = raw.get("case", {}) or {}
    return {
        "case_id": case_raw.get("id", cfg_file.stem),
        "nonlinear_enabled": bool(nonlinear_raw.get("enabled", False)),
        "nonlinear_backend": nonlinear_raw.get("backend", "scipy"),
        "linear_backend": linear_raw.get("backend", "scipy"),
        "petsc_prefix": petsc_raw.get("options_prefix", ""),
    }


def _select_bootstrap_backend(
    phase_a: Mapping[str, object],
    backend_override: Optional[str],
    force_nonlinear: bool,
) -> Optional[str]:
    """Select backend for early PETSc/MPI bootstrap."""
    if backend_override is not None:
        return _normalize_backend(backend_override)
    if force_nonlinear or bool(phase_a.get("nonlinear_enabled", False)):
        return _normalize_backend(str(phase_a.get("nonlinear_backend", "scipy")))
    return _normalize_backend(str(phase_a.get("linear_backend", "scipy")))


def _maybe_bootstrap_petsc(backend_norm: Optional[str]) -> None:
    """Call MPI/PETSc bootstrap if backend requires PETSc."""
    if backend_norm in ("petsc_serial", "petsc_mpi"):
        from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

        bootstrap_mpi_before_petsc()


def _format_prefix(prefix: str) -> str:
    value = str(prefix or "").strip()
    if not value:
        return ""
    return value if value.endswith("_") else f"{value}_"


def _get_mpi_rank_size() -> Tuple[int, int]:
    """Best-effort MPI rank/size detection without forcing PETSc imports."""
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank()), int(MPI.COMM_WORLD.Get_size())
    except Exception:
        pass

    def _env_int(name: str) -> Optional[int]:
        raw = os.environ.get(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except Exception:
            return None

    size = (
        _env_int("OMPI_COMM_WORLD_SIZE")
        or _env_int("PMI_SIZE")
        or _env_int("MV2_COMM_WORLD_SIZE")
    )
    rank = (
        _env_int("OMPI_COMM_WORLD_RANK")
        or _env_int("PMI_RANK")
        or _env_int("MV2_COMM_WORLD_RANK")
    )
    if size is None or size < 1:
        size = 1
    if rank is None or rank < 0:
        rank = 0
    return rank, size


# -----------------------------------------------------------------------------
# YAML loader
# -----------------------------------------------------------------------------
def _resolve_path(base: Path, value: str | Path) -> Path:
    """Resolve a possibly relative path against base."""
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _read_yaml_text(cfg_file: Path) -> str:
    try:
        return cfg_file.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return cfg_file.read_text()


def _load_case_config(cfg_path: str) -> CaseConfig:
    """Load YAML file into CaseConfig with nested dataclasses."""
    cfg_file = Path(cfg_path).expanduser().resolve()
    raw = yaml.safe_load(_read_yaml_text(cfg_file))
    base = cfg_file.parent

    case_cfg = CaseMeta(**raw["case"])

    paths_raw = raw["paths"]
    output_root = _resolve_path(base, paths_raw["output_root"])
    case_dir_raw = paths_raw.get("case_dir", output_root / case_cfg.id)
    case_dir = _resolve_path(base, case_dir_raw)
    mech_dir_raw = paths_raw.get("mechanism_dir", base)
    mechanism_dir = _resolve_path(base, mech_dir_raw)
    paths_cfg = CasePaths(
        output_root=output_root,
        case_dir=case_dir,
        mechanism_dir=mechanism_dir,
        gas_mech=paths_raw["gas_mech"],
    )

    conv_cfg = CaseConventions(**raw["conventions"])

    phys_raw = raw["physics"]
    eq_raw = phys_raw.get("interface", {}).get("equilibrium", {})
    if "background_fill" in eq_raw:
        raise ValueError(
            "Config key physics.interface.equilibrium.background_fill has been removed; "
            "interface equilibrium now always uses interface_noncondensables."
        )
    allowed_eq_keys = {
        "method",
        "condensables_gas",
        "Ts_guard_dT",
        "Ts_guard_width_K",
        "Ts_sat_eps_K",
        "eps_bg",
        "sat_tol_enter",
        "sat_tol_exit",
        "regime_lock_max",
    }
    unknown_eq_keys = set(eq_raw.keys()) - allowed_eq_keys
    if unknown_eq_keys:
        raise ValueError(
            "Unsupported equilibrium keys in physics.interface.equilibrium: "
            f"{sorted(unknown_eq_keys)}. Use p2db-only equilibrium config."
        )
    eq_cfg = CaseEquilibrium(
        method=eq_raw.get("method", "raoult_psat"),
        condensables_gas=list(eq_raw.get("condensables_gas", [])),
        Ts_guard_dT=float(eq_raw.get("Ts_guard_dT", 3.0)),
        Ts_guard_width_K=float(eq_raw.get("Ts_guard_width_K", 0.5)),
        Ts_sat_eps_K=float(eq_raw.get("Ts_sat_eps_K", 0.01)),
        eps_bg=float(eq_raw.get("eps_bg", 1.0e-5)),
        sat_tol_enter=float(eq_raw.get("sat_tol_enter", 1.0e-3)),
        sat_tol_exit=float(eq_raw.get("sat_tol_exit", 5.0e-3)),
        regime_lock_max=int(eq_raw.get("regime_lock_max", 1)),
    )
    iface_raw = phys_raw.get("interface", {})
    interface_cfg = CaseInterface(
        type=iface_raw.get("type", "no_condensation"),
        bc_mode=iface_raw.get("bc_mode", "Ts_fixed"),
        Ts_fixed=float(iface_raw.get("Ts_fixed", 300.0)),
        equilibrium=eq_cfg,
    )
    liquid_raw = phys_raw.get("liquid", {}) or {}
    liquid_cfg = CaseLiquid(
        backend=str(liquid_raw.get("backend", "p2db")),
        db_file=str(liquid_raw.get("db_file", "")),
    )
    physics_cfg = CasePhysics(
        model=phys_raw.get("model", "droplet_1d_spherical_noChem"),
        enable_liquid=phys_raw.get("enable_liquid", True),
        include_chemistry=phys_raw.get("include_chemistry", False),
        solve_Tg=phys_raw.get("solve_Tg", True),
        solve_Yg=phys_raw.get("solve_Yg", True),
        solve_Tl=phys_raw.get("solve_Tl", True),
        solve_Yl=phys_raw.get("solve_Yl", False),
        include_Ts=phys_raw.get("include_Ts", False),
        include_mpp=phys_raw.get("include_mpp", True),
        include_Rd=phys_raw.get("include_Rd", True),
        stefan_velocity=phys_raw.get("stefan_velocity", True),
        interface=interface_cfg,
        liquid=liquid_cfg,
    )

    species_raw = raw["species"]
    deprecated_keys = {"gas_species", "gas_solved_species", "solve_gas_mode", "solve_gas_species"}
    found = [k for k in deprecated_keys if k in species_raw]
    if found:
        raise ValueError(
            f"Deprecated species keys not allowed: {found}. "
            "Gas species now come solely from the mechanism; remove these entries from YAML."
        )
    species_cfg = CaseSpecies(
        gas_balance_species=species_raw["gas_balance_species"],
        gas_mechanism_phase=species_raw.get("gas_mechanism_phase", "gas"),
        liq_species=list(species_raw.get("liq_species", [])),
        liq_balance_species=species_raw["liq_balance_species"],
        liq2gas_map=species_raw.get("liq2gas_map", {}),
        mw_kg_per_mol=species_raw.get("mw_kg_per_mol", {}),
        molar_volume_cm3_per_mol=species_raw.get("molar_volume_cm3_per_mol", {}),
    )

    mesh_raw = raw["geometry"]["mesh"]
    mesh_method = mesh_raw.get("method", None)
    liq_method = mesh_raw.get("liq_method", mesh_method)
    gas_method = mesh_raw.get("gas_method", mesh_method)
    if liq_method is None or gas_method is None:
        raise ValueError("mesh.method or both mesh.liq_method/gas_method must be provided.")
    mesh_cfg = CaseMesh(
        liq_method=liq_method,
        liq_beta=mesh_raw.get("liq_beta", None),
        liq_center_bias=mesh_raw.get("liq_center_bias", None),
        gas_method=gas_method,
        gas_beta=mesh_raw.get("gas_beta", None),
        gas_center_bias=mesh_raw.get("gas_center_bias", None),
        enforce_interface_continuity=mesh_raw.get("enforce_interface_continuity", False),
        continuity_tol=mesh_raw.get("continuity_tol", 0.0),
        method=mesh_method,
        control=mesh_raw.get("control", None),
        iface_dr=mesh_raw.get("iface_dr", None),
        adj_max_target=float(mesh_raw.get("adj_max_target", 1.30)),
        global_max_target=mesh_raw.get("global_max_target", None),
        liq_adj_max_target=mesh_raw.get("liq_adj_max_target", None),
        auto_adjust_n_liq=bool(mesh_raw.get("auto_adjust_n_liq", False)),
        phase2_enabled=bool(mesh_raw.get("phase2_enabled", False)),
        phase2_method=str(mesh_raw.get("phase2_method", "band_geometric")),
        phase2_adj_max_target=float(mesh_raw.get("phase2_adj_max_target", 1.30)),
        phase2_gas_band_frac_target=float(mesh_raw.get("phase2_gas_band_frac_target", 0.15)),
        phase2_gas_band_frac_min=float(mesh_raw.get("phase2_gas_band_frac_min", 0.10)),
        phase2_gas_band_frac_max=float(mesh_raw.get("phase2_gas_band_frac_max", 0.20)),
        phase2_dr_if_eps=float(mesh_raw.get("phase2_dr_if_eps", 1.0e-15)),
        exp_power=float(mesh_raw.get("exp_power", 2.0)),
    )
    geom_raw = raw["geometry"]
    geom_cfg = CaseGeometry(
        a0=geom_raw["a0"],
        R_inf=geom_raw["R_inf"],
        N_liq=geom_raw["N_liq"],
        N_gas=geom_raw["N_gas"],
        mesh=mesh_cfg,
    )

    time_raw = raw["time"]
    time_cfg = CaseTime(
        t0=time_raw["t0"],
        dt=time_raw["dt"],
        t_end=time_raw["t_end"],
        max_steps=time_raw.get("max_steps", None),
    )

    disc_raw = raw["discretization"]
    disc_cfg = CaseDiscretization(
        time_scheme=disc_raw["time_scheme"],
        theta=disc_raw["theta"],
        mass_matrix=disc_raw["mass_matrix"],
    )

    init_raw = raw["initial"]
    init_cfg = CaseInitial(
        T_inf=init_raw["T_inf"],
        P_inf=init_raw["P_inf"],
        T_d0=init_raw["T_d0"],
        Yg=init_raw["Yg"],
        Yl=init_raw["Yl"],
        Y_seed=init_raw["Y_seed"],
        # Unified init time scale; legacy t_init_Y/D_init_Y are ignored downstream
        t_init_T=init_raw.get("t_init_T", 1.0e-6),
        Y_vap_if0=init_raw.get("Y_vap_if0", 1.0e-6),
    )

    petsc_raw = raw.get("petsc", {}) or {}
    petsc_cfg = CasePETSc(
        options_prefix=str(petsc_raw.get("options_prefix", "")),
        ksp_type=str(petsc_raw.get("ksp_type", "gmres")),
        pc_type=str(petsc_raw.get("pc_type", "ilu")),
        rtol=float(petsc_raw.get("rtol", 1.0e-8)),
        atol=float(petsc_raw.get("atol", 1.0e-12)),
        max_it=int(petsc_raw.get("max_it", 200)),
        restart=int(petsc_raw.get("restart", 30)),
        monitor=bool(petsc_raw.get("monitor", False)),
        snes_type=str(petsc_raw.get("snes_type", "newtonls")),
        linesearch_type=str(petsc_raw.get("linesearch_type", "bt")),
        jacobian_mode=str(petsc_raw.get("jacobian_mode", "fd")),
        fd_eps=float(petsc_raw.get("fd_eps", 1.0e-8)),
        snes_monitor=bool(petsc_raw.get("snes_monitor", False)),
    )

    solver_raw = raw.get("solver", {}) or {}
    linear_raw = solver_raw.get("linear", {}) or {}
    linear_cfg = LinearSolverConfig(
        backend=str(linear_raw.get("backend", "scipy")),
        method=linear_raw.get("method", None),
        assembly_mode=str(linear_raw.get("assembly_mode", "bridge_dense")),
        pc_type=linear_raw.get("pc_type", None),
        asm_overlap=linear_raw.get("asm_overlap", None),
        fieldsplit=linear_raw.get("fieldsplit", None),
    )
    solver_cfg = CaseSolver(linear=linear_cfg)

    io_raw = raw["io"]
    fields_raw = io_raw.get("fields", {})
    io_fields_cfg = CaseIOFields(
        scalars=list(fields_raw.get("scalars", [])),
        gas=list(fields_raw.get("gas", [])),
        liquid=list(fields_raw.get("liquid", [])),
        interface=list(fields_raw.get("interface", [])),
    )
    io_cfg = CaseIO(
        write_every=io_raw["write_every"],
        formats=list(io_raw.get("formats", [])),
        save_grid=io_raw.get("save_grid", False),
        fields=io_fields_cfg,
        scalars_write_every=int(io_raw.get("scalars_write_every", io_raw.get("write_every", 1))),
    )

    checks_raw = raw["checks"]
    checks_cfg = CaseChecks(
        enforce_sumY=checks_raw["enforce_sumY"],
        sumY_tol=checks_raw["sumY_tol"],
        clamp_negative_Y=checks_raw["clamp_negative_Y"],
        min_Y=checks_raw["min_Y"],
        enforce_T_bounds=checks_raw["enforce_T_bounds"],
        T_min=checks_raw["T_min"],
        T_max=checks_raw["T_max"],
        enforce_unique_index=checks_raw.get("enforce_unique_index", True),
        enforce_grid_state_props_split=checks_raw.get("enforce_grid_state_props_split", True),
        enforce_assembly_purity=checks_raw.get("enforce_assembly_purity", True),
    )

    nonlinear_raw = raw.get("nonlinear", {}) or {}
    nonlinear_cfg = CaseNonlinear(
        enabled=bool(nonlinear_raw.get("enabled", False)),
        backend=str(nonlinear_raw.get("backend", "scipy")),
        solver=str(nonlinear_raw.get("solver", "newton_krylov")),
        krylov_method=str(nonlinear_raw.get("krylov_method", "lgmres")),
        max_outer_iter=int(nonlinear_raw.get("max_outer_iter", 20)),
        inner_maxiter=int(nonlinear_raw.get("inner_maxiter", 20)),
        f_rtol=float(nonlinear_raw.get("f_rtol", 1.0e-6)),
        f_atol=float(nonlinear_raw.get("f_atol", 1.0e-10)),
        use_scaled_unknowns=bool(nonlinear_raw.get("use_scaled_unknowns", True)),
        use_scaled_residual=bool(nonlinear_raw.get("use_scaled_residual", True)),
        residual_scale_floor=float(nonlinear_raw.get("residual_scale_floor", 1.0e-12)),
        verbose=bool(nonlinear_raw.get("verbose", False)),
        log_every=int(nonlinear_raw.get("log_every", 5)),
        smoke=bool(nonlinear_raw.get("smoke", False)),  # P3.5-5-1: Load smoke field
        sanitize_mode=str(nonlinear_raw.get("sanitize_mode", "penalty")),
        penalty_value=float(nonlinear_raw.get("penalty_value", 1.0e20)),
        penalty_scope=str(nonlinear_raw.get("penalty_scope", "interface_only")),
        ts_linesearch_cap=bool(nonlinear_raw.get("ts_linesearch_cap", True)),
        ts_linesearch_alpha=float(nonlinear_raw.get("ts_linesearch_alpha", 0.8)),
        ts_upper_mode=str(nonlinear_raw.get("ts_upper_mode", "tbub_last")),
        ls_max_dTs=float(nonlinear_raw.get("ls_max_dTs", 0.2)),
        ls_max_dmpp_rel=float(nonlinear_raw.get("ls_max_dmpp_rel", 0.05)),
        ls_max_dmpp_ref=float(nonlinear_raw.get("ls_max_dmpp_ref", 1.0e-4)),
        ls_shrink=float(nonlinear_raw.get("ls_shrink", 0.5)),
        enable_vi_bounds=bool(nonlinear_raw.get("enable_vi_bounds", False)),
        Ts_lower=float(nonlinear_raw.get("Ts_lower", 250.0)),
    )

    remap_raw = raw.get("remap", {}) or {}
    remap_cfg = CaseRemap(
        post_correct_interface=bool(remap_raw.get("post_correct_interface", False)),
        post_correct_tol=float(remap_raw.get("post_correct_tol", 1.0e-10)),
        post_correct_rtol=float(remap_raw.get("post_correct_rtol", 1.0e-3)),
        post_correct_tol_skip_abs=float(remap_raw.get("post_correct_tol_skip_abs", 1.0e-8)),
        post_correct_eps_rd_rel=float(remap_raw.get("post_correct_eps_rd_rel", 1.0e-6)),
        post_correct_skip_if_uncovered_zero=bool(
            remap_raw.get("post_correct_skip_if_uncovered_zero", True)
        ),
        post_correct_improve_min=float(remap_raw.get("post_correct_improve_min", 1.0e-3)),
        post_correct_max_iter=int(remap_raw.get("post_correct_max_iter", 12)),
        post_correct_damping=float(remap_raw.get("post_correct_damping", 0.7)),
        post_correct_fd_eps_T=float(remap_raw.get("post_correct_fd_eps_T", 1.0e-3)),
        post_correct_fd_eps_m=float(remap_raw.get("post_correct_fd_eps_m", 1.0e-6)),
    )

    output_raw = raw.get("output", {}) or {}
    output_cfg = CaseOutput(
        u_enabled=bool(output_raw.get("u_enabled", False)),
        u_every=int(output_raw.get("u_every", 1)),
    )

    cfg = CaseConfig(
        case=case_cfg,
        paths=paths_cfg,
        conventions=conv_cfg,
        physics=physics_cfg,
        species=species_cfg,
        geometry=geom_cfg,
        time=time_cfg,
        discretization=disc_cfg,
        initial=init_cfg,
        petsc=petsc_cfg,
        io=io_cfg,
        checks=checks_cfg,
        nonlinear=nonlinear_cfg,
        solver=solver_cfg,
        remap=remap_cfg,
        output=output_cfg,
    )

    LinearSolverConfigTyped.from_cfg(cfg)
    return cfg


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def _maybe_fill_gas_species(cfg: CaseConfig, gas_model: GasPropertiesModel) -> None:
    """Mechanism-driven gas species ordering; always use full mechanism order."""
    mech_names = list(gas_model.gas.species_names)
    mech_map = {nm: i for i, nm in enumerate(mech_names)}

    cfg.species.gas_species_full = mech_names
    cfg.species.gas_name_to_index = mech_map
    cl = cfg.species.gas_balance_species
    if cl and cl not in mech_map:
        raise ValueError(f"gas_balance_species '{cl}' not found in mechanism species.")

    # condensables must exist
    conds = list(cfg.physics.interface.equilibrium.condensables_gas or [])
    missing = [s for s in conds if s not in mech_map]
    if missing:
        raise ValueError(f"condensables_gas not found in mechanism species: {missing}")

    # liquid name map (not mechanism-driven but useful)
    cfg.species.liq_name_to_index = {nm: i for i, nm in enumerate(cfg.species.liq_species)}


def _build_mass_fractions(
    names: Sequence[str],
    values: Mapping[str, float],
    closure_name: str,
    seed: float,
    n_cells: int,
) -> np.ndarray:
    """Build full mass-fraction array with closure species filled as complement."""
    Ns = len(names)
    Y = np.full((Ns, n_cells), float(seed), dtype=np.float64)
    for i, name in enumerate(names):
        if name in values:
            Y[i, :] = float(values[name])

    if closure_name in names:
        idx = names.index(closure_name)
        others = np.sum(Y, axis=0) - Y[idx, :]
        Y[idx, :] = np.maximum(1.0 - others, 0.0)

    sums = np.sum(Y, axis=0)
    for j in range(n_cells):
        s = float(sums[j])
        if s > 0.0:
            Y[:, j] /= s
        elif Ns > 0:
            Y[0, j] = 1.0
    return Y


def _prepare_run_dir(cfg: CaseConfig, cfg_path: str) -> Path:
    """Create per-run output directory and copy cfg yaml into it."""
    out_root = Path(cfg.paths.output_root)
    case_id = getattr(cfg.case, "id", "case")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / case_id / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.paths.case_dir = run_dir

    try:
        shutil.copy2(cfg_path, run_dir / "config.yaml")
    except Exception as exc:  # pragma: no cover - best-effort copy
        logger.warning("Failed to copy cfg to run dir: %s", exc)
    return run_dir


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
def _is_root_rank() -> bool:
    """Return True when running on the root MPI rank (or single-process)."""
    global _ROOT_RANK
    if _ROOT_RANK is not None:
        return _ROOT_RANK == 0
    _ROOT_RANK = 0 if is_root_rank() else 1
    return _ROOT_RANK == 0


def _log_step(res: StepResult, step_id: int) -> None:
    """Emit one-line step summary."""
    if not _is_root_rank():
        return
    d = res.diag
    if getattr(d, "nonlinear_method", ""):
        logger.info(
            "step=%d t=[%.6e -> %.6e] dt=%.3e Ts=%.3f Rd=%.3e mpp=%.3e Tg[min,max]=[%.3f, %.3f] "
            "nl_conv=%s nl_iter=%d nl_res_inf=%.3e",
            step_id,
            d.t_old,
            d.t_new,
            d.dt,
            d.Ts,
            d.Rd,
            d.mpp,
            d.Tg_min,
            d.Tg_max,
            d.nonlinear_converged,
            d.nonlinear_n_iter,
            d.nonlinear_residual_inf,
        )
    else:
        logger.info(
            "step=%d t=[%.6e -> %.6e] dt=%.3e Ts=%.3f Rd=%.3e mpp=%.3e Tg[min,max]=[%.3f, %.3f] lin_conv=%s rel=%.3e",
            step_id,
            d.t_old,
            d.t_new,
            d.dt,
            d.Ts,
            d.Rd,
            d.mpp,
            d.Tg_min,
            d.Tg_max,
            d.linear_converged,
            d.linear_rel_residual,
        )


def _sanity_check_state(state: State) -> Optional[str]:
    """Lightweight driver-level sanity checks."""
    if not np.isfinite(state.Ts):
        return "Non-finite Ts"
    if not np.isfinite(state.Rd) or state.Rd <= 0.0:
        return "Rd is non-positive or non-finite"
    if not np.isfinite(state.mpp):
        return "Non-finite mpp"
    if np.any(~np.isfinite(state.Tg)):
        return "Non-finite Tg entries"
    if np.any(~np.isfinite(state.Tl)):
        return "Non-finite Tl entries"
    return None


# -----------------------------------------------------------------------------
# Optional spatial writer (dynamic import to avoid io module shadowing)
# -----------------------------------------------------------------------------
def _load_writers_module():
    """Dynamically import io.writers to avoid stdlib io shadowing."""
    module = sys.modules.get(_WRITERS_MODULE_NAME)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent.parent / "io" / "writers.py"
    spec = importlib.util.spec_from_file_location(_WRITERS_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load writers module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_WRITERS_MODULE_NAME] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _maybe_write_spatial(cfg: CaseConfig, grid: Grid1D, state: State, step_id: int, t: float | None = None) -> None:
    """Write spatial fields at configured frequency."""
    write_every = int(getattr(cfg.io, "write_every", 0) or 0)
    if write_every <= 0:
        return
    if step_id % write_every != 0:
        return
    try:
        module = _load_writers_module()
        write_fn = getattr(module, "write_step_spatial", None)
        if write_fn is None:
            return
        write_fn(cfg=cfg, grid=grid, state=state, step_id=step_id, t=t)
    except Exception as exc:  # pragma: no cover - best-effort output
        logger.warning("write_step_spatial failed at step %s: %s", step_id, exc)


# Track if mapping.json has been written (module-level)
_MAPPING_WRITTEN = False


def _write_mapping_once(cfg: CaseConfig, grid: Grid1D, layout) -> None:
    """Write mapping.json once at the start of the run."""
    global _MAPPING_WRITTEN
    if _MAPPING_WRITTEN:
        return

    try:
        module = _load_writers_module()
        write_fn = getattr(module, "write_mapping_json", None)
        if write_fn is None:
            return
        write_fn(cfg=cfg, grid=grid, layout=layout, run_dir=None)
        _MAPPING_WRITTEN = True
    except Exception as exc:  # pragma: no cover - best-effort output
        logger.warning("write_mapping_json failed: %s", exc)


def _maybe_write_u(cfg, grid: Grid1D, state: State, layout, step_id: int, t: float) -> None:
    """Write u vector and grid coordinates at configured frequency."""
    try:
        module = _load_writers_module()
        should_write_fn = getattr(module, "should_write_u", None)
        if should_write_fn is None:
            logger.debug("should_write_u function not found in writers module")
            return

        should_write = should_write_fn(cfg, step_id)
        if not should_write:
            logger.debug(f"Skipping u output at step {step_id} (should_write_u returned False)")
            return

        logger.debug(f"Writing u vector at step {step_id}, t={t:.6e}")

        # Pack state into u vector
        from core.layout import pack_state

        u, _, _ = pack_state(state, layout)

        # Write u vector
        write_fn = getattr(module, "write_step_u", None)
        if write_fn is None:
            logger.warning("write_step_u function not found in writers module")
            return
        write_fn(cfg=cfg, step_id=step_id, t=t, u=u, grid=grid, run_dir=None)

        if _is_root_rank():
            logger.info(f"Wrote u vector snapshot at step {step_id}, t={t:.6e}")

    except Exception as exc:  # pragma: no cover - best-effort output
        logger.warning("write_step_u failed at step %s: %s", step_id, exc)
        import traceback
        logger.debug(traceback.format_exc())


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def run_case(
    cfg_path: str,
    *,
    backend: Optional[str] = None,
    prefix: Optional[str] = None,
    max_outer_iter: Optional[int] = None,
    dry_run: bool = False,
    max_steps: Optional[int] = None,
    log_level: int | str = logging.INFO,
) -> int:
    """Run one evaporation case. Return 0 on success, non-zero on early failure."""
    cfg_path = str(cfg_path)
    try:
        rank, size = _get_mpi_rank_size()
        level = get_log_level_from_env(default=log_level)
        setup_logging(rank, level=level, quiet_nonroot=True)

        backend_norm = _normalize_backend(backend)
        if backend is not None and backend_norm not in ("scipy", "petsc_serial", "petsc_mpi"):
            raise ValueError(f"Unknown backend override: {backend!r}")
        force_nonlinear = backend is not None or max_outer_iter is not None
        phase_a = _load_case_backend_phase_a(cfg_path)
        bootstrap_backend = _select_bootstrap_backend(phase_a, backend_norm, force_nonlinear)
        _maybe_bootstrap_petsc(bootstrap_backend)

        cfg = _load_case_config(cfg_path)
        if backend_norm is not None:
            cfg.nonlinear.backend = backend_norm
        if max_outer_iter is not None:
            cfg.nonlinear.max_outer_iter = int(max_outer_iter)
        if force_nonlinear:
            cfg.nonlinear.enabled = True

        if cfg.time.dt <= 0.0:
            logger.error("cfg.time.dt must be positive (got %s)", cfg.time.dt)
            return 2
        if cfg.time.t_end <= cfg.time.t0:
            logger.error("cfg.time.t_end must exceed t0 (t0=%s, t_end=%s)", cfg.time.t0, cfg.time.t_end)
            return 2

        nl_cfg = getattr(cfg, "nonlinear", None)
        use_nonlinear = bool(getattr(nl_cfg, "enabled", False))
        nl_backend = str(getattr(nl_cfg, "backend", "scipy")).lower()
        solver_cfg = getattr(cfg, "solver", None)
        linear_cfg = getattr(solver_cfg, "linear", None)
        linear_backend = str(getattr(linear_cfg, "backend", "scipy")).lower()
        backend_for_rules = _normalize_backend(nl_backend if use_nonlinear else linear_backend)
        if (not use_nonlinear) and backend_for_rules == "petsc_serial":
            backend_for_rules = "petsc_mpi"

        is_root = rank == 0
        if backend_for_rules == "scipy" and size > 1:
            if is_root:
                logger.error("backend=scipy requires MPI size==1 (got size=%d).", size)
            return 2
        if use_nonlinear and backend_for_rules == "petsc_serial" and size > 1:
            mpi_mode = getattr(getattr(cfg, "petsc", None), "mpi_mode", None)
            allow_redundant = str(mpi_mode).lower() == "redundant" if mpi_mode is not None else False
            if not allow_redundant:
                if is_root:
                    logger.error(
                        "backend=petsc_serial requires MPI size==1 (got size=%d); "
                        "use petsc_mpi for parallel runs.",
                        size,
                    )
                return 2
        if backend_for_rules == "petsc_mpi" and size == 1 and is_root:
            logger.warning("backend=petsc_mpi with size=1; running effectively in serial.")

        if size > 1:
            if is_root:
                logger.warning("MPI size=%d: scalars output is root-only.", size)
            else:
                try:
                    cfg.io.fields.scalars = []
                except Exception:
                    pass
                try:
                    cfg.io.scalars_write_every = 0
                except Exception:
                    pass

        petsc_needed = backend_for_rules in ("petsc_serial", "petsc_mpi")
        if prefix is not None:
            cfg.petsc.options_prefix = _format_prefix(prefix)
        elif petsc_needed and not str(getattr(cfg.petsc, "options_prefix", "") or "").strip():
            case_id = getattr(cfg.case, "id", "case")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg.petsc.options_prefix = f"evap_{case_id}_{stamp}_"
        if petsc_needed and is_root and getattr(cfg.petsc, "options_prefix", ""):
            logger.info("PETSc options_prefix: %s", cfg.petsc.options_prefix)

        if is_root:
            jac_mode = getattr(cfg.petsc, "jacobian_mode", "")
            logger.info(
                "Case: %s backend=%s mpi_size=%d jacobian_mode=%s",
                cfg_path,
                backend_for_rules,
                size,
                jac_mode,
            )

        if use_nonlinear:
            if nl_backend in ("petsc", "snes", "petsc_snes", "petsc_serial", "petsc_mpi"):
                petsc_cfg = getattr(cfg, "petsc", None)
                logger.info(
                    "Global nonlinear solve enabled (backend=%s, snes=%s, linesearch=%s, jacobian=%s, ksp=%s, pc=%s, max_outer_iter=%d).",
                    nl_backend,
                    getattr(petsc_cfg, "snes_type", "newtonls"),
                    getattr(petsc_cfg, "linesearch_type", "bt"),
                    getattr(petsc_cfg, "jacobian_mode", "fd"),
                    getattr(petsc_cfg, "ksp_type", "gmres"),
                    getattr(petsc_cfg, "pc_type", "ilu"),
                    int(getattr(nl_cfg, "max_outer_iter", 0)),
                )
            else:
                logger.info(
                    "Global nonlinear solve enabled (backend=%s, solver=%s, krylov=%s, max_outer_iter=%d).",
                    nl_backend,
                    getattr(nl_cfg, "solver", "newton_krylov"),
                    getattr(nl_cfg, "krylov_method", "lgmres"),
                    int(getattr(nl_cfg, "max_outer_iter", 0)),
                )
        else:
            logger.info(
                "Global nonlinear solve disabled; using linear backend=%s.",
                linear_backend,
            )

        run_dir = _prepare_run_dir(cfg, cfg_path)
        logger.info("Run directory: %s", run_dir)
        try:
            log_path = run_dir / "run.log"
            root_logger = logging.getLogger()
            existing = [
                h for h in root_logger.handlers
                if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path
            ]
            if not existing:
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                file_handler.setLevel(level)
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
                )
                root_logger.addHandler(file_handler)
                logger.info("Logging to file: %s", log_path)
        except Exception as exc:
            logger.warning("Failed to set up file logging: %s", exc)

        gas_model, liq_model = get_or_build_models(cfg)
        _maybe_fill_gas_species(cfg, gas_model)

        grid = build_grid(cfg)
        layout = build_layout(cfg, grid)

        # Write mapping.json once after layout is created
        _write_mapping_once(cfg, grid, layout)

        state = build_initial_state_erfc(cfg, grid, gas_model, liq_model)
        props, _ = compute_props(cfg, grid, state)
        if dry_run:
            logger.info("Dry run requested: config and models built; skipping time loop.")
            return 0

        # Output writers and diagnostics
        scalars_writer = None
        eq_model = None
        need_eq = False
        Rd_stop = None

        # Radius-based early stop criterion
        RD_STOP_FACTOR = 0.1
        try:
            Rd0 = float(state.Rd)
        except Exception:
            Rd0 = None
        if Rd0 is not None and np.isfinite(Rd0) and Rd0 > 0.0:
            Rd_stop = RD_STOP_FACTOR * Rd0
            logger.info(
                "Radius stop criterion enabled: Rd_stop = %.6e = %.3f * Rd0 (Rd0 = %.6e).",
                Rd_stop,
                RD_STOP_FACTOR,
                Rd0,
            )
        else:
            logger.warning("Could not determine initial Rd; radius stop criterion disabled.")

        try:
            if size == 1 or is_root:
                writers_module = _load_writers_module()
                writer_cls = getattr(writers_module, "ScalarsWriter", None)
                if writer_cls is None:
                    logger.warning("ScalarsWriter not found in io.writers; scalar output disabled.")
                else:
                    scalars_writer = writer_cls(cfg, state, props, out_dir=run_dir)
        except Exception as exc:
            logger.warning("Failed to initialize scalars writer: %s", exc)

        io_fields = getattr(getattr(cfg, "io", None), "fields", None)
        scalars_fields = list(getattr(io_fields, "scalars", []) or [])
        need_eq = any(name.startswith("Yg_eq_") for name in scalars_fields)
        if scalars_writer is not None and getattr(scalars_writer, "enabled", False) and need_eq and cfg.physics.include_mpp:
            Ns_g = len(cfg.species.gas_species_full)
            Ns_l = len(cfg.species.liq_species)
            M_g = np.ones(Ns_g, dtype=float)
            M_l = np.ones(Ns_l, dtype=float)
            try:
                M_g = np.asarray(gas_model.gas.molecular_weights, dtype=float) / 1000.0
            except Exception:
                pass
            try:
                for i, name in enumerate(cfg.species.liq_species):
                    if name in cfg.species.mw_kg_per_mol:
                        M_l[i] = float(cfg.species.mw_kg_per_mol[name])
            except Exception:
                pass
            try:
                eq_model = build_equilibrium_model(cfg, Ns_g=Ns_g, Ns_l=Ns_l, M_g=M_g, M_l=M_l)
            except Exception as exc:
                logger.warning("Failed to build equilibrium model for diagnostics: %s", exc)
                eq_model = None

        t = float(cfg.time.t0)
        step_id = 0
        effective_max_steps = max_steps if max_steps is not None else cfg.time.max_steps
        ended_by_radius = False

        _maybe_write_spatial(cfg, grid, state, step_id, t)

        # Write initial u vector and grid coordinates
        _maybe_write_u(cfg, grid, state, layout, step_id, t)

        while t < cfg.time.t_end:
            step_id += 1
            if effective_max_steps is not None and step_id > effective_max_steps:
                logger.error(
                    "Reached max_steps=%s before t_end=%.3e (t=%.3e); aborting.",
                    effective_max_steps,
                    cfg.time.t_end,
                    t,
                )
                return 2

            res = advance_one_step_scipy(cfg, grid, layout, state, props, t)
            _log_step(res, step_id)
            if use_nonlinear and bool(getattr(nl_cfg, "verbose", False)):
                try:
                    nl_info = res.diag.extra.get("nonlinear", {}) or {}
                    extra = nl_info.get("extra", {}) or {}
                except Exception:
                    extra = {}
                if extra:
                    logger.info(
                        "[NL extra] n_func_eval=%s n_jac_eval=%s ksp_its_total=%s "
                        "time_total=%.3e time_func=%.3e time_jac=%.3e time_linear=%.3e",
                        extra.get("n_func_eval", "NA"),
                        extra.get("n_jac_eval", "NA"),
                        extra.get("ksp_its_total", "NA"),
                        float(extra.get("time_total", float("nan"))),
                        float(extra.get("time_func", float("nan"))),
                        float(extra.get("time_jac", float("nan"))),
                        float(extra.get("time_linear_total", float("nan"))),
                    )

            if not res.success:
                logger.error("Step %s failed: %s", step_id, res.message)
                if res.diag.extra:
                    logger.error("Diagnostics extra: %s", res.diag.extra)
                return 2
            if use_nonlinear:
                if not res.diag.nonlinear_converged:
                    logger.error("Nonlinear solver not converged at step %s (diag)", step_id)
                    return 2
            else:
                if not res.diag.linear_converged:
                    logger.error("Linear solver not converged at step %s (diag)", step_id)
                    return 2

            sanity_msg = _sanity_check_state(res.state_new)
            if sanity_msg is not None:
                logger.error("Sanity check failed at step %s: %s", step_id, sanity_msg)
                return 2

            state = res.state_new
            props = res.props_new
            t = res.diag.t_new

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
                        if pc_stats.status == "ERROR" and is_root:
                            logger.warning(
                                "post-correct error: res0=%.3e res_best=%.3e iters=%d reason=%s",
                                pc_stats.res0,
                                pc_stats.res_best,
                                pc_stats.iters,
                                pc_stats.reason,
                            )
                        props, _ = compute_props(cfg, grid_new, state)
                    # Rebuild equilibrium model to stay consistent with remapped state
                    if (
                        scalars_writer is not None
                        and getattr(scalars_writer, "enabled", False)
                        and need_eq
                        and cfg.physics.include_mpp
                    ):
                        try:
                            Ns_g = len(cfg.species.gas_species_full)
                            Ns_l = len(cfg.species.liq_species)
                            M_g = np.ones(Ns_g, dtype=float)
                            M_l = np.ones(Ns_l, dtype=float)
                            try:
                                M_g = np.asarray(gas_model.gas.molecular_weights, dtype=float) / 1000.0
                            except Exception:
                                pass
                            try:
                                for i, name in enumerate(cfg.species.liq_species):
                                    if name in cfg.species.mw_kg_per_mol:
                                        M_l[i] = float(cfg.species.mw_kg_per_mol[name])
                            except Exception:
                                pass
                            eq_model = build_equilibrium_model(cfg, Ns_g=Ns_g, Ns_l=Ns_l, M_g=M_g, M_l=M_l)
                        except Exception as exc:
                            logger.warning("Failed to rebuild equilibrium model after remeshing: %s", exc)
                    grid = grid_new
                except Exception as exc:
                    logger.error("Failed to remesh with updated Rd: %s", exc)
                    return 2

            if scalars_writer is not None:
                try:
                    scalars_writer.maybe_write(
                        step_id=step_id,
                        grid=grid,
                        state=state,
                        props=props,
                        diag=res.diag,
                        t=t,
                        eq_model=eq_model,
                    )
                except Exception as exc:
                    logger.warning("Failed to write scalars at step %s: %s", step_id, exc)

            _maybe_write_spatial(cfg, grid, state, step_id, t)

            # Write u vector and grid coordinates (new spatial output)
            _maybe_write_u(cfg, grid, state, layout, step_id, t)

            if Rd_stop is not None:
                try:
                    Rd_now = float(state.Rd)
                except Exception:
                    Rd_now = None
                if Rd_now is not None and np.isfinite(Rd_now) and Rd_now <= Rd_stop:
                    logger.info(
                        "Stopping run: Rd=%.6e <= Rd_stop=%.6e at t=%.6e (step %d).",
                        Rd_now,
                        Rd_stop,
                        t,
                        step_id,
                    )
                    ended_by_radius = True
                    break

        if ended_by_radius:
            logger.info(
                "Completed run early: t=%.6e after %d steps; Rd hit threshold Rd_stop=%.6e (final Rd=%.6e).",
                t,
                step_id,
                float(Rd_stop) if Rd_stop is not None else float("nan"),
                float(state.Rd),
            )
        else:
            logger.info("Completed run: t=%.6e reached t_end=%.6e after %d steps.", t, cfg.time.t_end, step_id)
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Unhandled exception:\n%s", tb)
        return 99
    finally:
        try:
            if "scalars_writer" in locals() and scalars_writer is not None:
                scalars_writer.close()
        except Exception:
            pass


def _parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run an evaporation case.")
    parser.add_argument("case_yaml", help="Path to case YAML file.")
    parser.add_argument(
        "--backend",
        choices=("scipy", "petsc_serial", "petsc_mpi"),
        default=None,
        help="Override nonlinear backend (default: use YAML).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Override PETSc options prefix (default: use YAML or auto-generate).",
    )
    parser.add_argument(
        "--max_outer_iter",
        type=int,
        default=None,
        help="Override nonlinear max_outer_iter (default: use YAML).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load config and build models only; skip time stepping.",
    )
    args, unknown = parser.parse_known_args(argv)
    return args, list(unknown)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, petsc_args = _parse_args(argv)
    # Prevent PETSc from parsing driver-specific CLI flags.
    sys.argv = [sys.argv[0]] + list(petsc_args)
    return run_case(
        args.case_yaml,
        backend=args.backend,
        prefix=args.prefix,
        max_outer_iter=args.max_outer_iter,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
