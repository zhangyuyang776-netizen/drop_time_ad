"""
Output helpers:
- ScalarsWriter: configurable scalar output to CSV with optional cumulative diagnostics.
- write_step_scalars: stateless per-step scalars (fields controlled by cfg.io.fields.scalars).
- write_step_spatial: spatial snapshot output.
- get_run_dir: unified output directory management (3D_out/case_xxx/run_yyy/).
- build_u_mapping: generate mapping.json for u vector layout.
- write_step_u: write u vector + grid coordinates to npz files.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.types import CaseConfig, Grid1D, State
from properties.equilibrium import compute_interface_equilibrium

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from solvers.timestepper import StepDiagnostics
    from core.layout import UnknownLayout

logger = logging.getLogger(__name__)

# ============================================================================
# Step 1: Unified output directory management
# ============================================================================

_RUN_DIR_CACHE: dict[int, Path] = {}


def get_run_dir(cfg: CaseConfig) -> Path:
    """
    Get the 3D_out subdirectory within the current run directory.

    Returns <case_dir>/3D_out where case_dir is the existing run directory
    created by _prepare_run_dir() in the driver.

    Uses module-level cache to ensure same directory across entire run.
    """
    cfg_id = id(cfg)
    if cfg_id in _RUN_DIR_CACHE:
        return _RUN_DIR_CACHE[cfg_id]

    # Get the existing run directory from cfg.paths.case_dir
    # This is set by _prepare_run_dir() in the driver
    case_dir = getattr(getattr(cfg, "paths", None), "case_dir", None)

    if case_dir is None:
        # Fallback: create a default directory
        logger.warning("cfg.paths.case_dir not set, using default 'out' directory")
        case_dir = Path("out")

    case_dir = Path(case_dir)

    # Create 3D_out subdirectory within the run directory
    spatial_out_dir = case_dir / "3D_out"
    spatial_out_dir.mkdir(parents=True, exist_ok=True)

    # Cache it
    _RUN_DIR_CACHE[cfg_id] = spatial_out_dir

    logger.info(f"Spatial output directory: {spatial_out_dir}")
    return spatial_out_dir


# ============================================================================
# Step 2: Build and write mapping.json
# ============================================================================


def build_u_mapping(cfg: CaseConfig, grid: Grid1D, layout: "UnknownLayout") -> dict:
    """
    Build mapping metadata for the u vector layout.

    Returns a dictionary describing how to unpack u into fields:
    - version, endianness, dtype, ordering
    - blocks: list of {name, offset, size, shape, optional}
    - meta: grid sizes, species lists, etc.
    """
    blocks = []
    offset = 0

    # Iterate through blocks in layout order
    for block_name, block_slice in layout.iter_blocks():
        block_start = block_slice.start
        block_stop = block_slice.stop
        block_size = block_stop - block_start

        # Determine shape based on block type
        if block_name == "Tg":
            shape = [layout.Ng]
            name_display = "Tg"
        elif block_name == "Yg":
            # Yg is stored as (ig * Ns_g_eff + k_red)
            # Shape reflects actual storage order: (Ng, Ns_g_eff)
            # Post-processing should transpose to (Ns_g_eff, Ng)
            shape = [layout.Ng, layout.Ns_g_eff]
            name_display = "Yg"
        elif block_name == "Tl":
            shape = [layout.Nl]
            name_display = "Tl"
        elif block_name == "Yl":
            # Yl is stored as (il * Ns_l_eff + k_red)
            # Shape reflects actual storage order: (Nl, Ns_l_eff)
            # Post-processing should transpose to (Ns_l_eff, Nl)
            shape = [layout.Nl, layout.Ns_l_eff]
            name_display = "Yl"
        elif block_name == "Ts":
            shape = [1]
            name_display = "Ts"
        elif block_name == "mpp":
            shape = [1]
            name_display = "mpp"
        elif block_name == "Rd":
            shape = [1]
            name_display = "Rd"
        else:
            # Unknown block, use flat shape
            shape = [block_size]
            name_display = block_name

        block_info = {
            "name": name_display,
            "offset": int(block_start),
            "size": int(block_size),
            "shape": [int(s) for s in shape],
        }

        # Mark optional blocks (Yl might not always be present)
        if block_name in ("Yl", "Ts", "mpp", "Rd"):
            block_info["optional"] = True

        blocks.append(block_info)

    # Build metadata
    meta = {
        "Ng": int(layout.Ng),
        "Nl": int(layout.Nl),
        "Ns_g_full": int(layout.Ns_g_full),
        "Ns_g_eff": int(layout.Ns_g_eff),
        "Ns_l_full": int(layout.Ns_l_full),
        "Ns_l_eff": int(layout.Ns_l_eff),
        "species_g_full": list(layout.gas_species_full),
        "species_g_reduced": list(layout.gas_species_reduced),
        "species_g_closure": layout.gas_closure_species,
        "species_l_full": list(layout.liq_species_full),
        "species_l_reduced": list(layout.liq_species_reduced),
        "species_l_closure": layout.liq_closure_species,
    }

    # Determine system properties
    endianness = sys.byteorder  # 'little' or 'big'

    mapping = {
        "version": 1,
        "endianness": endianness,
        "dtype": "float64",
        "ordering": "C",  # Row-major (C-style)
        "total_size": int(layout.size),
        "blocks": blocks,
        "meta": meta,
    }

    return mapping


def write_mapping_json(cfg: CaseConfig, grid: Grid1D, layout: "UnknownLayout", run_dir: Path | None = None) -> None:
    """
    Write mapping.json to the run directory.

    Uses atomic write (temp file + rename) to avoid corruption.
    Rank0-only in parallel contexts.
    """
    # Check if we're rank 0 in MPI context (simple check)
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank != 0:
            return  # Only rank 0 writes
    except ImportError:
        pass  # Serial mode, proceed

    if run_dir is None:
        run_dir = get_run_dir(cfg)

    mapping = build_u_mapping(cfg, grid, layout)

    out_path = run_dir / "mapping.json"
    tmp_path = run_dir / "mapping.json.tmp"

    # Write to temp file
    with open(tmp_path, "w") as f:
        json.dump(mapping, f, indent=2)

    # Atomic rename
    os.replace(tmp_path, out_path)

    logger.info(f"Wrote mapping.json: {out_path}")


# ============================================================================
# Step 3: Write u vector + grid coordinates to npz files
# ============================================================================


def write_step_u(
    cfg: CaseConfig, step_id: int, t: float, u: np.ndarray, grid: Grid1D, run_dir: Path | None = None
) -> None:
    """
    Write u vector and grid coordinates for a single time step.

    Output: <run_dir>/steps/step_{step_id:06d}_time_{t:.6e}s.npz

    Contents:
    - step_id: int
    - t: float (seconds)
    - u: float64 1D array (full unknown vector)
    - r_g: float64 1D array (gas phase cell centers)
    - r_l: float64 1D array (liquid phase cell centers)
    - rf_g: float64 1D array (gas phase face centers)
    - rf_l: float64 1D array (liquid phase face centers)
    - iface_f: int (interface face index)

    Rank0-only in parallel contexts.
    """
    # Check if we're rank 0 in MPI context
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank != 0:
            return  # Only rank 0 writes
    except ImportError:
        pass  # Serial mode, proceed

    if run_dir is None:
        run_dir = get_run_dir(cfg)

    steps_dir = run_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = f"step_{step_id:06d}_time_{t:.6e}s.npz"
    out_path = steps_dir / filename

    # Extract grid coordinates
    r_c = np.asarray(grid.r_c, dtype=np.float64)
    r_f = np.asarray(grid.r_f, dtype=np.float64)

    # Split into gas/liquid
    liq_slice, gas_slice = grid.split_cells()
    r_l = r_c[liq_slice]
    r_g = r_c[gas_slice]

    # Face coordinates (interface is at index Nl)
    rf_l = r_f[: grid.Nl + 1]  # 0 to Nl (inclusive)
    rf_g = r_f[grid.Nl :]  # Nl to end

    # Save to npz
    np.savez(
        out_path,
        step_id=np.asarray(step_id, dtype=np.int32),
        t=np.asarray(t, dtype=np.float64),
        u=np.asarray(u, dtype=np.float64),
        r_g=r_g,
        r_l=r_l,
        rf_g=rf_g,
        rf_l=rf_l,
        iface_f=np.asarray(grid.iface_f, dtype=np.int32),
    )

    logger.debug(f"Wrote step file: {out_path}")


# ============================================================================
# Step 4: Output control strategy
# ============================================================================


def should_write_u(cfg: CaseConfig, step_id: int) -> bool:
    """
    Determine if u vector should be written for this time step.

    Checks:
    1. Environment variable DROPLET_WRITE_U=1 (force enable)
    2. cfg.output.u_enabled (bool, default False)
    3. cfg.output.u_every (int, default 1) - write every N steps

    Returns True if u should be written for this step_id.
    """
    # Check environment variable (override)
    env_enabled = os.getenv("DROPLET_WRITE_U", "0") == "1"
    if env_enabled:
        logger.debug(f"u output enabled via DROPLET_WRITE_U env var at step {step_id}")
        return True

    # Check config
    output_cfg = getattr(cfg, "output", None)
    if output_cfg is None:
        logger.debug(f"No output config found, u output disabled at step {step_id}")
        return False

    u_enabled = getattr(output_cfg, "u_enabled", False)
    if not u_enabled:
        logger.debug(f"u_enabled=False, u output disabled at step {step_id}")
        return False

    u_every = int(getattr(output_cfg, "u_every", 1))
    if u_every <= 0:
        logger.warning(f"u_every={u_every} <= 0, u output disabled at step {step_id}")
        return False

    # Check frequency
    should_write = (step_id % u_every) == 0
    if should_write:
        logger.debug(f"u output enabled at step {step_id} (u_every={u_every})")
    return should_write


def _ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _scalar_fields(cfg: CaseConfig) -> list[str]:
    fields = getattr(getattr(cfg, "io", None), "fields", None)
    return list(getattr(fields, "scalars", []) or [])


def _scalar_output_path(cfg: CaseConfig, out_dir: Path | str | None = None) -> Path:
    base = Path(out_dir) if out_dir is not None else (Path(cfg.paths.case_dir) / "scalars")
    return base / "scalars.csv"


def _as_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _is_finite(value: float) -> bool:
    try:
        return np.isfinite(value)
    except Exception:
        return False


def _compute_scalar_row(
    fields: list[str],
    *,
    cfg: CaseConfig,
    grid: Grid1D | None,
    state: State,
    props,
    diag: "StepDiagnostics" | None,
    step_id: int | None,
    t: float | None,
    integ: dict | None,
    eq_model,
    unknown_fields: set[str] | None = None,
) -> list[float]:
    gas_names = list(getattr(cfg.species, "gas_species_full", []))

    t_old = _as_float(getattr(diag, "t_old", math.nan)) if diag is not None else math.nan
    t_new = _as_float(getattr(diag, "t_new", math.nan)) if diag is not None else math.nan
    dt = _as_float(getattr(diag, "dt", math.nan)) if diag is not None else math.nan
    t_val = _as_float(t) if t is not None else (t_new if _is_finite(t_new) else math.nan)

    Ts = _as_float(getattr(state, "Ts", math.nan))
    Rd = _as_float(getattr(state, "Rd", math.nan))
    mpp = _as_float(getattr(state, "mpp", math.nan))

    A_if = 4.0 * math.pi * Rd * Rd if _is_finite(Rd) else math.nan
    m_dot = A_if * mpp if _is_finite(A_if) and _is_finite(mpp) else math.nan

    m_evap_cum = math.nan
    m_drop0 = math.nan
    m_drop = math.nan
    mass_closure_err = math.nan
    if integ is not None:
        m_drop0 = _as_float(integ.get("m_drop0", math.nan))
        if _is_finite(m_dot):
            if _is_finite(dt):
                if integ.get("m_dot_prev") is None:
                    integ["m_evap_cum"] = _as_float(integ.get("m_evap_cum", 0.0)) + m_dot * dt
                else:
                    integ["m_evap_cum"] = _as_float(integ.get("m_evap_cum", 0.0)) + 0.5 * (
                        integ["m_dot_prev"] + m_dot
                    ) * dt
            integ["m_dot_prev"] = m_dot
        m_evap_cum = _as_float(integ.get("m_evap_cum", math.nan))

    rho_l_mean = math.nan
    if props is not None and hasattr(props, "rho_l"):
        try:
            if props.rho_l is not None and props.rho_l.size:
                rho_l_mean = float(np.mean(props.rho_l))
        except Exception:
            pass
    if _is_finite(rho_l_mean) and _is_finite(Rd):
        m_drop = 4.0 / 3.0 * math.pi * rho_l_mean * Rd**3
    if _is_finite(m_drop) and _is_finite(m_drop0):
        mass_closure_err = m_drop + m_evap_cum - m_drop0

    Tg = getattr(state, "Tg", np.array([], dtype=float))
    Tl = getattr(state, "Tl", np.array([], dtype=float))
    Tg_min = _as_float(getattr(diag, "Tg_min", math.nan)) if diag is not None else math.nan
    Tg_max = _as_float(getattr(diag, "Tg_max", math.nan)) if diag is not None else math.nan
    if not _is_finite(Tg_min) and Tg.size:
        Tg_min = float(np.min(Tg))
    if not _is_finite(Tg_max) and Tg.size:
        Tg_max = float(np.max(Tg))

    Tg_mean = float(np.mean(Tg)) if Tg.size else math.nan
    Tl_mean = float(np.mean(Tl)) if Tl.size else math.nan
    Tg0 = float(Tg[0]) if Tg.size else math.nan
    Tg_if = Tg0
    Tg_far = float(Tg[-1]) if Tg.size else math.nan
    Tl_center = float(Tl[0]) if Tl.size else math.nan
    Tl_if = float(Tl[-1]) if Tl.size else math.nan
    Tl_min = float(np.min(Tl)) if Tl.size else math.nan
    Tl_max = float(np.max(Tl)) if Tl.size else math.nan

    Yg0 = state.Yg[:, 0] if getattr(state, "Yg", np.array([], dtype=float)).size else np.array([], dtype=float)
    sumYg0 = float(np.sum(Yg0)) if Yg0.size else math.nan

    cp_g0 = math.nan
    k_g0 = math.nan
    if props is not None:
        try:
            if hasattr(props, "cp_g") and props.cp_g.size:
                cp_g0 = float(props.cp_g[0])
        except Exception:
            pass
        try:
            if hasattr(props, "k_g") and props.k_g.size:
                k_g0 = float(props.k_g[0])
        except Exception:
            pass

    energy_balance_if = _as_float(getattr(diag, "energy_balance_if", math.nan)) if diag is not None else math.nan
    mass_balance_rd = _as_float(getattr(diag, "mass_balance_rd", math.nan)) if diag is not None else math.nan

    Yg_eq = None
    need_eq = any(name.startswith("Yg_eq_") for name in fields)
    if need_eq and eq_model is not None and grid is not None and gas_names:
        try:
            il_if = grid.Nl - 1
            ig_if = 0
            Yl_face = state.Yl[:, il_if] if state.Yl.size else np.zeros((len(cfg.species.liq_species),), float)
            Yg_face = state.Yg[:, ig_if] if state.Yg.size else np.zeros((len(gas_names),), float)
            Pg = float(getattr(cfg.initial, "P_inf", 101325.0))
            Yg_eq, _, _ = compute_interface_equilibrium(
                eq_model, Ts=float(state.Ts), Pg=Pg, Yl_face=Yl_face, Yg_face=Yg_face
            )
        except Exception:
            Yg_eq = None

    base = {
        "step": step_id if step_id is not None else math.nan,
        "t": t_val,
        "t_old": t_old,
        "t_new": t_new,
        "dt": dt,
        "Rd": Rd,
        "mpp": mpp,
        "Ts": Ts,
        "A_if": A_if,
        "m_dot": m_dot,
        "m_evap_cum": m_evap_cum,
        "m_drop": m_drop,
        "m_drop0": m_drop0,
        "mass_closure_err": mass_closure_err,
        "Tg0": Tg0,
        "Tg_min": Tg_min,
        "Tg_max": Tg_max,
        "Tg_if": Tg_if,
        "Tg_far": Tg_far,
        "Tg_mean": Tg_mean,
        "Tl_center": Tl_center,
        "Tl_if": Tl_if,
        "Tl_min": Tl_min,
        "Tl_max": Tl_max,
        "Tl_mean": Tl_mean,
        "sumYg0": sumYg0,
        "cp_g0": cp_g0,
        "k_g0": k_g0,
        "energy_balance_if": energy_balance_if,
        "mass_balance_rd": mass_balance_rd,
    }

    row: list[float] = []
    for field in fields:
        if field in base:
            row.append(_as_float(base[field]))
            continue
        if field.startswith("Yg0_"):
            name = field[len("Yg0_") :]
            val = math.nan
            if name in gas_names:
                idx = gas_names.index(name)
                if idx < Yg0.shape[0]:
                    val = float(Yg0[idx])
            row.append(val)
            continue
        if field.startswith("Yg_eq_"):
            name = field[len("Yg_eq_") :]
            val = math.nan
            if Yg_eq is not None and name in gas_names:
                idx = gas_names.index(name)
                if idx < Yg_eq.shape[0]:
                    val = float(Yg_eq[idx])
            row.append(val)
            continue
        if diag is not None and hasattr(diag, field):
            val = getattr(diag, field)
            if np.isscalar(val):
                row.append(_as_float(val))
                continue
        if hasattr(state, field):
            val = getattr(state, field)
            if np.isscalar(val):
                row.append(_as_float(val))
                continue
        if unknown_fields is not None and field not in unknown_fields:
            logger.warning("Unknown scalar field '%s'; writing NaN.", field)
            unknown_fields.add(field)
        row.append(math.nan)
    return row


class ScalarsWriter:
    def __init__(
        self,
        cfg: CaseConfig,
        state0: State | None = None,
        props0=None,
        *,
        out_dir: Path | str | None = None,
    ) -> None:
        self.cfg = cfg
        self.fields = _scalar_fields(cfg)
        self.scalars_every = int(getattr(cfg.io, "scalars_write_every", getattr(cfg.io, "write_every", 1)) or 0)
        self.enabled = self.scalars_every > 0
        self.out_path = _scalar_output_path(cfg, out_dir=out_dir)
        self._fh = None
        self._writer = None
        self._unknown_fields: set[str] = set()
        self._integ = {"m_evap_cum": 0.0, "m_dot_prev": None, "m_drop0": math.nan}
        if state0 is not None and props0 is not None:
            self._init_mass_integrator(state0, props0)
        if self.enabled:
            self._open()

    def _open(self) -> None:
        _ensure_parent(self.out_path)
        self._fh = self.out_path.open("w", newline="")
        self._writer = csv.writer(self._fh)
        if self.fields:
            self._writer.writerow(self.fields)
        try:
            self._fh.flush()
        except Exception:
            pass

    def _init_mass_integrator(self, state0: State, props0) -> None:
        try:
            rho_l_mean = float(np.mean(props0.rho_l)) if hasattr(props0, "rho_l") else math.nan
            Rd0 = float(state0.Rd)
            if np.isfinite(rho_l_mean) and Rd0 > 0.0:
                self._integ["m_drop0"] = 4.0 / 3.0 * math.pi * rho_l_mean * Rd0**3
        except Exception:
            pass

    def maybe_write(
        self,
        *,
        step_id: int,
        grid: Grid1D,
        state: State,
        props,
        diag: "StepDiagnostics",
        t: float | None = None,
        eq_model=None,
    ) -> None:
        if not self.enabled:
            return
        if step_id % self.scalars_every != 0:
            return
        if self._writer is None or self._fh is None:
            self._open()
        if not self.fields:
            return
        row = _compute_scalar_row(
            self.fields,
            cfg=self.cfg,
            grid=grid,
            state=state,
            props=props,
            diag=diag,
            step_id=step_id,
            t=t,
            integ=self._integ,
            eq_model=eq_model,
            unknown_fields=self._unknown_fields,
        )
        self._writer.writerow(row)
        try:
            self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
                self._writer = None


def write_step_scalars(cfg: CaseConfig, t: float, state: State, diag: "StepDiagnostics") -> None:
    fields = _scalar_fields(cfg)
    if not fields:
        return
    out_path = _scalar_output_path(cfg, out_dir=None)
    _ensure_parent(out_path)

    row = _compute_scalar_row(
        fields,
        cfg=cfg,
        grid=None,
        state=state,
        props=None,
        diag=diag,
        step_id=None,
        t=t,
        integ=None,
        eq_model=None,
        unknown_fields=None,
    )

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header and fields:
            writer.writerow(fields)
        if fields:
            writer.writerow(row)


def write_step_spatial(cfg: CaseConfig, grid: Grid1D, state: State, step_id: int | None = None, t: float | None = None) -> None:
    """
    Write spatial snapshot to ``spatial/snapshot_XXXXXX.npz`` under the case directory.

    - Uses cfg.io.fields.{gas, liquid, interface} as allow-lists.
    - Includes r_c/r_f (if present) and a running r_index for convenience.
    - Adds step_id/t when provided by the caller.
    """
    out_dir = (Path(cfg.paths.case_dir) / "spatial") if hasattr(cfg, "paths") else Path("spatial")
    out_dir.mkdir(parents=True, exist_ok=True)

    counter_path = out_dir / "_spatial_index.txt"
    try:
        idx = int(counter_path.read_text().strip())
    except FileNotFoundError:
        idx = 0
    except ValueError:
        idx = 0
    counter_path.write_text(str(idx + 1))

    out_path = out_dir / f"snapshot_{idx:06d}.npz"

    data = {}

    # Grid metadata (best effort)
    try:
        rc = getattr(grid, "r_c", None)
        rf = getattr(grid, "r_f", None)
        if rc is not None:
            rc_arr = np.asarray(rc)
            data["r_c"] = rc_arr
            data["r_index"] = np.arange(rc_arr.size)
        if rf is not None:
            data["r_f"] = np.asarray(rf)
    except Exception:
        pass

    if step_id is not None:
        data["step_id"] = np.asarray(step_id)
    if t is not None:
        data["t"] = np.asarray(t)

    fields = getattr(cfg, "io", None)
    field_cfg = getattr(fields, "fields", None) if fields is not None else None

    gas_fields = getattr(field_cfg, "gas", []) or []
    for name in gas_fields:
        if not hasattr(state, name):
            continue
        data[name] = np.asarray(getattr(state, name))

    liq_fields = getattr(field_cfg, "liquid", []) or []
    for name in liq_fields:
        if not hasattr(state, name):
            continue
        data[name] = np.asarray(getattr(state, name))

    iface_fields = getattr(field_cfg, "interface", []) or []
    for name in iface_fields:
        if not hasattr(state, name):
            continue
        data[name] = np.asarray(getattr(state, name))

    np.savez(out_path, **data)


# ============================================================================
# P0.2: Interface diagnostics CSV output
# ============================================================================

_INTERFACE_DIAG_FIELDS = [
    "step",
    "t",
    "dt",
    "Ts",
    "Rd",
    "mpp",
    "m_dot",
    "psat",
    "sum_y_cond",
    "eq_source",
    "eq_exc_type",
    "eq_exc_msg",
]


def write_interface_diag(cfg: CaseConfig, diag: "StepDiagnostics") -> None:
    """
    Write interface diagnostics to interface_diag.csv (rank0 only, append mode).

    P0.2: This function extracts interface_diag dict from diag.extra and writes
    one row per timestep to interface_diag.csv in the case directory.

    The CSV includes:
    - step: auto-incremented step counter (using _interface_diag_index.txt)
    - t, dt: time and timestep
    - Ts, Rd, mpp, m_dot: interface state variables
    - psat, sum_y_cond: equilibrium diagnostics
    - eq_source, eq_exc_type, eq_exc_msg: equilibrium computation status

    Only writes on rank0 (checked via is_root_rank).
    """
    # Check if we're on rank0
    try:
        from core.logging_utils import is_root_rank

        if not is_root_rank():
            return
    except Exception:
        pass  # If import fails, assume rank0 and proceed

    # Extract interface_diag from diag.extra
    try:
        interface_diag = diag.extra.get("interface_diag")
        if interface_diag is None:
            return  # No interface_diag data, skip
    except Exception:
        return

    # Determine output path
    try:
        case_dir = Path(cfg.paths.case_dir) if hasattr(cfg, "paths") else Path(".")
    except Exception:
        case_dir = Path(".")

    out_path = case_dir / "interface_diag.csv"

    # Get or increment step counter (using persistent file)
    counter_path = case_dir / "_interface_diag_index.txt"
    try:
        step = int(counter_path.read_text().strip())
    except (FileNotFoundError, ValueError):
        step = 0
    counter_path.write_text(str(step + 1))

    # Prepare row data
    row = {
        "step": step,
        "t": interface_diag.get("t", ""),
        "dt": interface_diag.get("dt", ""),
        "Ts": interface_diag.get("Ts", ""),
        "Rd": interface_diag.get("Rd", ""),
        "mpp": interface_diag.get("mpp", ""),
        "m_dot": interface_diag.get("m_dot", ""),
        "psat": interface_diag.get("psat", ""),
        "sum_y_cond": interface_diag.get("sum_y_cond", ""),
        "eq_source": interface_diag.get("eq_source", ""),
        "eq_exc_type": interface_diag.get("eq_exc_type", ""),
        "eq_exc_msg": interface_diag.get("eq_exc_msg", ""),
    }

    # Write to CSV (append mode, write header if new file)
    is_new = not out_path.exists()
    try:
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_INTERFACE_DIAG_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        logger.warning("write_interface_diag failed: %s", exc)
